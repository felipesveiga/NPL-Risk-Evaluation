from pyspark.ml.pipeline import Estimator
from pyspark.ml.param.shared import HasInputCols
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark import keyword_only
from models.pipe.components.models.default_proportions.model import DefaultProportionsModel
from models.pipe.components.models.default_proportions._params import HasDefaultProportions
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType
from pyspark.sql.dataframe import DataFrame
from typing import List, Dict

class DefaultProportions(Estimator, HasInputCols, HasDefaultProportions, DefaultParamsReadable, DefaultParamsWritable):
    '''
        An Estimator encharged for measuring the categorical values proportions of defaulted individuals.
    
        Parameter
        ---------
        `inputCols`: List[str]
            A list with the categorical columns names.
            
        Method
        ------
        `_fit`: Measures the proportions and stores them in a DataFrame.
        
        References
        ----------
        https://stackoverflow.com/questions/37270446/how-to-create-a-custom-estimator-in-pyspark
    '''
    @keyword_only
    def __init__(self, inputCols:List[str]=None):
        super(DefaultProportions, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
    
    @keyword_only
    def setParams(self, inputCols:List[str]=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
        
    def setInputCols(self, value:List[str]):
        return self.setParams(inputCols=value)
    
    def setPredictionCol(self, value:str):
        return self.setParams(predictionCol=value)
    
    @staticmethod
    def __measure_proportions(dataset:DataFrame, input_cols:List[str], target:str)->Dict[str, DataFrame]:
        '''
            Responsible for quantifying the categories' proportions of indebted individuals. They will be stored in a `pyspark.sql.dataframe.DataFrame`
            object. Each dataframe is kept in the `defaultedProportions` dictionary, which maps them to the categorical column's name.
            
            Paramters
            ---------
            `dataset`: DataFrame
                A `pyspark.sql.dataframe.DataFrame` with the project's data.
            `inputCols`: List[str]
                A list with the categorical columns names.    
            `target`: str
                The `dataset`'s target name.
            
            Returns
            -------
            A dictionary mapping the categorical columns' names to the DataFrame that stores the classes proportions.
        '''
        defaultProportions = {}
        len_df = dataset.count()
        for c in input_cols:
            col_prop = c+'_NPL_PROP'
            df_gb = dataset.groupBy([c, target]).count()
            df_props = (df_gb
                        .where(f'{target}==1')
                        .withColumn(col_prop, (col('count')/len_df)
                        .cast(FloatType()))#DecimalType(8,7)))
                        .select([c, col_prop]))
            defaultProportions[c] = list(map(lambda row: row.asDict(), df_props.collect())) # DF to list to able saving the class.
        return defaultProportions
            
    def _fit(self, dataset:DataFrame, target:str='NPL')->DefaultProportionsModel:
        '''
            Measures the proportions of indebted individuals from each dataset's categorical columns.
            
            Parameters
            ---------
            `dataset`: DataFrame
                A `pyspark.sql.dataframe.DataFrame` with the project's data.   
            `target`: str
                The `dataset`'s target name.
                
            Returns
            -------
            A fitted DefaultProportionsModel ready for transforming the dataset.
        '''
        x = self.getInputCols()
        defaultProportions = self.__measure_proportions(dataset, x, target)
        return DefaultProportionsModel(inputCols=x, defaultProportions=defaultProportions)