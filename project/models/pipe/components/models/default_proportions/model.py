from pyspark.ml.pipeline import Model
from pyspark.ml.param.shared import HasInputCols
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark import keyword_only
from models.pipe.components.models.default_proportions._params import HasDefaultProportions
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from typing import List, Dict, Union

spark = SparkSession.builder.appName('NPL').getOrCreate()

class DefaultProportionsModel(Model, HasInputCols, HasDefaultProportions, DefaultParamsReadable, DefaultParamsWritable):
    '''
        The `pyspark.ml.pipeline.Model` that replaces categorical values for their respective proportion of defaulted individuals.
        
        Parameters
        ---------
        `inputCols: List[str]
            A list of the categorical features names.
        `defaultProportions`: Dict[str, DataFrame]
            A dictionary mapping the name of the categorical column to a DataFrame storing the proportions.
            
        Method
        -------
        `_transform`: Performs the replacements.
    '''
    @keyword_only
    def __init__(self, inputCols:List[str]=None, defaultProportions:Dict[str,List[Dict[str, Union[str, float]]]]=None):
        super(DefaultProportionsModel, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        
    @keyword_only
    def setParams(self, inputCols:List[str]=None, defaultProportions:Dict[str,List[Dict[str, Union[str, float]]]]=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
    
    def _transform(self, dataset:DataFrame)->DataFrame:
        '''
            Method that exchanges the categorical values for their proportion of defaulted individuals of the training set.
            
            Parameter
            ---------
            `dataset`: DataFrame
                A `pyspark.sql.dataframe.DataFrame` with the project's data.
                
            Returns
            -------
            The DataFrame with the mentioned transformation.
        '''
        x = self.getInputCols()
        defaultProportions = self.getDefaultProportions()
        for c in x:
            df = spark.createDataFrame(defaultProportions[c]) # Converting our list of dicts into a DataFrame.
            dataset = dataset.join(df, on=c, how='left').drop(c)
        return dataset