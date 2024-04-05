from pyspark.sql import DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from models.pipe.components.transformers.base import _BaseTransformer

class ImputeOccupation(_BaseTransformer):
    '''
        A Transformer encharged of imputing proper OCCUPATION_TYPE values for rows that are NULL in such column.  
        
        Parameters
        ----------
        `inputCol`: str
            The name of the input column.
            
        Method
        ------
        `_transform`: Applies the mentioned imputation process,
    '''
    @staticmethod
    @udf(returnType=StringType())
    def __impute_occupation(col_occupation:str, col_days_employed:int)->str:
        '''
        Assigns new categories to the OCCUPATION_TYPE column, if it is null. 
        
        In case the row's DAYS_EMPLOYED shows that the client is currently employed, we impute 'UNDEFINED'; otherwise, we insert 'UNEMPLOYED'.
        
        Parameters
        ----------
        `col_occupation`: str
            The row's OCCUPATION_TYPE value.
        `col_days_employed`: int
            The row's DAYS_EMPLOYED value.
        
        Returns
        -------
        The row's OCCUPATION_TYPE treated value.
        '''
        # 'Undefined' profession logic.
        if (col_occupation is None) and (col_days_employed<=0):
            return 'Undefined'

        # 'Unemployed' logic.
        elif (col_occupation is None) and (col_days_employed>0):
            return 'Unemployed'
        
        # Otherwise, it outputs the current value.
        else:
            return col_occupation
    
    def _transform(self, dataset:DataFrame)->DataFrame:
        '''
            Performs the new occupation categories imputation.
            
            Parameter
            ---------
            `dataset`: `pyspark.sql.DataFrame`
                The project's independent variables.
                
            Returns
            -------
            The DataFrame with the treated 'OCCUPATION_TYPE' column.
        '''
        x = self.getInputCol()
        dataset = dataset.withColumn(x, self.__impute_occupation(x, 'DAYS_EMPLOYED'))
        return dataset