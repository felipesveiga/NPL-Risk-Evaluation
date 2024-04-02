from pyspark.ml.param.shared import HasInputCol, HasInputCols, HasOutputCol, HasOutputCols, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark import keyword_only
from pyspark.ml import Transformer
from typing import List

class _BaseTransformer(Transformer, HasInputCol, HasInputCols, HasOutputCol, HasOutputCols, DefaultParamsReadable, DefaultParamsWritable):
    '''
        A class that provide the basic functionalities for the project's custom transformers.
        
        If you consider that your object will require even more customization, you can just overwrite any of the default methods when creating it. Also,
        remember that any extra parameter must be set with a `self._setDefault` method right in the `__init__` function, mentioned in the `self.setParams`
        method and own a getter function.
        
        Lastly, don't forget to define the `_transform` function!
        
        Parameters
        ----------
        `inputCol`: str
            The name of the input column.
        `inputCols`: List[str]
            List containing the name of input columns.
        `outputCol`: str
            The name of the output column.
        `outputCols`: List[str]
            List containing the name of output columns.
        
        References
        ----------
        https://medium.com/@zeid.zandi/utilizing-the-power-of-pyspark-pipelines-in-data-science-projects-benefits-and-limitations-2-2-9063e4bebd05
        https://www.crowdstrike.com/blog/deep-dive-into-custom-spark-transformers-for-machine-learning-pipelines/
    '''
    @keyword_only
    def __init__(self, inputCol:str=None, inputCols:List[str]=None, outputCol:str=None, outputCols:List[str]=None, *args)->None:
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        
    @keyword_only
    def setParams(self, inputCol=None, inputCols=None, outputCol=None, outputCols=None): 
        kwargs = self._input_kwargs                                                          
        return self._set(**kwargs)
    
    def setInputCol(self, value):
        return self.setParams(inputCol=value)
    
    def setInputCols(self, value):
        self.setParams(inputCols=value)
    
    def setOutputCol(self, value):
        return self.setParams(outputCol=value)
    
    def setOutputCols(self, value):
        return self.setParams(outputCols=value)