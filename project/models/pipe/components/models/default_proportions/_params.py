from pyspark.ml.param.shared import Param, Params
from typing import Dict

doc = '''
    Parameter that stores the proportions of defaulted individuals for each X features' categories.
'''
class HasDefaultProportions(Params):
    '''
        A class dedicated to store the categories' proportion of indebted individuals. 
    '''
    defaultProportions:Dict[str, Dict[str, float]] = Param(Params._dummy(), 'defaultProportions', doc)
    
    def __init__(self):
        super(HasDefaultProportions, self).__init__()
    
    def setDefaultProportions(self, value):
        return self._set(defaultProportions=value)
    
    def getDefaultProportions(self):
        return self.getOrDefault(self.defaultProportions)
