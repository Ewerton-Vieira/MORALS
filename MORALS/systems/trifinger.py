import numpy as np 
from MORALS.systems.system import BaseSystem

class Trifinger(BaseSystem):
    def __init__(self,**kwargs):
        self.name = "trifinger"
        self.state_bounds = NotImplementedError
    
    # def get_true_bounds(self):
    #     return NotImplementedError
    
    # def get_bounds(self):
    #     return NotImplementedError