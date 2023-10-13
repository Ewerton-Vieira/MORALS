import numpy as np 
from MORALS.systems.system import BaseSystem

class Pendulum3links(BaseSystem):
    def __init__(self,**kwargs):
        self.name = "pendulum3links"
        self.state_bounds = NotImplementedError
    
    # def get_true_bounds(self):
    #     return NotImplementedError
    
    # def get_bounds(self):
    #     return NotImplementedError