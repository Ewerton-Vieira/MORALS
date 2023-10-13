import numpy as np 
from MORALS.systems.system import BaseSystem

class Bistable(BaseSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "bistable"
        self.state_bounds = np.array([[-2, 2]]*10)
        

    def f(self,s):
        return [np.arctan(4*s[0])] + [s[i]/4 for i in range(1, len(s))]

    # def get_true_bounds(self):
    #     return NotImplementedError
    
    # def get_bounds(self):
    #     return NotImplementedError