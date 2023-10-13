import numpy as np 
from MORALS.systems.system import BaseSystem

class Leslie_map(BaseSystem):
    def __init__(self, dims=10, **kwargs):
        super().__init__(**kwargs)
        self.params = [19.6, 23.68] # Parameter for Leslie map   
        self.N = dims
        self.state_bounds = np.array([[-0.001, -0.001], [90.0, 70.0]]+[[-1.1, 1.1]]*(self.N-2))
        self.name = "leslie_map"
        

    def f(self,s):
        return [(self.params[0] * s[0] + self.params[1] * s[1]) * np.exp(-0.1 * (s[0] + s[1])), 0.7 * s[0]] + [0.25*s[i] for i in range(2, len(s))]

    # def get_true_bounds(self):
    #     return self.state_bounds
    
    # def get_bounds(self):
    #     return NotImplementedError