import numpy as np 
from MORALS.systems.system import BaseSystem

class N_CML(BaseSystem):
    def __init__(self, dims=3, **kwargs):
        super().__init__(**kwargs)
        self.N = dims
        self.delta=0.06
        self.a=-.785
        self.epsilon=-.024
        self.state_bounds = np.array([[-1.1, 1.1]]*self.N)
        self.name = f"{self.N}_CML"
        

    def f_a(self, s, a):
            return 1 - a*(s**2)
    
    def F_i(self, s, i, delta, a, epsilon):
        if i == 0:
            return (1-epsilon)*self.f_a(s[i], a) + 0.5*(epsilon + delta)*self.f_a(s[i+1], a)
        elif i == self.N-1:
            return (1-epsilon)*self.f_a(s[i], a) + 0.5*(epsilon - delta)*self.f_a(s[i-1], a)
        else:
            return (1-epsilon)*self.f_a(s[i], a) + 0.5*(epsilon - delta)*self.f_a(s[i-1], a) + 0.5*(epsilon + delta)*self.f_a(s[i+1], a)
    
    def f(self,s):
        return [self.F_i(s, i, self.delta, self.a, self.epsilon) for i in range(self.N)]

    # def get_true_bounds(self):
    #     return self.state_bounds
    
    # def get_bounds(self):
    #     return NotImplementedError