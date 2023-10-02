import numpy as np 
from src.systems.system import BaseSystem


class Bistable_Rot(BaseSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "bistable"
        self.state_bounds = np.array([[-2, 2]]*2)

        # theta = np.radians(90)
        # c, s = np.cos(theta), np.sin(theta)
        # self.R = np.array(((c, -s), (s, c)))
        
    def R(self, theta=np.radians(90)):
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c, -s), (s, c)))
    
    def f(self,s):
        return [np.arctan(2*s[0])] + [s[i]/2 for i in range(1, len(s))]

    def transform(self,x):
        "clockwise rotation"
        return np.matmul(x, self.R(np.pi*np.linalg.norm(x)))
        # x=np.array(x)
        # for i in range(len(x)):
        #     x[i,:] = np.matmul(x[i,:], self.R(np.linalg.norm(x[i,:])))
        # return x


        # return np.matmul(x, self.R)

    # def get_true_bounds(self):
    #     return NotImplementedError
    
    # def get_bounds(self):
    #     return NotImplementedError