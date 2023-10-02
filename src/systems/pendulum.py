import numpy as np 
from src.systems.system import BaseSystem

class Pendulum(BaseSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "pendulum"

        self.l = 0.5
        self.state_bounds = np.array([[-self.l, self.l], [-self.l, self.l], [-2*self.l*np.pi, 2*self.l*np.pi], [-2*self.l*np.pi, 2*self.l*np.pi]])
        
    def transform(self,s):
        x = self.l * np.sin(s[:,0])
        y = self.l * np.cos(s[:,0])
        xdot = self.l * np.cos(s[:,0]) * s[:,1]
        ydot = -self.l * np.sin(s[:,0]) * s[:,1]
        return np.array([x,y,xdot,ydot]).T
    
    # def transform(self,s):
    #     x = self.l * np.sin(s[0])
    #     y = self.l * np.cos(s[0])
    #     xdot = self.l * np.cos(s[0]) * s[1]
    #     ydot = -self.l * np.sin(s[0]) * s[1]
    #     return np.array([x,y,xdot,ydot])
    
    # def inverse_transform(self,s):
    #     theta = np.arctan2(s[0],s[1])
    #     thetadot = -(-s[1] * s[2] + s[0] * s[3]) / (s[0]*s[0] + s[1]*s[1])
    #     return np.array([theta,thetadot])
     
    def inverse_transform(self,s):
        theta = np.arctan2(s[:,0],s[:,1])
        thetadot = -(-s[:,1] * s[:,2] + s[:,0] * s[:,3]) / (s[:,0]*s[:,0] + s[:,1]*s[:,1])
        return np.array([theta,thetadot])
    
    def get_true_bounds(self):
        return np.array([[-np.pi, np.pi], [-2*np.pi, 2*np.pi]])