import numpy as np 
from src.systems.system import BaseSystem
import matplotlib.pyplot as plt

# fix the transform

class NdPendulum(BaseSystem):
    def __init__(self, dims=9, **kwargs):
        super().__init__(**kwargs)
        self.name = "ndpendulum"

        # self.state_bounds = np.array([[-np.pi, np.pi], [-2*np.pi, 2*np.pi]])
        self.state_bounds = NotImplementedError
        
        
        # Find smallest N such that dims < N*N
        N = int(np.ceil(np.sqrt(dims)))
        # Get a grid of N*N points
        th = np.linspace(-np.pi, np.pi, N)
        thdot = np.linspace(-2*np.pi, 2*np.pi, N)

        index = np.arange(N*N)

        # set random choice, it might not work with trainning.
        # np.random.seed(dims)
        # np.random.shuffle(index)
        # index = np.random.choice(N*N, dims, replace=False)
        

        index = index[0:dims]

        xx, yy = np.meshgrid(th, thdot)

        # plt.plot(xx, yy, marker='o', color='k', linestyle='none')
        # plt.show()

        self.centers = [[xx[k%N, (k//N)%N],
        yy[k%N, (k//N)%N]]for k in index]

        self.centers = np.array(self.centers)

        # plt.plot(self.centers[:,0], self.centers[:,1], marker='o', color='k', linestyle='none')
        # plt.show()
        # self.centers = np.random.choice(np.array(np.meshgrid(th, thdot)).T.reshape(-1, 2), size=dims, replace=False)

        self.l = 0.5
    
    # it needs to be vectorized
    def transform(self,s):
        pt = np.zeros((self.centers.shape[0],))
        for i in range(self.centers.shape[0]):
            pt[i] = np.exp(-np.linalg.norm(s-self.centers[i])**2)
        return pt

    def inverse_transform(self,s):
        return s
    
    def get_true_bounds(self):
        return np.array([[-np.pi, np.pi], [-2*np.pi, 2*np.pi]])
    
    # def get_bounds(self):
    #     return NotImplementedError