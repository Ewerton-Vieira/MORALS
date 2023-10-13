import numpy as np

"""For euclidian: space get_bounds=get_true_bounds. For manifold: get_true_bounds=parametrization 
and get_bounds have to be defined as a box that contain the manifold 
(the box can be defined or obtained from data)"""

class BaseSystem:
    def __init__(self, **kwargs):
        self.name = "base_system"

        self.state_bounds = NotImplementedError
    
    def f(self,s):
        return s

    # def sample_state(self):
    #     return np.random.uniform(self.state_bounds[:,0], self.state_bounds[:,1])

    def sample_state(self, num_pts):
        sample_ = np.random.uniform(self.get_true_bounds()[:,0], self.get_true_bounds()[:,1], size=(num_pts, self.dimension()))
        return self.transform(sample_)
    
    def get_bounds(self): # bounds on the embedded space
        return self.state_bounds

    def get_true_bounds(self): # bounds of the parametrization
        return self.get_bounds()
    
    def dimension(self): # dimension of the manifold
        return self.get_true_bounds().shape[0]
    
    def transform(self, s):
        return s
    
    def inverse_transform(self, s):
        return s