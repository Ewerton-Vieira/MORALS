from MORALS.systems.system import BaseSystem

class Basic(BaseSystem):
    def __init__(self,**kwargs):
        self.name = "basic"
        self.state_bounds = NotImplementedError
    
    # def get_true_bounds(self):
    #     return NotImplementedError
    
    # def get_bounds(self):
    #     return NotImplementedError