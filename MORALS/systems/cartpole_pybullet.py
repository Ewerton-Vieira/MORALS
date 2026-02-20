import numpy as np
from MORALS.systems.system import BaseSystem

class CartpolePybullet(BaseSystem):
    """CartPole system from PyBullet simulation with LQR controller

    State space (parametrization): [x, theta, x_dot, theta_dot]
    - x: cart position (m)
    - theta: pole angle (rad)
    - x_dot: cart velocity (m/s)
    - theta_dot: pole angular velocity (rad/s)

    Embedded space: [x, cos(theta), sin(theta), x_dot, theta_dot] (5D)

    Bounds based on achieved values from dataset (not just initial sampling range)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "cartpole_pybullet"
        self.state_bounds = NotImplementedError
        # Indices of dimensions to normalize (post-transform)
        # Transform: [x, theta, x_dot, theta_dot] -> [x, cos(theta), sin(theta), x_dot, theta_dot]
        # cos(theta), sin(theta) at indices 1,2 are NOT normalized
        # x, x_dot, theta_dot at indices 0,3,4 ARE normalized
        self.normalize_indices = [0, 3, 4]

    def transform(self, x):
        """Transform from parametrization to embedded space

        [x, theta, x_dot, theta_dot] -> [x, cos(theta), sin(theta), x_dot, theta_dot]
        """
        x = np.array(x)
        return np.array([x[:,0], np.cos(x[:,1]), np.sin(x[:,1]), x[:,2], x[:,3]]).T

    def achieved_goal(self, s):
        """Check if state has achieved the goal (stabilized at origin)"""
        diff = np.sqrt(s[1]*s[1] + s[3]*s[3])
        return (diff < 0.1)

    def get_true_bounds(self):
        """Bounds in parametrization space [x, theta, x_dot, theta_dot]

        Based on achieved bounds from dataset:
        - x: [-6.044173, 6.047770] m (rounded to [-6.05, 6.05])
        - theta: [-π, π] rad
        - x_dot: [-7.314936, 7.337680] m/s (rounded to [-7.35, 7.35])
        - theta_dot: [-8.499409, 8.571074] rad/s (rounded to [-8.6, 8.6])
        """
        return np.array([
            [-6.05, 6.05],      # x (cart position)
            [-np.pi, np.pi],    # theta (pole angle)
            [-7.35, 7.35],      # x_dot (cart velocity)
            [-8.6, 8.6]         # theta_dot (pole angular velocity)
        ])
