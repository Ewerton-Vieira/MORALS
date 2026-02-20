import numpy as np
from MORALS.systems.system import BaseSystem

class Quadrotor2D(BaseSystem):
    """2D Quadrotor system from PyBullet simulation with RL controller

    State space (parametrization): [x, z, theta, x_dot, z_dot, theta_dot]
    - x: horizontal position (m)
    - z: altitude (m)
    - theta: pitch angle (rad)
    - x_dot: horizontal velocity (m/s)
    - z_dot: vertical velocity (m/s)
    - theta_dot: pitch rate (rad/s)

    Embedded space: [x, z, sin(theta), cos(theta), x_dot, z_dot, theta_dot] (7D)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "quadrotor2D"
        self.state_bounds = NotImplementedError
        # Indices of dimensions to normalize (post-transform)
        # sin(theta), cos(theta) at indices 2,3 are NOT normalized
        self.normalize_indices = [0, 1, 4, 5, 6]  # x, z, x_dot, z_dot, theta_dot

    def transform(self, s):
        """Transform from parametrization to embedded space

        [x, z, theta, x_dot, z_dot, theta_dot] -> [x, z, sin(theta), cos(theta), x_dot, z_dot, theta_dot]
        """
        s = np.array(s)
        if s.ndim == 1:
            theta = s[2]
            return np.array([s[0], s[1], np.sin(theta), np.cos(theta), s[3], s[4], s[5]])
        else:
            theta = s[:, 2]
            return np.column_stack([s[:, 0], s[:, 1], np.sin(theta), np.cos(theta),
                                   s[:, 3], s[:, 4], s[:, 5]])

    def get_true_bounds(self):
        """Bounds in parametrization space [x, z, theta, x_dot, z_dot, theta_dot]"""
        return np.array([
            [-1.01, 1.02],       # x
            [0.08, 1.52],        # z
            [-np.pi, np.pi],     # theta
            [-1.22, 1.22],       # x_dot
            [-1.32, 1.11],       # z_dot
            [-12.76, 12.76]      # theta_dot
        ])
