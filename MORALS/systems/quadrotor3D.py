import numpy as np
from MORALS.systems.system import BaseSystem

class Quadrotor3D(BaseSystem):
    """3D Quadrotor system from PyBullet simulation with LQR controller

    State space: [x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r] (13D)
    - x, y, z: position (m)
    - qw, qx, qy, qz: unit quaternion (scalar-first, canonicalized with qw >= 0)
    - x_dot, y_dot, z_dot: linear velocity (m/s)
    - p, q, r: angular velocity in body frame (rad/s)

    No transform is applied - quaternion representation is already suitable.
    Quaternion components are NOT normalized (already unit norm, bounded in [-1, 1]).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "quadrotor3D"
        self.state_bounds = NotImplementedError
        # Indices of dimensions to normalize (quaternion indices 3-6 are NOT normalized)
        # Normalize: positions (0-2), linear vel (7-9), angular vel (10-12)
        self.normalize_indices = [0, 1, 2, 7, 8, 9, 10, 11, 12]

    def transform(self, s):
        """Identity transform - state is already in suitable representation"""
        return np.array(s)

    def get_true_bounds(self):
        """Bounds in state space"""
        return np.array([
            [-1.84, 1.84],       # x
            [-1.84, 1.84],       # y
            [0.06, 3.03],        # z
            [0.0, 1.0],          # qw (canonicalized)
            [-1.0, 1.0],         # qx
            [-1.0, 1.0],         # qy
            [-1.0, 1.0],         # qz
            [-3.22, 3.22],       # x_dot
            [-3.22, 3.22],       # y_dot
            [-3.32, 3.0],        # z_dot
            [-39.62, 38.24],     # p
            [-38.87, 39.11],     # q
            [-39.61, 39.26]      # r
        ])
