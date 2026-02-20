import numpy as np
from MORALS.systems.system import BaseSystem

class MountainCar(BaseSystem):
    """Mountain Car system in 2D Euclidean space (position, velocity)

    State space: [position, velocity]
    - Position: horizontal coordinate on mountain terrain [-2.0, 1.0] rad
    - Velocity: rate of change of position [-0.1, 0.1] rad/s

    The terrain is sinusoidal with height = sin(3 * x)
    Goal: reach position π/6 ≈ 0.524 rad with low velocity
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "mountain_car"

        # State bounds: [position, velocity]
        # Position: [-2.0, 1.0] rad
        # Velocity: [-0.1, 0.1] rad/s
        self.state_bounds = np.array([[-2.0, 1.0], [-0.1, 0.1]])

    def transform(self, s):
        """Identity transform - keep data in Euclidean coordinates"""
        return s

    def inverse_transform(self, s):
        """Identity inverse transform - already in Euclidean coordinates"""
        return s

    def get_true_bounds(self):
        """Bounds in parametrization space (same as embedded space for this system)"""
        return self.state_bounds
