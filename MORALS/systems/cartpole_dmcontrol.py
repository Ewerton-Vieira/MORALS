import numpy as np
from MORALS.systems.system import BaseSystem

class CartpoleDmcontrol(BaseSystem):
    """CartPole system from DeepMind Control Suite with LQR controller

    State space (parametrization): [x, theta, x_dot, theta_dot]
    - x: cart position (m)
    - theta: pole angle (rad, UNWRAPPED - can exceed ±π)
    - x_dot: cart velocity (m/s)
    - theta_dot: pole angular velocity (rad/s)

    Embedded space: [x, cos(theta), sin(theta), x_dot, theta_dot] (5D)

    Key difference from other cartpole systems: theta is UNWRAPPED, allowing
    the pole to perform multiple full rotations. Max observed ~±130 rad (~20 rotations).

    Bounds based on achieved values from dataset
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "cartpole_dmcontrol"
        self.state_bounds = NotImplementedError

    def transform(self, x):
        """Transform from parametrization to embedded space

        [x, theta, x_dot, theta_dot] -> [x, cos(theta), sin(theta), x_dot, theta_dot]

        Note: cos/sin wraps theta naturally, so unwrapped theta values work correctly
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
        - x: [-2.069982, 2.064363] m (rounded to [-2.1, 2.1])
        - theta: [-130.202002, 130.823991] rad (rounded to [-131, 131])
        - x_dot: [-8.524513, 8.624525] m/s (rounded to [-8.7, 8.7])
        - theta_dot: [-18.588253, 18.275148] rad/s (rounded to [-18.6, 18.6])

        Note: theta bounds represent unwrapped angles (~20 full rotations)
        """
        return np.array([
            [-2.1, 2.1],      # x (cart position)
            [-131, 131],      # theta (pole angle, UNWRAPPED)
            [-8.7, 8.7],      # x_dot (cart velocity)
            [-18.6, 18.6]     # theta_dot (pole angular velocity)
        ])
