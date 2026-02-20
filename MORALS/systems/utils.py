from MORALS.systems.pendulum import Pendulum, Pendulum2D
from MORALS.systems.ndpendulum import NdPendulum
from MORALS.systems.cartpole import Cartpole
from MORALS.systems.cartpole_pybullet import CartpolePybullet
from MORALS.systems.cartpole_dmcontrol import CartpoleDmcontrol
from MORALS.systems.bistable import Bistable
from MORALS.systems.N_CML import N_CML
from MORALS.systems.leslie_map import Leslie_map
from MORALS.systems.humanoid import Humanoid
from MORALS.systems.trifinger import Trifinger
from MORALS.systems.bistable_rot import Bistable_Rot
from MORALS.systems.unifinger import Unifinger
from MORALS.systems.pendulum3links import Pendulum3links
from MORALS.systems.basic import Basic
from MORALS.systems.mountain_car import MountainCar
from MORALS.systems.quadrotor2D import Quadrotor2D
from MORALS.systems.quadrotor3D import Quadrotor3D

def get_system(name, dims=10, **kwargs):
    if name == "pendulum":
        system = Pendulum(**kwargs)
    elif name == "pendulum2d":
        system = Pendulum2D(**kwargs)
    elif name == "ndpendulum" and dims is not None:
        system = NdPendulum(dims, **kwargs)
    elif name == "cartpole":
        system = Cartpole(**kwargs)
    elif name == "cartpole_pybullet":
        system = CartpolePybullet(**kwargs)
    elif name == "cartpole_dmcontrol":
        system = CartpoleDmcontrol(**kwargs)
    elif name == "bistable":
        system = Bistable(**kwargs)
    elif name == "N_CML":
        system = N_CML(**kwargs)
    elif name == "leslie_map":
        system = Leslie_map(**kwargs)
    elif name == "humanoid":
        system = Humanoid(**kwargs)
    elif name == "trifinger":
        system = Trifinger(**kwargs)
    elif name == "bistable_rot":
        system = Bistable_Rot(**kwargs)
    elif name == "unifinger":
        system = Unifinger(**kwargs)
    elif name == "pendulum3links":
        system = Pendulum3links(**kwargs)
    elif name == "":
        system = Basic(**kwargs)
    elif name == "basic":
        system = Basic(**kwargs)
    elif name == "mountain_car":
        system = MountainCar(**kwargs)
    elif name == "quadrotor2D" or name == "quadrotor2D_rl":
        system = Quadrotor2D(**kwargs)
    elif name == "quadrotor3D" or name == "quadrotor3D_lqr":
        system = Quadrotor3D(**kwargs)
    else:
        print("That system does not exist!")
        raise NotImplementedError
    return system
