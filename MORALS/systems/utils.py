from MORALS.systems.pendulum import Pendulum
from MORALS.systems.ndpendulum import NdPendulum
from MORALS.systems.cartpole import Cartpole
from MORALS.systems.bistable import Bistable
from MORALS.systems.N_CML import N_CML
from MORALS.systems.leslie_map import Leslie_map
from MORALS.systems.humanoid import Humanoid
from MORALS.systems.trifinger import Trifinger
from MORALS.systems.bistable_rot import Bistable_Rot
from MORALS.systems.unifinger import Unifinger
from MORALS.systems.pendulum3links import Pendulum3links

def get_system(name, dims=10, **kwargs):
    if name == "pendulum":
        system = Pendulum(**kwargs)
    elif name == "ndpendulum" and dims is not None:
        system = NdPendulum(dims, **kwargs)
    elif name == "cartpole":
        system = Cartpole(**kwargs)
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
    else:
        print("That system does not exist!")
        raise NotImplementedError
    return system
