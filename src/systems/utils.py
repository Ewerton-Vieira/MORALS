from src.systems.pendulum import Pendulum
from src.systems.ndpendulum import NdPendulum
from src.systems.cartpole import Cartpole
from src.systems.bistable import Bistable
from src.systems.N_CML import N_CML
from src.systems.leslie_map import Leslie_map
from src.systems.humanoid import Humanoid
from src.systems.trifinger import Trifinger
from src.systems.bistable_rot import Bistable_Rot
from src.systems.unifinger import Unifinger
from src.systems.pendulum3links import Pendulum3links

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
