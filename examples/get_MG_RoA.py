from src.systems.utils import get_system
from src.models import *
from src.dynamics_utils import DynamicsUtils
# from MORALS.systems import pendulum
from src.data_utils import DynamicsDataset
# from torch.utils.data import DataLoader
import argparse

import dytop.CMGDB_util as CMGDB_util
import dytop.RoA as RoA
import dytop.Grid as Grid
import dytop.dyn_tools 

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

import numpy as np

def write_experiments(morse_graph, experiment_name, output_dir, name="MG_attractors.txt"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    name = f"{output_dir}/{name}"

    with open(name, "w") as file:
        file.write(experiment_name)

        counting_attractors = 0
        list_attractors = []
        if morse_graph:
            S = set(morse_graph.vertices())
            while len(S) != 0:
                v = S.pop()
                if len(morse_graph.adjacencies(v)) == 0:
                    counting_attractors += 1
                    list_attractors.append(v)

        file.write(f":{list_attractors},{counting_attractors}\n")


def compute_roa(map_graph, morse_graph, lower_bounds, upper_bounds, config, base_name):

    startTime = datetime.now()

    roa = RoA.RoA(map_graph, morse_graph)

    print(f"Time to build the regions of attraction = {datetime.now() - startTime}")

    roa.dir_path = ""

    roa.save_file(base_name)

    if config['low_dims'] == 2:

        # fig, ax = roa.PlotTiles(name_plot=base_name)

        # plt.savefig(base_name, bbox_inches='tight')

        dir_path = os.path.abspath(os.getcwd()) + "/"

        fig, ax = RoA.PlotTiles(lower_bounds, upper_bounds,
                    from_file=base_name, dir_path=dir_path)

        out_pic = roa.dir_path + base_name + "_RoA_"

        plt.savefig(out_pic, bbox_inches='tight')

def main(args, config, experiment_name):

    dyn_utils = DynamicsUtils(config)

    MG_util = CMGDB_util.CMGDB_util()

    system = get_system(config['system'], config['high_dims'])

    sb = args.sub
    number_of_steps = np.ceil(12 / config['step'])  # at least 0.6 seconds in total
    if config['system'] == "discrete_map":
        number_of_steps = 1

    

    subdiv_init = subdiv_min = subdiv_max = sb  # non adaptive proceedure




    # Get the limits
    if type(system.get_true_bounds()) == type(NotImplementedError):  # use data for not implemented bounds
        a = np.loadtxt(os.path.join(config['model_dir'], 'X_min.txt'), delimiter=',').tolist()
        b = np.loadtxt(os.path.join(config['model_dir'], 'X_max.txt'), delimiter=',').tolist()
        system.state_bounds = np.array([a,b]).T

    lower_bounds_original_space = system.get_true_bounds()[:,0].tolist()
    upper_bounds_original_space = system.get_true_bounds()[:,1].tolist()
    print("Bounds for decoded space", lower_bounds_original_space, upper_bounds_original_space)


    # dim_original_space = config['high_dims']
    

    dim_latent_space = config['low_dims']
    lower_bounds = [-1]*dim_latent_space
    upper_bounds = [1]*dim_latent_space
    print("Bounds for encoded space", lower_bounds, upper_bounds)

    grid = Grid.Grid(lower_bounds, upper_bounds, sb)


    # flag_true_bounds = False
    # if type(system.get_true_bounds()) != type(NotImplementedError):  # update bounds using the true_bounds
    #     lower_bounds_original_space = system.get_true_bounds()[:,0].tolist()
    #     upper_bounds_original_space = system.get_true_bounds()[:,1].tolist()
    #     flag_true_bounds = True

    if args.validation_type == 'uniform': # uniform sample (high memory usage)
        grid_original_space = Grid.Grid(lower_bounds_original_space, upper_bounds_original_space, sb + 2)
        original_space_sample = grid_original_space.uniform_sample()
        original_space_sample = system.transform(original_space_sample)

        # if flag_true_bounds:
        #     original_space_sample = system.sample_state(2**(sb + 2))
        #     # original_space_sample = np.array([system.transform(i) for i in original_space_sample])

    elif args.validation_type == 'random': # random sample (ideal for high dim space)
        original_space_sample = system.sample_state(2**(sb + 2))
        # original_space_sample = np.array([system.transform(system.sample_state()) for i in range(2**(sb+2))])

    else: # sample from trajectories (ideal for large trajectory set)
        dataset = DynamicsDataset(config)
        latent_space_sample = dyn_utils.encode(dataset.Xt.numpy(), normalize=False)

    latent_space_sample = dyn_utils.encode(original_space_sample)
    print("data on the latent space", latent_space_sample.shape)
    print(latent_space_sample[1])

    valid_grid_latent_space = grid.valid_grid(latent_space_sample)

    def g(X):
        return dyn_tools.iterate(dyn_utils.f, X, n=number_of_steps).tolist()

    phase_periodic = [False, False]

    K = [1.1 * (1 + args.Lips/100)] * dim_latent_space
    
    def F(rect):
        return MG_util.BoxMapK_valid(g, rect, K, valid_grid_latent_space, grid.point2cell)


    # base name for the output files.
    base_name = f"{config['output_dir']}{args.name_out}"
    
    
    print(base_name)

    base_name = f"{config['output_dir']}/MG"

    MG_util.dir_path = ""
    
    morse_graph, map_graph = MG_util.run_CMGDB(
        subdiv_min, subdiv_max, lower_bounds, upper_bounds, phase_periodic, F, base_name, subdiv_init)


    write_experiments(morse_graph, experiment_name, config['output_dir'])

    if args.RoA:
     
        compute_roa(map_graph, morse_graph, lower_bounds, upper_bounds, config, base_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir',help='Directory of config files',type=str,default='config/')
    parser.add_argument('--config',help='Config file inside config_dir',type=str,default='pendulum_lqr_1K.txt')
    parser.add_argument('--name_out',help='Name of the out file',type=str,default='out_exp')
    parser.add_argument('--RoA',help='Compute RoA',action='store_true')
    parser.add_argument('--sub',help='Select subdivision',type=int,default=14)
    parser.add_argument('--validation_type',help='Select the type of Validation for the lantent discretization',type=str,default='random')
    parser.add_argument('--Lips',help='increase Lipschitz constant by x%',type=int,default=0)

    args = parser.parse_args()
    config_fname = args.config_dir + args.config

    with open(config_fname) as f:
        config = eval(f.read())

    experiment_name = f"{config['experiment']}&{config['num_layers']}&{config['data_dir'][5::]}&{config['step']}&{args.sub}"

    if os.path.exists(config['model_dir']):
        main(args, config, experiment_name)
    else:
        write_experiments(False, experiment_name, config['output_dir'])