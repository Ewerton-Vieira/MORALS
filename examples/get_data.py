import sys 
import os
from tqdm import tqdm
from MORALS.systems.utils import get_system
import numpy as np

np.set_printoptions(suppress=True)
np.random.seed(101193)

import argparse

def is_in_att(x):
    x = np.array(x)
    if np.linalg.norm(x - np.array([1.39]+[0]*9)) < 0.1:
        return True 
    else:
        return False

def sample_points(lower_bounds, upper_bounds, num_pts):
    # Sample num_pts in dimension dim, where each
    # component of the sampled points are in the
    # ranges given by lower_bounds and upper_bounds
    dim = len(lower_bounds)
    X = np.random.uniform(lower_bounds, upper_bounds, size=(num_pts, dim))
    return X

def grid_points(lower_bounds, upper_bounds, num_pts):
    # Returns a grid of initial conditions,
    assert len(lower_bounds) == 2, "Currently, grid points only works for dimensionality 2!"
    X = []
    num = int(np.sqrt(num_pts))
    dim1 = np.linspace(lower_bounds[0],upper_bounds[0],num)
    dim2 = np.linspace(lower_bounds[1],upper_bounds[1],num)
    for i in range(dim1.shape[0]):
        for j in range(dim2.shape[0]):
            X.append([dim1[i],dim2[j]])
    return np.vstack(X)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', help='Trajectory length', type=float, default=0.5)
    parser.add_argument('--time_step', help='Time step', type=float, default=0.1)
    parser.add_argument('--num_trajs', help='Number of trajectories', type=int, default=1000)
    parser.add_argument('--save_dir', help='Save directory', type=str, default='/data/bistable')
    parser.add_argument('--system', help='Select the system', type=str, default='bistable')
    parser.add_argument('--sample', help='Samples initial conditions instead of on a grid', type=str, default='random')
    parser.add_argument('--labels', help='Get labels', action='store_true')
    parser.add_argument('--horizon', help='Time horizon to get the labels', type=type, default=20)
    parser.add_argument('--save_labels', help='Save labels file.txt', type=str, default='/data/bistable_success')

    args = parser.parse_args()
    
    system = get_system(args.system)

    num_trajs = args.num_trajs
    num_steps = int(args.time/args.time_step)

    true_bounds = system.get_true_bounds()

    dim = len(true_bounds) // 2

    
    if args.sample == 'random':
        lower_bounds = true_bounds[:,0]
        upper_bounds = true_bounds[:,1]
        X = sample_points(lower_bounds, upper_bounds, num_trajs)
    else:
        lower_bounds = true_bounds[0]
        upper_bounds = true_bounds[1]
        X = grid_points(lower_bounds, upper_bounds, num_trajs)

    if args.system == "bistable" or args.system == "bistable_rot":

        f = system.f

    else:
        NotImplementedError


    # Create the data directory if it doesn't exist
    # save_dir = args.save_dir

    save_dir = os.getcwd() + args.save_dir
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    counter = 0
    labels = str()
    for x in tqdm(X):
        # Get the full trajectory
        traj = [x]
        # traj = [system.transform(x)]
        state_temp = x
        for k in range(num_steps):
            state_temp = f(state_temp)
            traj.append(state_temp)
            # traj.append(system.transform(state_temp))
        
        traj = np.array(traj)
        np.savetxt(f"{save_dir}/{counter}.txt",traj,delimiter=",")

        if args.labels == True:
            for k in range(args.horizon * num_steps):
                state_temp = f(state_temp)
            
            labels += f"{counter}.txt,"
            if is_in_att(state_temp):
                labels += "1\n"
            else:
                labels += "0\n"
        counter += 1 

    if args.labels == True:
        with open(f"{os.getcwd()}{args.save_labels}.txt", 'w') as file:
            file.write(labels)

