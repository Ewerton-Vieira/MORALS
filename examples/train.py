from MORALS.data_utils import DynamicsDataset, LabelsDataset
from MORALS.models import *
from MORALS.training import Training, TrainingConfig

import numpy as np
import scipy
from tqdm import tqdm
import pickle
import argparse
import torch
import os
import shutil

from torch.utils.data import DataLoader


def check_collapse(encoder, dataset):
    
    dim_high = len(dataset[0][0])

    dim_low = len(encoder(dataset[0][0]))

    epsilon = 0.05
    epsilon = np.power(epsilon, 1/dim_low)
    distance = 1.5

    test_freq = int(min(10000, (len(dataset)-dim_low)/dim_low))

    # dynamics_train_dataset = np.random.shuffle(dynamics_train_dataset)
    for test_index in range(test_freq):

        matrix3 = np.array([encoder(dataset[i + test_index * dim_low][0]).detach().numpy() for i in range(dim_low+1)])        
        a = scipy.spatial.distance_matrix(matrix3, matrix3)
        ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
        b = np.where(a == 0, 4, a)
        ind2 = np.unravel_index(np.argmin(b, axis=None), b.shape)

        # check only points in annulus (inner and outter radius) = (epsilon, distance).
        if all([b[ind2] > epsilon, a[ind] < distance]):  
            
            matrix = [encoder(dataset[i + test_index * dim_low][0]).detach().numpy() - encoder(dataset[dim_low+1 + test_index * dim_low][0]).detach().numpy() for i in range(dim_low)]
            # print(b[ind2], "\n", a[ind],"\n", np.linalg.det(matrix)**2,"\n", matrix)
            if np.linalg.det(matrix)**2 > epsilon**2:
                return False
            
    print("\033[91m Collapse\033[00m")
    return True

def main(parser):



    args = parser.parse_args()
    # Support both full path and filename in config_dir
    if os.path.isabs(args.config) or os.path.exists(args.config):
        config_fname = args.config
    else:
        config_fname = os.path.join(args.config_dir, args.config)

    with open(config_fname) as f:
        config = eval(f.read())

    # Construct output directories
    if 'output_dir_name' in config:
        output_dir = f"{config['output_dir_prefix']}/{config['output_dir_name']}"
    elif 'output_dir_prefix' in config:
        output_dir = f"{config['output_dir_prefix']}/{config['num_train_trajectories']}_{config['experiment']}"
    if 'output_dir_prefix' in config:
        config['output_dir'] = output_dir + '/'
        config['model_dir'] = output_dir + '/models/'
        config['log_dir'] = output_dir + '/logs/'

    output_dir = config['output_dir']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(config['model_dir']):
        os.makedirs(config['model_dir'])
    if not os.path.exists(config['log_dir']):
        os.makedirs(config['log_dir'])

    # Save the updated config (with expanded paths) to the output directory
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        f.write(str(config))
    
    torch.manual_seed(config["seed"])

    # Check if we're using indexed dataset splitting (new format)
    if "shuffled_indices_fname" in config.keys():
        print("Using indexed dataset splitting based on shuffled_indices.txt")

        # Load shuffled indices file
        with open(config["shuffled_indices_fname"], 'r') as f:
            shuffled_files = [line.strip() for line in f.readlines()]

        # Load labels file - supports both single-column (just labels) and multi-column formats
        roa_labels_data = np.loadtxt(config["labels_fname"], delimiter=',')
        if roa_labels_data.ndim == 1:
            # Single column: just labels (e.g., shuffled_labels_0.txt)
            labels = roa_labels_data.astype(int)
        else:
            # Multi-column: last column is the label (e.g., roa_labels.txt with state+label)
            labels = roa_labels_data[:, -1].astype(int)

        # Get train/val/test split sizes
        num_train = config.get("num_train_trajectories", 1000)
        num_val = config.get("num_val_trajectories", 100)

        # Split the files and labels
        train_files = shuffled_files[:num_train]
        train_labels = labels[:num_train]

        val_files = shuffled_files[num_train:num_train+num_val]
        val_labels = labels[num_train:num_train+num_val]

        print(f"Training on {len(train_files)} trajectories")
        print(f"Validating on {len(val_files)} trajectories (for early stopping)")
        print(f"Total test set size: {len(shuffled_files) - num_train} trajectories (validation + remaining)")

        # Create datasets with specific file lists
        # During training, use validation set for early stopping
        dynamics_train_dataset = DynamicsDataset(config, file_list=train_files)
        dynamics_test_dataset = DynamicsDataset(config, file_list=val_files)

        dynamics_train_loader = DataLoader(dynamics_train_dataset, batch_size=config["batch_size"], shuffle=True)
        dynamics_test_loader = DataLoader(dynamics_test_dataset, batch_size=config["batch_size"], shuffle=True)

        if "labels_fname" in config.keys():
            labels_train_dataset = LabelsDataset(config, file_list=train_files, labels_list=train_labels)
            labels_test_dataset = LabelsDataset(config, file_list=val_files, labels_list=val_labels)

            labels_train_loader = DataLoader(labels_train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=labels_train_dataset.collate_fn)
            labels_test_loader = DataLoader(labels_test_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=labels_test_dataset.collate_fn)
        else:
            labels_train_loader = dynamics_train_loader
            labels_test_loader = dynamics_test_loader
    else:
        # Legacy behavior: random splitting
        print("Using legacy random dataset splitting")

        dynamics_dataset = DynamicsDataset(config)

        np.random.seed(config["seed"])

        dynamics_train_size = int(args.splitting_data*len(dynamics_dataset))
        dynamics_test_size = len(dynamics_dataset) - dynamics_train_size
        dynamics_train_dataset, dynamics_test_dataset = torch.utils.data.random_split(dynamics_dataset, [dynamics_train_size, dynamics_test_size])
        dynamics_train_loader = DataLoader(dynamics_train_dataset, batch_size=config["batch_size"], shuffle=True)
        dynamics_test_loader = DataLoader(dynamics_test_dataset, batch_size=config["batch_size"], shuffle=True)

        if "labels_fname" in config.keys():
            labels_dataset = LabelsDataset(config)
            labels_train_size = int(0.8 * len(labels_dataset))
            labels_test_size = len(labels_dataset) - labels_train_size
            labels_train_dataset, labels_test_dataset = torch.utils.data.random_split(labels_dataset, [labels_train_size, labels_test_size])
            labels_train_loader = DataLoader(labels_train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=labels_dataset.collate_fn)
            labels_test_loader = DataLoader(labels_test_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=labels_dataset.collate_fn)
        else:
            labels_train_loader = dynamics_train_loader
            labels_test_loader = dynamics_test_loader


    if args.verbose:
        print("Train size: ", len(dynamics_train_dataset))
        print("Test size: ", len(dynamics_test_dataset))

    # loaders = {
    #     'train_dynamics': dynamics_train_loader,
    #     'test_dynamics': dynamics_test_loader
    # }
    # if "labels_fname" in config.keys(): 
    loaders = {
        'train_dynamics': dynamics_train_loader,
        'test_dynamics': dynamics_test_loader,
        'train_labels': labels_train_loader,
        'test_labels': labels_test_loader
    }

    trainer = Training(config, loaders, args.verbose)
    experiment = TrainingConfig(config['experiment'])

    for i,exp in enumerate(experiment):
        if args.verbose:
            print("Training loss weights: ", exp)
        trainer.train(config["epochs"], config["patience"], exp)
        trainer.save_logs(suffix =str(i))
        trainer.reset_losses()


    if args.collapse:
        check_collapse(trainer.encoder, dynamics_train_dataset)
    trainer.save_models()    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',help='Path to config file (can be full path or filename in config_dir)',type=str,required=True)
    parser.add_argument('--config_dir',help='Directory of config files (only used if --config is not a full path)',type=str,default='config/')
    parser.add_argument('--verbose',help='Print training output',action='store_true', default=True)
    parser.add_argument('--collapse',help='Check for collapse',action='store_true')
    parser.add_argument('--splitting_data',help='Set proportion to split train-test data',type=float, default=0.998)

    main(parser)