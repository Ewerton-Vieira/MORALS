from src.data_utils import DynamicsDataset, LabelsDataset
from src.models import *
from src.training import Training, TrainingConfig

import numpy as np
import scipy
from tqdm import tqdm
import pickle
import argparse

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir',help='Directory of config files',type=str,default='config/')
    parser.add_argument('--config',help='Config file inside config_dir',type=str,default='bistable.txt')
    parser.add_argument('--verbose',help='Print training output',action='store_true')
    parser.add_argument('--collapse',help='Check for collapse',action='store_true')


    args = parser.parse_args()
    config_fname = args.config_dir + args.config

    with open(config_fname) as f:
        config = eval(f.read())
    
    torch.manual_seed(config["seed"])
    
    dynamics_dataset = DynamicsDataset(config)
    
    
    np.random.seed(config["seed"])

    dynamics_train_size = int(0.8*len(dynamics_dataset))
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
    main()