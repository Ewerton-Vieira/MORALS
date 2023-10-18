from MORALS.data_utils import DynamicsDataset
from MORALS.classifier import ClassifierTraining
import argparse
import torch

from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir',help='Directory of config files',type=str,default='config/')
    parser.add_argument('--config',help='Config file inside config_dir',type=str,default='pendulum_lqr.txt')
    parser.add_argument('--verbose',help='Print training output',action='store_true')

    args = parser.parse_args()
    config_fname = args.config_dir + args.config

    with open(config_fname) as f:
        config = eval(f.read())
    
    torch.manual_seed(config["seed"])

    dynamics_dataset = DynamicsDataset(config)

    dynamics_train_size = int(0.8*len(dynamics_dataset))
    dynamics_test_size = len(dynamics_dataset) - dynamics_train_size
    dynamics_train_dataset, dynamics_test_dataset = torch.utils.data.random_split(dynamics_dataset, [dynamics_train_size, dynamics_test_size])
    dynamics_train_loader = DataLoader(dynamics_train_dataset, batch_size=config["batch_size"], shuffle=True)
    dynamics_test_loader = DataLoader(dynamics_test_dataset, batch_size=config["batch_size"], shuffle=True)

    if args.verbose:
        print("Train size: ", len(dynamics_train_dataset))
        print("Test size: ", len(dynamics_test_dataset))

    loaders = {
        'train_dynamics': dynamics_train_loader,
        'test_dynamics': dynamics_test_loader,
    }

    trainer = ClassifierTraining(config, loaders, args.verbose)
    trainer.train(config["epochs"],config["patience"])
    trainer.save_logs()
    trainer.save_model()

if __name__ == "__main__":
    main()