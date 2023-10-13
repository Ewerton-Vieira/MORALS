import numpy as np
import torch

from MORALS.systems.utils import get_system
from MORALS.models import *

import os

class DynamicsUtils:
    def __init__(self, config):
        self.system = get_system(config['system'])

        assert os.path.exists(config['model_dir']), "model expected"
                            

        if config["use_limits"]:
            raise NotImplementedError
        else:
            self.X_min = np.loadtxt(os.path.join(config['model_dir'], 'X_min.txt'), delimiter=',')
            self.X_max = np.loadtxt(os.path.join(config['model_dir'], 'X_max.txt'), delimiter=',')

        self.encoder  = Encoder(config)
        self.decoder  = Decoder(config)
        self.dynamics = LatentDynamics(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = torch.load(os.path.join(config['model_dir'], 'encoder.pt'), map_location=self.device)
        self.decoder = torch.load(os.path.join(config['model_dir'], 'decoder.pt'), map_location=self.device)
        self.dynamics = torch.load(os.path.join(config['model_dir'], 'dynamics.pt'), map_location=self.device)

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.dynamics.to(self.device)

    def f(self, z):
        # This function takes as input a latent state and returns the next latent state
        z = torch.tensor(z, dtype=torch.float32).to(self.device)
        return self.dynamics(z).detach().to('cpu').numpy()

    def encode(self, x, normalize=True):
        # This function takes as input a raw state (un-normalized)
        # and returns the latent state
        if normalize:
            x = (x - self.X_min) / (self.X_max - self.X_min)
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        return self.encoder(x).detach().to('cpu').numpy()

    def decode(self, x):
        # This function takes as input a latent state
        # and returns the raw state (un-normalized)
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = self.decoder(x).detach().to('cpu').numpy()
        return x * (self.X_max - self.X_min) + self.X_min

class EnsembleDynamics:
    def __init__(self, configs):
        self.dynamics = [DynamicsUtils(config) for config in configs]
    
    def f(self, z):
        # This function takes as input a latent state and returns the next latent state
        all_dyn = [dynamics.f(z) for dynamics in self.dynamics]
        mean_dyn = np.mean(all_dyn, axis=0)
        std_dyn = np.std(all_dyn, axis=0)
        return (mean_dyn, std_dyn)

    def encode(self, x):
        # This function takes as input a raw state (un-normalized)
        # and returns the latent state
        all_enc = [dynamics.encode(x) for dynamics in self.dynamics]
        mean_enc = np.mean(all_enc, axis=0)
        std_enc = np.std(all_enc, axis=0)
        return (mean_enc, std_enc)

    def decode(self, x):
        print("Decoding ensemble dynamics is not implemented.")
        raise NotImplementedError
