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

        self.encoder = torch.load(os.path.join(config['model_dir'], 'encoder.pt'), map_location=self.device, weights_only=False)
        self.decoder = torch.load(os.path.join(config['model_dir'], 'decoder.pt'), map_location=self.device, weights_only=False)
        self.dynamics = torch.load(os.path.join(config['model_dir'], 'dynamics.pt'), map_location=self.device, weights_only=False)

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.dynamics.to(self.device)

        # Keep CPU copies for small batch inference (avoids GPU transfer overhead)
        import copy
        self.dynamics_cpu = copy.deepcopy(self.dynamics).to('cpu')
        self.dynamics_cpu.eval()

    def f(self, z):
        # This function takes as input a latent state and returns the next latent state
        with torch.no_grad():
            z = torch.tensor(z, dtype=torch.float32).to(self.device)
            return self.dynamics(z).cpu().numpy()

    def f_iterate(self, z, n_steps=1):
        # Optimized version: iterate n_steps without converting back to numpy each step
        # Supports both single points and batched inputs
        with torch.no_grad():
            z = torch.tensor(z, dtype=torch.float32).to(self.device)
            for _ in range(int(n_steps)):
                z = self.dynamics(z)
            return z.cpu().numpy()

    def f_iterate_batch(self, z_batch, n_steps=1):
        """
        Batched version for multiple points. Input shape: (N, dim)
        Returns shape: (N, dim)
        Uses CPU for small batches to avoid GPU transfer overhead.
        """
        batch_size = z_batch.shape[0]

        # For small batches (<64), CPU is faster due to GPU transfer overhead
        if batch_size < 64:
            with torch.no_grad():
                z = torch.from_numpy(z_batch.astype(np.float32))  # CPU tensor, no copy if contiguous
                for _ in range(int(n_steps)):
                    z = self.dynamics_cpu(z)
                return z.numpy()
        else:
            with torch.no_grad():
                z = torch.tensor(z_batch, dtype=torch.float32).to(self.device)
                for _ in range(int(n_steps)):
                    z = self.dynamics(z)
                return z.cpu().numpy()

    def create_batched_evaluator(self, n_steps, max_batch_size=64):
        """
        Create an optimized evaluator with pre-allocated GPU tensors.
        Returns a function that takes numpy array and returns numpy array.
        """
        # Pre-allocate input tensor on GPU
        input_buffer = torch.zeros(max_batch_size, self.encoder.low_dims if hasattr(self.encoder, 'low_dims') else 2,
                                   dtype=torch.float32, device=self.device)

        def evaluate(z_batch):
            batch_size = z_batch.shape[0]
            with torch.no_grad():
                # Copy to pre-allocated buffer (faster than creating new tensor)
                input_buffer[:batch_size].copy_(torch.from_numpy(z_batch))
                z = input_buffer[:batch_size]
                for _ in range(int(n_steps)):
                    z = self.dynamics(z)
                return z.cpu().numpy()

        return evaluate

    def encode(self, x, normalize=True, batch_size=1_000_000):
        # This function takes as input a raw state (un-normalized)
        # and returns the latent state
        # Batched to avoid OOM for large inputs
        if normalize:
            x = (x - self.X_min) / (self.X_max - self.X_min)

        n_samples = x.shape[0]
        if n_samples <= batch_size:
            # Small enough to process at once
            with torch.no_grad():
                x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
                return self.encoder(x_tensor).cpu().numpy()
        else:
            # Process in batches
            results = []
            for i in range(0, n_samples, batch_size):
                batch = x[i:i + batch_size]
                with torch.no_grad():
                    batch_tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)
                    results.append(self.encoder(batch_tensor).cpu().numpy())
            return np.concatenate(results, axis=0)

    def decode(self, x):
        # This function takes as input a latent state
        # and returns the raw state (un-normalized)
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
            x = self.decoder(x).cpu().numpy()
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
