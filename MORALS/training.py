import torch
import os
import pickle
import numpy as np
from torch import nn
from tqdm import tqdm
from MORALS.models import *

class TrainingConfig:
    def __init__(self, weights_str):
        self.weights_str = weights_str
        self.parse_config()
    
    def parse_config(self):
        ids = self.weights_str.split('_')
        self.weights = []
        for _, id in enumerate(ids):
            w = [float(e) for e in id.split('x')[:-1]]
            if len(w) == 4:
                w.append(0.0)  # default fixed-point weight = 0
                w.append(0.0)  # default clustering weight = 0
            elif len(w) == 5:
                w.append(0.0)  # default clustering weight = 0
            elif len(w) != 6:
                print("Expected 4, 5, or 6 values per training config, got ", len(w))
                raise ValueError
            self.weights.append(w)
    
    def __getitem__(self, key):
        return self.weights[key]
    
    def __len__(self):
        return len(self.weights)

class LabelsLoss(nn.Module):
    def __init__(self, reduction='mean', scale=100.0, loss_type='sigmoid', margin=1.0):
        super(LabelsLoss, self).__init__()
        self.reduction = reduction
        self.scale = scale
        self.loss_type = loss_type
        self.margin = margin

    def forward(self, x, y):
        pairwise_distance = torch.linalg.vector_norm(x - y, ord=2, dim=1)

        if self.loss_type == 'margin':
            # Hinge loss: pushes pairs apart until distance >= margin
            loss = torch.clamp(self.margin - pairwise_distance, min=0)
        else:
            # Original sigmoid loss
            loss = torch.sigmoid(-self.scale * pairwise_distance)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError("Invalid reduction type")

class Training:
    def __init__(self, config, loaders, verbose):
        self.encoder = Encoder(config)
        self.dynamics = LatentDynamics(config)
        self.decoder = Decoder(config)

        self.verbose = bool(verbose)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)

        self.encoder.to(self.device)
        self.dynamics.to(self.device)
        self.decoder.to(self.device)

        self.dynamics_train_loader = loaders['train_dynamics']
        self.dynamics_test_loader = loaders['test_dynamics']
        self.labels_train_loader = loaders['train_labels']
        self.labels_test_loader = loaders['test_labels']

        self.reset_losses()

        self.dynamics_criterion = nn.MSELoss(reduction='mean')
        self.labels_criterion = LabelsLoss(
            reduction='mean',
            scale=config.get('contrastive_scale', 100.0),
            loss_type=config.get('contrastive_loss_type', 'sigmoid'),
            margin=config.get('contrastive_margin', 1.0),
        )

        self.lr = config["learning_rate"]

        # Nominal points for clustering loss
        nominal_success = config.get('nominal_success', [0.9, 0.9])
        nominal_failure = config.get('nominal_failure', [-0.9, -0.9])
        self.nominal_success = torch.tensor(nominal_success, dtype=torch.float32, device=self.device)
        self.nominal_failure = torch.tensor(nominal_failure, dtype=torch.float32, device=self.device)
        self.clustering_rollout_steps = config.get('clustering_rollout_steps', 10)
        self.clustering_grid_samples = config.get('clustering_grid_samples', 256)

        self.model_dir = config["model_dir"]
        self.log_dir = config["log_dir"]

    def save_models(self):
        torch.save(self.encoder, os.path.join(self.model_dir, 'encoder.pt'))
        torch.save(self.dynamics, os.path.join(self.model_dir, 'dynamics.pt'))
        torch.save(self.decoder, os.path.join(self.model_dir, 'decoder.pt'))
    
    def save_logs(self, suffix):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        with open(os.path.join(self.log_dir, 'train_losses_' + suffix + '.pkl'), 'wb') as f:
            pickle.dump(self.train_losses, f)
        
        with open(os.path.join(self.log_dir, 'test_losses_' + suffix + '.pkl'), 'wb') as f:
            pickle.dump(self.test_losses, f)
    
    def reset_losses(self):
        self.train_losses = {'loss_ae1': [], 'loss_ae2': [], 'loss_dyn': [], 'loss_contrastive': [], 'loss_fixed_point': [], 'loss_clustering': [], 'loss_total': []}
        self.test_losses = {'loss_ae1': [], 'loss_ae2': [], 'loss_dyn': [], 'loss_contrastive': [], 'loss_fixed_point': [], 'loss_clustering': [], 'loss_total': []}
    
    def forward(self, x_t, x_tau):
        x_t = x_t.to(self.device)
        x_tau = x_tau.to(self.device)

        z_t = self.encoder(x_t)
        x_t_pred = self.decoder(z_t)

        z_tau = self.encoder(x_tau)
        x_tau_pred = self.decoder(z_tau)

        z_tau_pred = self.dynamics(z_t)
        x_tau_pred_dyn = self.decoder(z_tau_pred)

        return (x_t, x_tau, x_t_pred, z_tau, z_tau_pred, x_tau_pred_dyn)

    def dynamics_losses(self, forward_pass, weight):
        x_t, x_tau, x_t_pred, z_tau, z_tau_pred, x_tau_pred_dyn = forward_pass

        loss_ae1 = self.dynamics_criterion(x_t, x_t_pred)
        loss_ae2 = self.dynamics_criterion(x_tau, x_tau_pred_dyn)
        loss_dyn = self.dynamics_criterion(z_tau_pred, z_tau)
        loss_total = loss_ae1 * weight[0] + loss_ae2 * weight[1] + loss_dyn * weight[2]
        return loss_ae1, loss_ae2, loss_dyn, loss_total

    def labels_losses(self, encodings, pairs, weight):
        return self.labels_criterion(encodings[pairs['successes']], encodings[pairs['failures']]) * weight

    def fixed_point_loss(self, z_final, weight):
        """Loss that forces dynamics(z_final) == z_final for all trajectory endpoints."""
        z_final_pred = self.dynamics(z_final)
        return self.dynamics_criterion(z_final_pred, z_final) * weight

    def clustering_loss(self, z_final, labels, weight):
        """Combined: encoder clustering + nominal fixed-points + dynamics rollout + global regularization."""
        success_mask = (labels == 1)
        failure_mask = (labels == 0)
        loss = torch.tensor(0.0, device=self.device)

        # (A) Encoder clustering: pull z_final toward nominal points
        if success_mask.any():
            loss = loss + torch.mean((z_final[success_mask] - self.nominal_success) ** 2)
        if failure_mask.any():
            loss = loss + torch.mean((z_final[failure_mask] - self.nominal_failure) ** 2)

        # (B) Nominal fixed-point: dynamics(nominal) == nominal
        nom_s = self.nominal_success.unsqueeze(0)
        nom_f = self.nominal_failure.unsqueeze(0)
        loss = loss + self.dynamics_criterion(self.dynamics(nom_s), nom_s)
        loss = loss + self.dynamics_criterion(self.dynamics(nom_f), nom_f)

        # (C) Dynamics rollout from data endpoints: pull toward correct nominal
        z = z_final.detach().clone()
        for _ in range(self.clustering_rollout_steps):
            z = self.dynamics(z)
        target = torch.where(
            labels.unsqueeze(1) == 1,
            self.nominal_success.unsqueeze(0).expand_as(z),
            self.nominal_failure.unsqueeze(0).expand_as(z),
        )
        loss = loss + self.dynamics_criterion(z, target)

        # (D) Global dynamics regularization: sample random points across latent space,
        # roll forward, and force convergence to the nearest nominal point.
        # This prevents spurious attractors in unconstrained regions.
        low_dims = self.nominal_success.shape[0]
        z_rand = torch.rand(self.clustering_grid_samples, low_dims, device=self.device) * 1.9 - 0.95
        for _ in range(self.clustering_rollout_steps):
            z_rand = self.dynamics(z_rand)
        dist_to_success = torch.sum((z_rand - self.nominal_success) ** 2, dim=1)
        dist_to_failure = torch.sum((z_rand - self.nominal_failure) ** 2, dim=1)
        min_dist = torch.minimum(dist_to_success, dist_to_failure)
        loss = loss + min_dist.mean()

        return loss * weight

    def train(self, epochs=1000, patience=50, weight=[1,1,1,0,0,0]):
        '''
        Function that trains all the models with all the losses and weight.
        It will stop if the test loss does not improve for "patience" epochs.
        weight: [ae1, ae2, dyn, contrastive, fixed_point, clustering]
        '''
        weight_bool = [bool(i) for i in weight]
        list_parameters = (weight_bool[0] or weight_bool[1] or weight_bool[2]) * (list(self.encoder.parameters()) + list(self.decoder.parameters()))
        if weight_bool[5]:
            # Clustering loss backprops through encoder
            if not (weight_bool[0] or weight_bool[1] or weight_bool[2]):
                list_parameters += list(self.encoder.parameters())
        list_parameters += (weight_bool[1] or weight_bool[2] or weight_bool[4] or weight_bool[5]) * list(self.dynamics.parameters())
        optimizer = torch.optim.Adam(list_parameters, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.001, patience=patience)
        for epoch in tqdm(range(epochs)):
            loss_ae1_train = 0
            loss_ae2_train = 0
            loss_dyn_train = 0
            loss_contrastive_train = 0
            loss_fixed_point_train = 0
            loss_clustering_train = 0

            epoch_train_loss = 0
            epoch_test_loss  = 0


            if weight_bool[0] or weight_bool[1] or weight_bool[2] or weight_bool[5]:
                self.encoder.train()
            if weight_bool[0] or weight_bool[1] or weight_bool[2]:
                self.decoder.train()
            if weight_bool[1] or weight_bool[2] or weight_bool[4] or weight_bool[5]:
                self.dynamics.train()

            num_batches = min(len(self.dynamics_train_loader), len(self.labels_train_loader))
            for (x_t, x_tau), (pairs, x_final, labels_batch) in zip(self.dynamics_train_loader, self.labels_train_loader):
                optimizer.zero_grad()

                # Forward pass
                forward_pass = self.forward(x_t, x_tau)
                # Compute losses
                loss_ae1, loss_ae2, loss_dyn, loss_total = self.dynamics_losses(forward_pass, weight)
                loss_con = 0
                loss_fp = 0
                if weight[3] != 0 or weight[4] != 0 or weight[5] != 0:
                    x_final = x_final.to(self.device)
                    z_final = self.encoder(x_final)
                    if weight[3] != 0:
                        loss_con = self.labels_losses(z_final, pairs, weight[3])
                        loss_total += loss_con
                        loss_contrastive_train += loss_con.item()
                    if weight[4] != 0:
                        loss_fp = self.fixed_point_loss(z_final, weight[4])
                        loss_total += loss_fp
                        loss_fixed_point_train += loss_fp.item()
                    if weight[5] != 0:
                        labels_batch = labels_batch.to(self.device)
                        loss_cl = self.clustering_loss(z_final, labels_batch, weight[5])
                        loss_total += loss_cl
                        loss_clustering_train += loss_cl.item()
                # Backward pass
                loss_total.backward()
                optimizer.step()

                loss_ae1_train += loss_ae1.item()
                loss_ae2_train += loss_ae2.item()
                loss_dyn_train += loss_dyn.item()
                epoch_train_loss += loss_total.item()

            epoch_train_loss /= num_batches

            self.train_losses['loss_ae1'].append(loss_ae1_train / num_batches)
            self.train_losses['loss_ae2'].append(loss_ae2_train / num_batches)
            self.train_losses['loss_dyn'].append(loss_dyn_train / num_batches)
            self.train_losses['loss_contrastive'].append(loss_contrastive_train / num_batches)
            self.train_losses['loss_fixed_point'].append(loss_fixed_point_train / num_batches)
            self.train_losses['loss_clustering'].append(loss_clustering_train / num_batches)
            self.train_losses['loss_total'].append(epoch_train_loss)

            with torch.no_grad():
                loss_ae1_test = 0
                loss_ae2_test = 0
                loss_dyn_test = 0
                loss_contrastive_test = 0
                loss_fixed_point_test = 0
                loss_clustering_test = 0

                if weight_bool[0] or weight_bool[1] or weight_bool[2] or weight_bool[5]:
                    self.encoder.eval()
                if weight_bool[0] or weight_bool[1] or weight_bool[2]:
                    self.decoder.eval()
                if weight_bool[1] or weight_bool[2] or weight_bool[4] or weight_bool[5]:
                    self.dynamics.eval()

                num_batches = min(len(self.dynamics_test_loader), len(self.labels_test_loader))
                for (x_t, x_tau), (pairs, x_final, labels_batch) in zip(self.dynamics_test_loader, self.labels_test_loader):
                    # Forward pass
                    forward_pass = self.forward(x_t, x_tau)
                    # Compute losses
                    loss_ae1, loss_ae2, loss_dyn, loss_total = self.dynamics_losses(forward_pass, weight)

                    loss_ae1_test += loss_ae1.item()
                    loss_ae2_test += loss_ae2.item()
                    loss_dyn_test += loss_dyn.item()
                    epoch_test_loss += loss_total.item()

                    if weight[3] != 0 or weight[4] != 0 or weight[5] != 0:
                        x_final = x_final.to(self.device)
                        z_final = self.encoder(x_final)
                        if weight[3] != 0:
                            loss_con = self.labels_losses(z_final, pairs, weight[3])
                            loss_contrastive_test += loss_con.item()
                            epoch_test_loss += loss_con.item()
                        if weight[4] != 0:
                            loss_fp = self.fixed_point_loss(z_final, weight[4])
                            loss_fixed_point_test += loss_fp.item()
                            epoch_test_loss += loss_fp.item()
                        if weight[5] != 0:
                            labels_batch = labels_batch.to(self.device)
                            loss_cl = self.clustering_loss(z_final, labels_batch, weight[5])
                            loss_clustering_test += loss_cl.item()
                            epoch_test_loss += loss_cl.item()

                epoch_test_loss /= num_batches

                self.test_losses['loss_ae1'].append(loss_ae1_test / num_batches)
                self.test_losses['loss_ae2'].append(loss_ae2_test / num_batches)
                self.test_losses['loss_dyn'].append(loss_dyn_test / num_batches)
                self.test_losses['loss_contrastive'].append(loss_contrastive_test / num_batches)
                self.test_losses['loss_fixed_point'].append(loss_fixed_point_test / num_batches)
                self.test_losses['loss_clustering'].append(loss_clustering_test / num_batches)
                self.test_losses['loss_total'].append(epoch_test_loss)

            scheduler.step(epoch_test_loss)
            
            if epoch >= patience:
                if np.mean(self.test_losses['loss_total'][-patience:]) > np.mean(self.test_losses['loss_total'][-patience-1:-1]):
                    if self.verbose:
                        print("Early stopping")
                    break
            
            if self.verbose:
                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch + 1, epochs, epoch_train_loss, epoch_test_loss))