import torch 
import os
import pickle 
import numpy as np
from torch import nn
from tqdm import tqdm
from MORALS.models import PhaseSpaceClassifier

class ClassifierTraining:
    def __init__(self,config,loaders,verbose):
        self.classifier = PhaseSpaceClassifier(config)

        self.verbose = bool(verbose)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ",self.device)

        self.classifier.to(self.device)

        self.train_loader = loaders['train_dynamics']
        self.test_loader = loaders['test_dynamics']

        self.reset_losses()

        self.lr = config["learning_rate"]

        self.model_dir = config["model_dir"]
        self.log_dir = config["log_dir"]

        # Todo: Get this from config
        self.penalty_matrix = np.array([
          [0,2,2,2,2],
          [0,1,2,2,2],
          [0,0,0,0,0],
          [2,2,2,1,0],
          [2,2,2,2,0]
        ])

        self.penalty_matrix = torch.from_numpy(self.penalty_matrix).float().to(self.device)

    def reset_losses(self):
        self.train_losses = {'loss_total': []}
        self.test_losses = {'loss_total': []}
    
    def save_model(self):
        torch.save(self.classifier, os.path.join(self.model_dir, 'classifier.pt'))
    
    def save_logs(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        with open(os.path.join(self.log_dir, 'train_losses.pkl'), 'wb') as f:
            pickle.dump(self.train_losses, f)
        
        with open(os.path.join(self.log_dir, 'test_losses.pkl'), 'wb') as f:
            pickle.dump(self.test_losses, f)
    
    def loss_function(self, forward_dict):
        probs_xt = forward_dict['probs_xt']
        probs_xnext = forward_dict['probs_xnext']

        loss = -torch.einsum('bi,ij,bj->',probs_xt,self.penalty_matrix,probs_xnext)
        loss /= probs_xt.shape[0]

        return loss
    
    def train(self,epochs=1000,patience=50):
        list_parameters = list(self.classifier.parameters())
        optimizer = torch.optim.Adam(list_parameters, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, verbose=self.verbose)

        for epoch in tqdm(range(epochs)):
            epoch_train_loss = 0

            self.classifier.train()

            for i, (x_t, x_tau, label) in enumerate(self.train_loader):
                optimizer.zero_grad()

                forward_dict = self.classifier(x_t.to(self.device), x_tau.to(self.device))
                loss = self.loss_function(forward_dict)

                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
            
            epoch_train_loss /= len(self.train_loader)
            self.train_losses['loss_total'].append(epoch_train_loss)

            epoch_test_loss = 0

            self.classifier.eval()

            with torch.no_grad():
                for i, (x_t, x_tau, label) in enumerate(self.test_loader):
                    forward_dict = self.classifier(x_t.to(self.device), x_tau.to(self.device))
                    loss = self.loss_function(forward_dict)
                    epoch_test_loss += loss.item()
                
                epoch_test_loss /= len(self.test_loader)
                self.test_losses['loss_total'].append(epoch_test_loss)

            scheduler.step(epoch_test_loss)

            if epoch >= patience:
                if np.mean(self.test_losses['loss_total'][-patience:]) > np.mean(self.test_losses['loss_total'][-patience-1:-1]):
                    break
            
            if self.verbose:
                print(f"Epoch: {epoch}, Train Loss: {epoch_train_loss}, Test Loss: {epoch_test_loss}")