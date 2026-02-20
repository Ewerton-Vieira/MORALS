from MORALS.systems.utils import get_system

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from MORALS.systems.utils import get_system
import torch
import os
import json


def load_achieved_bounds(dataset_description_fname):
    """Load achieved_bounds from dataset_description.json"""
    with open(dataset_description_fname, 'r') as f:
        desc = json.load(f)
    return desc.get('achieved_bounds', {}), desc.get('data_format', {}).get('trajectory_files', {}).get('state_order',
           desc.get('state_space', {}).get('state_order', []))

class DynamicsDataset(Dataset):
    def __init__(self, config, file_list=None):
        Xt = []
        Xnext = []

        step = config['step']
        subsample = config['subsample']
        system = get_system(config['system'], config['high_dims'])
        print("Getting data for: ",system.name)

        # If file_list is provided, use it; otherwise, load all txt files from data_dir
        if file_list is not None:
            txt_files = file_list
            print(f"Loading {len(txt_files)} specified trajectory files")
        else:
            txt_files = [f for f in os.listdir(config['data_dir']) if f.endswith('.txt')]
            print(f"Loading all {len(txt_files)} trajectory files from directory")

        # for f in tqdm(os.listdir(config['data_dir'])):
        for f in tqdm(txt_files):
            data = np.loadtxt(os.path.join(config['data_dir'], f), delimiter=',')
            indices = np.arange(data.shape[0])
            subsampled_indices = indices % subsample == 0
            subsampled_data_untransformed = data[subsampled_indices]
            subsampled_data = system.transform(subsampled_data_untransformed)
            Xt.append(subsampled_data[:-step])
            Xnext.append(subsampled_data[step:])

            if config.get('use_synthetic_attractors', False):
                # Synthetic attractor pairs: final state acts as fixed point
                final_state = subsampled_data[-1:]
                tail_states = subsampled_data[-(step):]
                final_repeated = np.repeat(final_state, len(tail_states), axis=0)
                Xt.append(tail_states)
                Xnext.append(final_repeated)
            # for i in range(subsampled_data.shape[0] - step):
            #     Xt.append(system.transform(subsampled_data[i]))
            #     Xnext.append(system.transform(subsampled_data[i + step]))
        self.Xt = np.vstack(Xt)
        self.Xnext = np.vstack(Xnext)
        assert len(self.Xt) == len(self.Xnext), "Xt and Xnext must have the same length"

        # Normalize the data - only normalize dimensions in system.normalize_indices
        # Get normalize_indices from system (defaults to all dimensions for backwards compatibility)
        normalize_indices = getattr(system, 'normalize_indices', list(range(self.Xt.shape[1])))

        # Determine normalization method: 'data' (old) or 'dataset_description' (new)
        # Default to 'data' for backwards compatibility
        normalization_method = config.get('normalization_method', 'data')

        if normalization_method == 'dataset_description' and 'dataset_description_fname' in config:
            # Load achieved_bounds from dataset_description.json
            achieved_bounds, state_order = load_achieved_bounds(config['dataset_description_fname'])
            print(f"Using achieved_bounds from {config['dataset_description_fname']}")
            print(f"State order: {state_order}")
            print(f"Normalizing indices: {normalize_indices}")

            # Initialize X_min and X_max for transformed space
            transformed_dim = self.Xt.shape[1]
            self.X_min = np.zeros(transformed_dim)
            self.X_max = np.ones(transformed_dim)

            # Build mapping from raw state index to achieved_bounds
            raw_bounds_min = []
            raw_bounds_max = []
            for state_name in state_order:
                if state_name in achieved_bounds:
                    raw_bounds_min.append(achieved_bounds[state_name]['min'])
                    raw_bounds_max.append(achieved_bounds[state_name]['max'])
                else:
                    # Fallback for unmapped states
                    raw_bounds_min.append(-1.0)
                    raw_bounds_max.append(1.0)

            # Get bounds for normalized dimensions based on system type
            # This maps transformed indices to their normalization bounds
            self._set_normalization_bounds(system, normalize_indices, raw_bounds_min, raw_bounds_max, state_order)

        elif config['use_limits']:
            raise NotImplementedError
        else:
            # Legacy: Get bounds from the max of both Xt and Xnext
            self.X_min = np.min(np.concatenate((self.Xt, self.Xnext), axis=0), axis=0)
            self.X_max = np.max(np.concatenate((self.Xt, self.Xnext), axis=0), axis=0)

        for i in range(self.X_min.shape[0]):
            if np.abs(self.X_min[i] - self.X_max[i]) < 1e-6:
                print("Warning: X_min and X_max are the same for dimension ", i)
                self.X_min[i] -= 1
                self.X_max[i] += 1

        # Apply normalization only to specified indices
        for i in normalize_indices:
            self.Xt[:, i] = (self.Xt[:, i] - self.X_min[i]) / (self.X_max[i] - self.X_min[i])
            self.Xnext[:, i] = (self.Xnext[:, i] - self.X_min[i]) / (self.X_max[i] - self.X_min[i])

        # If model_dir does nto exist, create it
        if not os.path.exists(config['model_dir']):
            os.makedirs(config['model_dir'])

        # Write the normalization parameters to a file
        np.savetxt(os.path.join(config['model_dir'], 'X_min.txt'), self.X_min, delimiter=',')
        np.savetxt(os.path.join(config['model_dir'], 'X_max.txt'), self.X_max, delimiter=',')

        # Convert to torch tensors
        self.Xt = torch.from_numpy(self.Xt).float()
        self.Xnext = torch.from_numpy(self.Xnext).float()

    def _set_normalization_bounds(self, system, normalize_indices, raw_bounds_min, raw_bounds_max, state_order):
        """Set X_min and X_max for normalized dimensions based on system type and achieved_bounds."""
        system_name = system.name

        if system_name == "pendulum":
            # Raw: [theta, theta_dot] -> Transformed: [x, y, xdot, ydot]
            # Normalize indices: [2, 3] (xdot, ydot)
            # xdot, ydot depend on theta_dot: max = l * |theta_dot_max|
            l = getattr(system, 'l', 0.5)
            theta_dot_idx = state_order.index('theta_dot') if 'theta_dot' in state_order else 1
            theta_dot_max = max(abs(raw_bounds_min[theta_dot_idx]), abs(raw_bounds_max[theta_dot_idx]))
            for i in normalize_indices:
                self.X_min[i] = -l * theta_dot_max
                self.X_max[i] = l * theta_dot_max

        elif system_name == "cartpole_pybullet":
            # Raw: [x, theta, x_dot, theta_dot] -> Transformed: [x, cos, sin, x_dot, theta_dot]
            # Normalize indices: [0, 3, 4]
            idx_map = {0: 'x', 3: 'x_dot', 4: 'theta_dot'}
            for trans_idx, raw_name in idx_map.items():
                if trans_idx in normalize_indices and raw_name in state_order:
                    raw_idx = state_order.index(raw_name)
                    self.X_min[trans_idx] = raw_bounds_min[raw_idx]
                    self.X_max[trans_idx] = raw_bounds_max[raw_idx]

        elif system_name == "quadrotor2D":
            # Raw: [x, z, theta, x_dot, z_dot, theta_dot] -> Transformed: [x, z, sin, cos, x_dot, z_dot, theta_dot]
            # Normalize indices: [0, 1, 4, 5, 6]
            idx_map = {0: 'x', 1: 'z', 4: 'x_dot', 5: 'z_dot', 6: 'theta_dot'}
            for trans_idx, raw_name in idx_map.items():
                if trans_idx in normalize_indices and raw_name in state_order:
                    raw_idx = state_order.index(raw_name)
                    self.X_min[trans_idx] = raw_bounds_min[raw_idx]
                    self.X_max[trans_idx] = raw_bounds_max[raw_idx]

        elif system_name == "quadrotor3D":
            # Raw = Transformed: [x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r]
            # Normalize indices: [0, 1, 2, 7, 8, 9, 10, 11, 12]
            idx_map = {0: 'x', 1: 'y', 2: 'z', 7: 'x_dot', 8: 'y_dot', 9: 'z_dot', 10: 'p', 11: 'q', 12: 'r'}
            for trans_idx, raw_name in idx_map.items():
                if trans_idx in normalize_indices and raw_name in state_order:
                    raw_idx = state_order.index(raw_name)
                    self.X_min[trans_idx] = raw_bounds_min[raw_idx]
                    self.X_max[trans_idx] = raw_bounds_max[raw_idx]

        else:
            # Fallback: use all dimensions from achieved_bounds (legacy behavior)
            print(f"Warning: No specific normalization mapping for system {system_name}, using raw bounds")
            for i in normalize_indices:
                if i < len(raw_bounds_min):
                    self.X_min[i] = raw_bounds_min[i]
                    self.X_max[i] = raw_bounds_max[i]

    def __len__(self):
        return len(self.Xt)

    def __getitem__(self, idx):
        return self.Xt[idx], self.Xnext[idx]

class LabelsDataset(Dataset):
    def __init__(self, config, file_list=None, labels_list=None):
        # If file_list and labels_list are provided directly, use them
        if file_list is not None and labels_list is not None:
            print(f"Loading {len(file_list)} specified files with provided labels")
            files_to_load = file_list
            labels_dict = {f: label for f, label in zip(file_list, labels_list)}
        else:
            # Legacy behavior: load labels from labels_fname and scan directory
            labels = np.loadtxt(config['labels_fname'], delimiter=',', dtype=str)
            labels_dict = {}
            for i in range(len(labels)):
                labels_dict[labels[i,0]] = int(labels[i,1])
            files_to_load = os.listdir(config['data_dir'])
            print(f"Loading all files from directory with labels from {config['labels_fname']}")

        self.final_points = []
        self.labels = []

        for f in tqdm(files_to_load):
            data = np.loadtxt(os.path.join(config['data_dir'], f), delimiter=',')
            self.final_points.append(data[-1])
            try:
                self.labels.append(labels_dict[f])
            except KeyError:
                print("No label found for ", f)
                self.labels.append(0)

        self.final_points = np.array(self.final_points)
        system = get_system(config['system'], config['high_dims'])
        self.final_points = system.transform(self.final_points)
        X_max = np.loadtxt(os.path.join(config['model_dir'], 'X_max.txt'), delimiter=',')
        X_min = np.loadtxt(os.path.join(config['model_dir'], 'X_min.txt'), delimiter=',')

        # Apply normalization only to specified indices (consistent with DynamicsDataset)
        normalize_indices = getattr(system, 'normalize_indices', list(range(self.final_points.shape[1])))
        for i in normalize_indices:
            self.final_points[:, i] = (self.final_points[:, i] - X_min[i]) / (X_max[i] - X_min[i])
        self.final_points = torch.from_numpy(self.final_points).float()

        self.labels = np.array(self.labels)
        self.labels = torch.from_numpy(self.labels).long()

        self.generate_opposite_pairs()

    def __len__(self):
        return len(self.opposite_pairs)

    def __getitem__(self, idx):
        return self.opposite_pairs[idx]

    def generate_opposite_pairs(self):
        """
        Generates the opposite pairs where each pair has the form (success, failure)
        """

        # Gathering the indices of the success and failure labels
        success_indices = torch.where(self.labels == 1)[0]
        failure_indices = torch.where(self.labels == 0)[0]

        # Generating cartesian product of success and failure indices
        self.opposite_pairs = torch.stack(torch.meshgrid(success_indices, failure_indices)).T.reshape(-1,2)

        print('Number of pairs: ', len(self.opposite_pairs))


    def collate_fn(self, batch):
        """
        Processes the batch of pairs to create a batch of points present in the pairs and a new batch of pairs is created with the updated indices of the points
        :param batch: batch of pairs of the form [success, failure]
        :return: updated batch of pairs and the batch of points
        """
        pairs = np.array([pair.to('cpu').numpy() for pair in batch])

        # Getting the points present in the batch of pairs
        unique_indices = np.unique(pairs)
        x_batch = self.final_points[unique_indices]
        batch_labels = self.labels[unique_indices]
        orig_indices_vs_new_indices = {orig_idx: new_idx for new_idx, orig_idx in enumerate(unique_indices)}

        # Updating the indices of the triplets to match the indices of the points in the batch
        updated_success_indices = []
        updated_failure_indices = []

        for success_idx, failure_idx in pairs:
            updated_success_indices.append(orig_indices_vs_new_indices[success_idx])
            updated_failure_indices.append(orig_indices_vs_new_indices[failure_idx])

        return {"successes": updated_success_indices, "failures": updated_failure_indices}, x_batch, batch_labels

class TrajectoryDataset:
    # Useful for plotting
    def __init__(self, config, file_list=None, labels_list=None):
        self.trajs = []
        subsample = config['subsample']

        system = get_system(config['system'], config['high_dims'])
        print("Getting data for: ",system.name)

        self.labels_dict = {}
        self.labels = []

        labels_fname = config.get('labels_fname', None)

        # Determine which files to load and their labels
        if file_list is not None and labels_list is not None:
            # Use provided file list and labels
            print(f"Loading {len(file_list)} specified trajectory files with provided labels")
            files_to_load = file_list
            self.labels_dict = {f: label for f, label in zip(file_list, labels_list)}
        elif labels_fname is not None:
            # Check if we're using the new roa_labels.txt format (state1, state2, ..., label)
            # or the old format (filename, label)
            try:
                labels_data = np.loadtxt(labels_fname, delimiter=',')
                if labels_data.ndim == 2 and labels_data.shape[1] > 2:
                    # New format: <state1> <state2> ... <label>
                    # Need shuffled_indices to map rows to filenames
                    if 'shuffled_indices_fname' in config:
                        print(f"Loading labels from new format: {labels_fname}")
                        with open(config['shuffled_indices_fname'], 'r') as f:
                            shuffled_files = [line.strip() for line in f.readlines()]
                        labels = labels_data[:, -1].astype(int)
                        self.labels_dict = {f: label for f, label in zip(shuffled_files, labels)}
                        files_to_load = shuffled_files
                        print(f"Loaded {len(files_to_load)} files from shuffled indices")
                    else:
                        raise ValueError("New roa_labels.txt format requires 'shuffled_indices_fname' in config")
                else:
                    # Old format: filename, label
                    print(f"Loading labels from old format: {labels_fname}")
                    labels = np.loadtxt(labels_fname, delimiter=',', dtype=str)
                    for i in range(len(labels)):
                        self.labels_dict[labels[i,0]] = int(labels[i,1])
                    files_to_load = os.listdir(config['data_dir'])
            except Exception as e:
                print(f"Warning: Could not parse labels file: {e}")
                files_to_load = os.listdir(config['data_dir'])
        else:
            # No labels provided
            print("No labels provided, loading all files from directory")
            files_to_load = os.listdir(config['data_dir'])

        for f in tqdm(files_to_load):
            file_path = os.path.join(config['data_dir'], f)
            if not os.path.isfile(file_path):
                continue
            raw_data = np.loadtxt(file_path, delimiter=',')
            if len(raw_data) == 0: continue
            indices = np.arange(raw_data.shape[0])
            subsampled_indices = indices % subsample == 0
            subsampled_data = raw_data[subsampled_indices]
            # Transform each state in the trajectory
            self.trajs.append(system.transform(subsampled_data))
            # data = []
            # for i in range(subsampled_data.shape[0]):
            #     data.append(system.transform(subsampled_data[i]))
            # data = np.array(data)
            # self.trajs.append(data)
            if labels_fname is not None:
                try:
                    self.labels.append(self.labels_dict[f])
                except KeyError:
                    print("No label found for ", f)
                    self.labels.append(-1)

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        return self.trajs[idx]

    def get_label(self,index):
        return self.labels[index]

    def get_successful_initial_conditions(self):
        assert len(self.trajs) == len(self.labels)
        initial_points = []
        for i in range(len(self.trajs)):
            if self.labels[i] == 1:
                initial_points.append(self.trajs[i][0])
        return np.array(initial_points)

    def get_unsuccessful_initial_conditions(self):
        assert len(self.trajs) == len(self.labels)
        initial_points = []
        for i in range(len(self.trajs)):
            if self.labels[i] == 0:
                initial_points.append(self.trajs[i][0])
        return np.array(initial_points)

    def get_successful_final_conditions(self):
        assert len(self.trajs) == len(self.labels)
        final_points = []
        for i in range(len(self.trajs)):
            if self.labels[i] == 1:
                final_points.append(self.trajs[i][-1])
        return np.array(final_points)

    def get_unsuccessful_final_conditions(self):
        assert len(self.trajs) == len(self.labels)
        final_points = []
        for i in range(len(self.trajs)):
            if self.labels[i] == 0:
                final_points.append(self.trajs[i][-1])
        return np.array(final_points)
