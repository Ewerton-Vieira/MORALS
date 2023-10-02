import numpy as np 
import torch
import torch.nn as nn

from src.systems.utils import get_system
from collections import defaultdict
from src.grid import Grid

import os

class MorseGraphOutputProcessor:
    def __init__(self, config):
        mg_roa_fname = os.path.join(config['output_dir'], 'MG_RoA_.csv')
        mg_att_fname = os.path.join(config['output_dir'], 'MG_attractors.txt')
        mg_fname = os.path.join(config['output_dir'], 'MG')

        self.dims = config['low_dims']

        # Check if the file exists
        if not os.path.exists(mg_roa_fname):
            raise FileNotFoundError("Morse Graph RoA file does not exist at: " + config['output_dir'])
        with open(mg_roa_fname, 'r') as f:
            lines = f.readlines()
            # Find indices where the first character is an alphabet
            self.indices = []
            for i, line in enumerate(lines):
                if line[0].isalpha():
                    self.indices.append(i)
            self.box_size = np.array(lines[self.indices[0]+1].split(',')).astype(np.float32)
            self.morse_nodes_data = np.vstack([np.array(line.split(',')).astype(np.float32) for line in lines[self.indices[1]+1:self.indices[2]]])
            if len(self.indices) >= 2:
                self.attractor_nodes_data = np.vstack([np.array(line.split(',')).astype(np.float32) for line in lines[self.indices[2]+1:]])
            else:
                line = lines[self.indices[2]+1]
                self.attractor_nodes_data = np.array(line.split(',').astype(np.float32))

        self.morse_nodes = np.unique(self.morse_nodes_data[:, 1])

        self.corner_points = {}
        for i in range(self.morse_nodes_data.shape[0]):
            self.corner_points[int(self.morse_nodes_data[i, 0])] = int(self.morse_nodes_data[i, 1])
        for i in range(self.attractor_nodes_data.shape[0]):
            self.corner_points[int(self.attractor_nodes_data[i, 0])] = int(self.attractor_nodes_data[i, 1])

        if not os.path.exists(mg_att_fname):
            raise FileNotFoundError("Morse Graph attractors file does not exist at: " + config['output_dir'])
        
        self.found_attractors = -1
        with open(mg_att_fname, 'r') as f:
            line = f.readline()
            # Obtain the last number after a comma
            self.found_attractors = int(line.split(",")[-1])
            # Find the numbers enclosed in square brackets
            self.attractor_nodes = np.array([int(x) for x in line.split("[")[1].split("]")[0].split(",")])

        if not os.path.exists(mg_fname):
            raise FileNotFoundError("Morse Graph file does not exist at: " + config['output_dir'])
        
        self.incoming_edges = defaultdict(list)
        self.outgoing_edges = defaultdict(list)
        with open(mg_fname, 'r') as f:
            # Check for lines of the form a -> b;
            for line in f.readlines():
                if line.find("->") != -1:
                    a = int(line.split("->")[0].strip())
                    b = int(line.split("->")[1].split(";")[0].strip())
                    self.outgoing_edges[a].append(b)
                    self.incoming_edges[b].append(a)

        lower_bounds = [-1.]*self.dims
        upper_bounds = [1.]*self.dims
        latent_space_area = np.prod(np.array(upper_bounds) - np.array(lower_bounds))
        box_area = np.prod(self.box_size)
        subdivisions = np.log2(latent_space_area/box_area)
        self.grid = Grid(lower_bounds, upper_bounds, int(subdivisions))

    def get_num_attractors(self):
        return self.found_attractors
    
    def get_corner_points_of_attractor(self, id):
        # Get the attractor nodes
        attractor_nodes = self.attractor_nodes_data[self.attractor_nodes_data[:, 1] == id]
        return attractor_nodes[:, 2:]        

    def get_corner_points_of_morse_node(self, id):
        morse_node_nodes = self.morse_nodes_data[self.morse_nodes_data[:, 1] == id]
        return morse_node_nodes[:, 2:]
    
    def which_morse_node(self, point):
        assert point.shape[0] == self.dims
        found = self.corner_points[self.grid.point2indexCMGDB(point)]
        return found