from MORALS.mg_utils import MorseGraphOutputProcessor
from MORALS.data_utils import *
from MORALS.dynamics_utils import *
import argparse 
import pandas as pd

np.set_printoptions(suppress=True)

import os
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

def system_data_process_cell(fname):
    return np.loadtxt(fname, delimiter=',', skiprows=1, usecols=range(1, 201))

def merge_predition_cell(fname1, fname2, output_dir):
    # Read first column of file1, skipping the first row
    col1 = pd.read_csv(fname1, usecols=[0], skiprows=1)

    # Read first column of file2 normally
    col2 = pd.read_csv(fname2, usecols=[0])

    # Merge (stack vertically)
    merged = pd.concat([col1, col2], ignore_index=True, axis=1)
    # Save to a new CSV
    merged.to_csv(output_dir + 'predicting_RoA.csv', index=False, header=False)
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--out_dir',type=str, default="cell_inter3")
    argparser.add_argument('--data_file',type=str, default="data/cell_raw/interpolated_cells.csv")

    args = argparser.parse_args()

    trajectories = None
    successful_final_conditions_true = None

    output_dir = os.path.join("output/", args.out_dir)
    precisions = {}
    recalls = {}

    for dir in tqdm(os.listdir(output_dir)):
        if not dir.endswith('config.txt'): continue
        config_fname = os.path.join(output_dir, dir)

        with open(config_fname) as f:
            config = eval(f.read())
        
        try:
            mg_out_utils = MorseGraphOutputProcessor(config)
        except FileNotFoundError:
            continue
        except ValueError:
            print("ValueError from: ", config_fname)
            continue
        except IndexError:
            print("IndexError from: ", config_fname)
            continue

        if mg_out_utils.get_num_attractors() <= 1: continue
        
        dynamics = DynamicsUtils(config)

        data_X = []

        subsample = config['subsample']
        system = get_system(config['system'], config['high_dims'])
        control = config['control']
        print("Getting data for: ",system.name)

        

        if args.data_file == "":
            txt_files = [f for f in os.listdir(config['data_dir']) if f.endswith('.txt')]
            for f in tqdm(txt_files):
                data_temp = np.loadtxt(os.path.join(config['data_dir'], f), delimiter=',')
                data_X.append(system.transform(data_temp))
            data_X = np.vstack(data_X)
        else:
            if system.name == "basic":
                data_X = system_data_process_cell(args.data_file)
                data_X = system.transform(data_X)
            else:
                raise NotImplementedError("missing data processing for system: ", system.name)
                


        

        encoded_data_X = dynamics.encode(data_X)

        predicted = []
        for i in encoded_data_X:
            predicted.append(mg_out_utils.which_morse_node(i))
        predicted = np.array(predicted)

        np.savetxt(output_dir + "predicted_labels.txt", predicted, delimiter=',', fmt='%d')

        if system.name == "basic":
            merge_predition_cell(args.data_file, output_dir + "predicted_labels.txt", output_dir)
