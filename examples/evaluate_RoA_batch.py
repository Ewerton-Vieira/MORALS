from MORALS.mg_utils import MorseGraphOutputProcessor
from MORALS.data_utils import TrajectoryDataset
from MORALS.dynamics_utils import DynamicsUtils

import argparse
import numpy as np
import os
import warnings
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)


def generate_key(design, num_layers, step):
    """Generate a unique key for grouping experiments."""
    return design + "_" + str(num_layers) + "_" + str(step)


def evaluate_single_experiment(output_subdir, trajectories, successful_final_conditions_true, 
                                unsuccessful_final_conditions_true):
    """
    Evaluate a single experiment's RoA.
    
    Args:
        output_subdir: Path to experiment output directory
        trajectories: TrajectoryDataset object
        successful_final_conditions_true: Array of successful final conditions
        unsuccessful_final_conditions_true: Array of unsuccessful final conditions
    
    Returns:
        Dictionary containing precision and recall, or None if evaluation failed
    """
    config_fname = os.path.join(output_subdir, "config.txt")
    
    if not os.path.exists(config_fname):
        return None
    
    with open(config_fname) as f:
        config = eval(f.read())
    
    try:
        mg_out_utils = MorseGraphOutputProcessor(config)
    except FileNotFoundError:
        return None
    except ValueError:
        print(f"ValueError from: {config_fname}")
        return None
    except IndexError:
        print(f"IndexError from: {config_fname}")
        return None
    
    if mg_out_utils.get_num_attractors() <= 1:
        return None
    
    dynamics = DynamicsUtils(config)
    
    encoded_successful_final_conditions = dynamics.encode(successful_final_conditions_true)
    encoded_unsuccessful_final_conditions = dynamics.encode(unsuccessful_final_conditions_true)
    
    # Identify success attractors
    success_attractors = set()
    for i in range(encoded_successful_final_conditions.shape[0]):
        node_id = mg_out_utils.which_morse_node(encoded_successful_final_conditions[i])
        success_attractors.add(node_id)
    
    if -1 in success_attractors:
        success_attractors.remove(-1)
    
    # Compute the basin of attraction: all nodes that flow into success attractors
    inside_nodes = set(success_attractors)
    changed = True
    while changed:
        changed = False
        for node in list(mg_out_utils.outgoing_edges.keys()):
            if node not in inside_nodes:
                outgoing = mg_out_utils.outgoing_edges[node]
                if any(target in inside_nodes for target in outgoing):
                    inside_nodes.add(node)
                    changed = True
    
    encoded_initial_states = dynamics.encode(np.vstack([elt[0] for elt in trajectories]))
    
    ground_truth = []
    predicted = []
    
    for i in range(len(trajectories)):
        ground_truth.append(trajectories.get_label(i))
        predicted_roa = mg_out_utils.which_morse_node(encoded_initial_states[i])
        
        if predicted_roa in inside_nodes:
            predicted.append(1)
        else:
            predicted.append(0)
    
    ground_truth = np.array(ground_truth)
    predicted = np.array(predicted)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predicted, average='binary', zero_division=0
    )
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'config': config
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Batch evaluate Region of Attraction (RoA) computation results across multiple experiments'
    )
    parser.add_argument(
        '--output_base',
        type=str,
        required=True,
        help='Base output directory containing experiment subdirectories (e.g., output/)'
    )
    parser.add_argument(
        '--experiment_dir',
        type=str,
        required=True,
        help='Experiment directory name within output_base (e.g., pendulum_lqr1k)'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        default="",
        help='Optional: Directory containing test trajectory data (overrides config)'
    )
    parser.add_argument(
        '--test_labels',
        type=str,
        default="",
        help='Optional: File containing test labels (overrides config)'
    )
    
    args = parser.parse_args()
    
    output_dir = os.path.join(args.output_base, args.experiment_dir)
    
    if not os.path.exists(output_dir):
        print(f"ERROR: Output directory does not exist: {output_dir}")
        exit(1)
    
    print("\n" + "=" * 80)
    print("BATCH REGION OF ATTRACTION EVALUATION")
    print("=" * 80)
    print(f"Base directory: {args.output_base}")
    print(f"Experiment directory: {args.experiment_dir}")
    print(f"Full path: {output_dir}")
    if args.test_data:
        print(f"Test data directory: {args.test_data}")
    if args.test_labels:
        print(f"Test labels file: {args.test_labels}")
    print("=" * 80 + "\n")
    
    trajectories = None
    successful_final_conditions_true = None
    unsuccessful_final_conditions_true = None
    
    precisions = defaultdict(list)
    recalls = defaultdict(list)
    f1_scores = defaultdict(list)
    
    subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    
    if len(subdirs) == 0:
        print(f"No subdirectories found in {output_dir}. Evaluating single experiment...")
        subdirs = ["."]
    
    print(f"Found {len(subdirs)} experiment(s) to evaluate\n")
    
    for subdir in tqdm(subdirs, desc="Evaluating experiments"):
        output_subdir = os.path.join(output_dir, subdir)
        
        if not os.path.isdir(output_subdir):
            continue
        
        config_fname = os.path.join(output_subdir, "config.txt")
        if not os.path.exists(config_fname):
            continue
        
        with open(config_fname) as f:
            config = eval(f.read())
        
        k = generate_key(config['experiment'], config['num_layers'], config['step'])
        
        if trajectories is None:
            print(f"\nLoading trajectory dataset (shared across all experiments)...")
            if args.test_data != "" and args.test_labels != "":
                config["data_dir"] = args.test_data
                config["labels_fname"] = args.test_labels
            
            trajectories = TrajectoryDataset(config)
            successful_final_conditions_true = trajectories.get_successful_final_conditions()
            unsuccessful_final_conditions_true = trajectories.get_unsuccessful_final_conditions()
            
            print(f"  Total trajectories: {len(trajectories)}")
            print(f"  Successful: {len(successful_final_conditions_true)}")
            print(f"  Unsuccessful: {len(unsuccessful_final_conditions_true)}\n")
        
        results = evaluate_single_experiment(
            output_subdir, 
            trajectories, 
            successful_final_conditions_true,
            unsuccessful_final_conditions_true
        )
        
        if results is not None:
            precisions[k].append(results['precision'])
            recalls[k].append(results['recall'])
            f1_scores[k].append(results['f1'])
    
    print("\n" + "=" * 80)
    print("BATCH EVALUATION RESULTS")
    print("=" * 80)
    
    if len(precisions) == 0:
        print("No valid experiments found for evaluation.")
    else:
        print(f"{'Experiment Key':<40} {'Precision':<15} {'Recall':<15} {'F1 Score':<15}")
        print("-" * 80)
        
        for k in sorted(precisions.keys()):
            mean_precision = np.mean(precisions[k])
            std_precision = np.std(precisions[k])
            mean_recall = np.mean(recalls[k])
            std_recall = np.std(recalls[k])
            mean_f1 = np.mean(f1_scores[k])
            std_f1 = np.std(f1_scores[k])
            
            print(f"{k:<40} {mean_precision:.4f}±{std_precision:.4f}  "
                  f"{mean_recall:.4f}±{std_recall:.4f}  "
                  f"{mean_f1:.4f}±{std_f1:.4f}")
        
        print("=" * 80 + "\n")
        
        results_file = os.path.join(output_dir, "precision_recall_summary.txt")
        with open(results_file, "w") as f:
            f.write("Experiment_Key,Mean_Precision,Std_Precision,Mean_Recall,Std_Recall,Mean_F1,Std_F1\n")
            for k in sorted(precisions.keys()):
                mean_precision = np.mean(precisions[k])
                std_precision = np.std(precisions[k])
                mean_recall = np.mean(recalls[k])
                std_recall = np.std(recalls[k])
                mean_f1 = np.mean(f1_scores[k])
                std_f1 = np.std(f1_scores[k])
                
                f.write(f"{k},{mean_precision},{std_precision},"
                       f"{mean_recall},{std_recall},{mean_f1},{std_f1}\n")
        
        print(f"Summary results saved to: {results_file}")

