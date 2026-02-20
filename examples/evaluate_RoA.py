from MORALS.mg_utils import MorseGraphOutputProcessor
from MORALS.dynamics_utils import DynamicsUtils
from MORALS.systems.utils import get_system

import argparse
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

# Hardcoded test set paths for each system
TEST_SET_PATHS = {
    'pendulum': '/common/users/shared/pracsys/genMoPlan/data_trajectories/pendulum_lqr_50k/test_set.txt',
    'cartpole_pybullet': '/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole_pybullet/test_set.txt',
    'quadrotor2D_rl': '/common/users/shared/pracsys/genMoPlan/data_trajectories/quadrotor2D_rl/test_set.txt',
    'quadrotor3D_lqr': '/common/users/shared/pracsys/genMoPlan/data_trajectories/quadrotor3D_lqr/test_set.txt',
}


class TestSetDataset:
    """Load test_set.txt directly without needing trajectory files.

    Format: init_state_1, ..., init_state_n, final_state_1, ..., final_state_n, label
    """
    def __init__(self, test_set_fname, config):
        print(f"Loading test set from: {test_set_fname}")
        data = np.loadtxt(test_set_fname, delimiter=',')

        # Determine raw state dimension from config
        system = get_system(config['system'], config['high_dims'])
        raw_dim = system.get_true_bounds().shape[0]

        # Parse columns: first raw_dim are initial state, next raw_dim are final state, last is label
        self.initial_states_raw = data[:, :raw_dim]
        self.final_states_raw = data[:, raw_dim:2*raw_dim]
        self.labels = data[:, -1].astype(int)

        # Transform states
        self.initial_states = system.transform(self.initial_states_raw)
        self.final_states = system.transform(self.final_states_raw)

        print(f"Loaded {len(self.labels)} test samples")
        print(f"  Raw state dim: {raw_dim}, Transformed dim: {self.initial_states.shape[1]}")
        print(f"  Successful: {np.sum(self.labels == 1)}, Unsuccessful: {np.sum(self.labels == 0)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.initial_states[idx], self.final_states[idx]

    def get_label(self, idx):
        return self.labels[idx]

    def get_successful_final_conditions(self):
        return self.final_states[self.labels == 1]

    def get_unsuccessful_final_conditions(self):
        return self.final_states[self.labels == 0]


def evaluate_roa(output_dir):
    """
    Evaluate RoA computation results using test trajectories.

    Args:
        output_dir: Directory containing MG and RoA output files

    Returns:
        Dictionary containing precision, recall, and F1 score
    """

    config_fname = os.path.join(output_dir, "config.txt")
    
    if not os.path.exists(config_fname):
        raise FileNotFoundError(f"Config file not found at: {config_fname}")
    
    print(f"\nEvaluating RoA for: {output_dir}")
    print("=" * 80)
    
    with open(config_fname) as f:
        config = eval(f.read())
    
    print(f"Loading Morse Graph outputs...")
    try:
        mg_out_utils = MorseGraphOutputProcessor(config)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return None
    except ValueError as e:
        print(f"ValueError: {e}")
        return None
    except IndexError as e:
        print(f"IndexError: {e}")
        return None
    
    num_attractors = mg_out_utils.get_num_attractors()
    print(f"Number of attractors found: {num_attractors}")
    
    if num_attractors <= 1:
        print("WARNING: Only 1 or fewer attractors found. Skipping evaluation.")
        return None
    
    print(f"Loading dynamics model...")
    dynamics = DynamicsUtils(config)
    
    # Load test data from hardcoded test_set.txt path based on system
    system = config['system']
    if system not in TEST_SET_PATHS:
        raise ValueError(f"Unknown system '{system}'. Supported systems: {list(TEST_SET_PATHS.keys())}")

    test_set_fname = TEST_SET_PATHS[system]
    print(f"Loading test data from test_set.txt: {test_set_fname}")
    trajectories = TestSetDataset(test_set_fname, config)
    print(f"Loaded {len(trajectories)} test samples")
    
    print(f"Getting successful and unsuccessful final conditions...")
    successful_final_conditions_true = trajectories.get_successful_final_conditions()
    unsuccessful_final_conditions_true = trajectories.get_unsuccessful_final_conditions()
    
    print(f"  Successful trajectories: {len(successful_final_conditions_true)}")
    print(f"  Unsuccessful trajectories: {len(unsuccessful_final_conditions_true)}")
    
    print(f"Encoding final conditions to latent space...")
    encoded_successful_final_conditions = dynamics.encode(successful_final_conditions_true)
    encoded_unsuccessful_final_conditions = dynamics.encode(unsuccessful_final_conditions_true)
    
    print(f"\nLatent space statistics:")
    print(f"  Morse graph bounds: lower={mg_out_utils.grid.lower_bounds}, upper={mg_out_utils.grid.upper_bounds}")
    print(f"  Successful final conditions - min: {np.min(encoded_successful_final_conditions, axis=0)}")
    print(f"  Successful final conditions - max: {np.max(encoded_successful_final_conditions, axis=0)}")
    
    print(f"\nIdentifying success attractors (where successful trajectories end)...")
    success_attractors = set()
    outside_count = 0
    for i in range(encoded_successful_final_conditions.shape[0]):
        node_id = mg_out_utils.which_morse_node(encoded_successful_final_conditions[i])
        success_attractors.add(node_id)
        if node_id is None:
            outside_count += 1
    
    if None in success_attractors:
        success_attractors.remove(None)
    
    print(f"  Success attractor(s): {success_attractors}")
    print(f"  WARNING: {outside_count}/{len(encoded_successful_final_conditions)} successful final conditions fall outside Morse graph coverage")
    
    print(f"\nComputing Region of Attraction (RoA) for success attractors...")
    # Compute the basin of attraction: all nodes that flow into success attractors
    inside_nodes = set(success_attractors)  # Start with the attractors themselves
    
    # Iteratively add nodes that flow into nodes already in the RoA
    changed = True
    while changed:
        changed = False
        for node in list(mg_out_utils.outgoing_edges.keys()):
            if node not in inside_nodes:
                # Check if this node flows to any node in the RoA
                outgoing = mg_out_utils.outgoing_edges[node]
                if any(target in inside_nodes for target in outgoing):
                    inside_nodes.add(node)
                    changed = True
    
    print(f"  RoA contains {len(inside_nodes)} Morse node(s): {inside_nodes}")
    
    # Compute separatrix nodes (boundary between RoA and non-RoA)
    print(f"\nComputing separatrix nodes...")
    if hasattr(mg_out_utils, "morse_nodes"):
        all_nodes = set(int(n) for n in mg_out_utils.morse_nodes.tolist())
    else:
        all_nodes = set(list(mg_out_utils.outgoing_edges.keys()) + list(mg_out_utils.incoming_edges.keys()))
    separatrix_nodes = set()
    for node in all_nodes:
        neighbors = set(mg_out_utils.outgoing_edges.get(node, [])) | set(mg_out_utils.incoming_edges.get(node, []))
        if not neighbors:
            continue
        touches_inside = any((nbr in inside_nodes) for nbr in neighbors)
        touches_outside = any((nbr not in inside_nodes) for nbr in neighbors)
        if touches_inside and touches_outside:
            separatrix_nodes.add(node)
    print(f"  Identified {len(separatrix_nodes)} separatrix node(s)")

    print(f"\nEncoding initial conditions and predicting outcomes...")
    encoded_initial_states = dynamics.encode(np.vstack([elt[0] for elt in trajectories]))
    
    # Ground-truth labels and predicted morse nodes
    ground_truth = np.array([trajectories.get_label(i) for i in range(len(trajectories))])
    predicted_nodes = [mg_out_utils.which_morse_node(encoded_initial_states[i]) for i in range(len(trajectories))]

    # Identify separatrix and out-of-coverage points; exclude both from evaluation
    separatrix_mask = np.array([pn in separatrix_nodes for pn in predicted_nodes], dtype=bool)
    out_of_coverage_mask = np.array([pn is None for pn in predicted_nodes], dtype=bool)
    num_total = len(trajectories)
    num_separatrix = int(np.sum(separatrix_mask))
    separatrix_percent = 100.0 * num_separatrix / num_total if num_total > 0 else 0.0
    num_out_of_coverage = int(np.sum(out_of_coverage_mask))
    out_of_coverage_percent = 100.0 * num_out_of_coverage / num_total if num_total > 0 else 0.0
    print(f"  Separatrix points: {num_separatrix}/{num_total} ({separatrix_percent:.2f}%)")
    print(f"  Out-of-coverage points: {num_out_of_coverage}/{num_total} ({out_of_coverage_percent:.2f}%)")

    # Predicted success/failure for remaining points
    predicted_labels = np.array([1 if (pn in inside_nodes) else 0 for pn in predicted_nodes], dtype=int)
    eval_mask = ~(separatrix_mask | out_of_coverage_mask)
    gt_eval = ground_truth[eval_mask]
    pred_eval = predicted_labels[eval_mask]

    # Compute metrics on non-separatrix points
    print(f"\nComputing metrics (excluding separatrix and out-of-coverage points)...")
    tp = tn = fp = fn = 0
    if gt_eval.size == 0:
        precision = recall = specificity = f1 = 0.0
    else:
        tp = int(np.sum((gt_eval == 1) & (pred_eval == 1)))
        tn = int(np.sum((gt_eval == 0) & (pred_eval == 0)))
        fp = int(np.sum((gt_eval == 0) & (pred_eval == 1)))
        fn = int(np.sum((gt_eval == 1) & (pred_eval == 0)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Separatrix %: {separatrix_percent:.2f}%  ({num_separatrix} of {num_total})")
    print(f"Out-of-coverage %: {out_of_coverage_percent:.2f}%  ({num_out_of_coverage} of {num_total})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("=" * 80 + "\n")
    
    return {
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'separatrix_percent': separatrix_percent,
        'num_separatrix': num_separatrix,
        'out_of_coverage_percent': out_of_coverage_percent,
        'num_out_of_coverage': num_out_of_coverage,
        'num_evaluated': int(gt_eval.size),
        'raw_count': num_total,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'num_attractors': num_attractors,
        'num_trajectories': len(trajectories),
        'num_successful': len(successful_final_conditions_true),
        'num_unsuccessful': len(unsuccessful_final_conditions_true)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate Region of Attraction (RoA) computation results'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory containing Morse graph and RoA output files (e.g., output/pendulum_lqr1k)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print(f"ERROR: Output directory does not exist: {args.output_dir}")
        exit(1)

    print("\n" + "=" * 80)
    print("REGION OF ATTRACTION EVALUATION")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    results = evaluate_roa(args.output_dir)
    
    if results is not None:
        results_file = os.path.join(args.output_dir, "evaluation_results.txt")
        with open(results_file, "w") as f:
            f.write(f"Raw: {results['raw_count']}\n")
            f.write(f"SeparatrixPercent: {results['separatrix_percent']:.6f}\n")
            f.write(f"SeparatrixCount: {results['num_separatrix']}\n")
            f.write(f"OutOfCoveragePercent: {results['out_of_coverage_percent']:.6f}\n")
            f.write(f"OutOfCoverageCount: {results['num_out_of_coverage']}\n")
            f.write(f"EvaluatedCount: {results['num_evaluated']}\n")
            f.write(f"TP: {results['tp']}\n")
            f.write(f"TN: {results['tn']}\n")
            f.write(f"FP: {results['fp']}\n")
            f.write(f"FN: {results['fn']}\n")
            f.write(f"Precision: {results['precision']:.6f}\n")
            f.write(f"Recall: {results['recall']:.6f}\n")
            f.write(f"Specificity: {results['specificity']:.6f}\n")
            f.write(f"F1 Score: {results['f1']:.6f}\n")
            f.write(f"Number of attractors: {results['num_attractors']}\n")
            f.write(f"Number of trajectories: {results['num_trajectories']}\n")
            f.write(f"Successful trajectories: {results['num_successful']}\n")
            f.write(f"Unsuccessful trajectories: {results['num_unsuccessful']}\n")
        
        print(f"Results saved to: {results_file}")
    else:
        print("Evaluation failed or skipped.")

