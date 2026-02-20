#!/usr/bin/env python3
"""
Aggregate Grid Search Results

This script aggregates the evaluation results from a grid search run
and creates a summary CSV file with all metrics.
"""

import os
import sys
import csv
import argparse
from pathlib import Path
import re


def parse_evaluation_results(results_file):
    """
    Parse an evaluation results file and extract metrics.

    Args:
        results_file: Path to evaluation_results.txt

    Returns:
        Dictionary with metrics, or None if file doesn't exist or parsing fails
    """
    if not os.path.exists(results_file):
        return None

    try:
        results = {}
        with open(results_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('Precision:'):
                    results['precision'] = float(line.split(':')[1].strip())
                elif line.startswith('Recall:'):
                    results['recall'] = float(line.split(':')[1].strip())
                elif line.startswith('F1 Score:'):
                    results['f1'] = float(line.split(':')[1].strip())
                elif line.startswith('Number of attractors:'):
                    results['num_attractors'] = int(line.split(':')[1].strip())
                elif line.startswith('Number of trajectories:'):
                    results['num_trajectories'] = int(line.split(':')[1].strip())
                elif line.startswith('Successful trajectories:'):
                    results['num_successful'] = int(line.split(':')[1].strip())
                elif line.startswith('Unsuccessful trajectories:'):
                    results['num_unsuccessful'] = int(line.split(':')[1].strip())

        return results
    except Exception as e:
        print(f"Error parsing {results_file}: {e}")
        return None


def get_system_from_config(config_file):
    """
    Extract system name from a config file.

    Args:
        config_file: Path to config.txt

    Returns:
        System name or None if not found
    """
    if not os.path.exists(config_file):
        return None

    try:
        with open(config_file, 'r') as f:
            config = eval(f.read())

        system = config.get('system', 'unknown')
        control = config.get('control', '')

        if control:
            return f"{system}_{control}"
        return system
    except Exception as e:
        print(f"Error reading system from {config_file}: {e}")
        return None


def aggregate_results(grid_search_dir, output_csv=None, include_system=True):
    """
    Aggregate results from all runs in a grid search directory.

    Args:
        grid_search_dir: Base directory containing grid search results
        output_csv: Path to output CSV file (default: grid_search_results.csv in grid_search_dir)
        include_system: Whether to extract and include system name from configs
    """
    if not os.path.exists(grid_search_dir):
        print(f"ERROR: Grid search directory not found: {grid_search_dir}")
        return

    if output_csv is None:
        output_csv = os.path.join(grid_search_dir, 'grid_search_results.csv')

    # Collect all results
    all_results = []

    # Check if this is a system-specific directory (grid_search_{system})
    # or contains multiple system directories
    system_dirs = []
    dir_name = os.path.basename(grid_search_dir)

    if dir_name.startswith('grid_search_'):
        # This is a single system directory
        system_dirs.append((grid_search_dir, dir_name.replace('grid_search_', '')))
    else:
        # Look for grid_search_{system} subdirectories
        for item in os.listdir(grid_search_dir):
            item_path = os.path.join(grid_search_dir, item)
            if os.path.isdir(item_path) and item.startswith('grid_search_'):
                system_name = item.replace('grid_search_', '')
                system_dirs.append((item_path, system_name))

    if not system_dirs:
        # No system directories found, treat as flat structure
        system_dirs.append((grid_search_dir, None))

    # Process each system directory
    for system_dir, system_name_from_dir in system_dirs:
        print(f"\nProcessing directory: {system_dir}")
        if system_name_from_dir:
            print(f"  System (from directory): {system_name_from_dir}")

        # Iterate through experiment directories
        for experiment in os.listdir(system_dir):
            experiment_path = os.path.join(system_dir, experiment)

            # Skip files and special directories
            if not os.path.isdir(experiment_path):
                continue
            if experiment.startswith('grid_search'):
                continue

            # Check if this is an experiment directory (should contain subN directories)
            sub_dirs = [d for d in os.listdir(experiment_path) if d.startswith('sub') and os.path.isdir(os.path.join(experiment_path, d))]

            if not sub_dirs:
                print(f"Skipping {experiment}: no sub directories found")
                continue

            print(f"  Experiment: {experiment}")

            # Get system name from config if available
            system_name = system_name_from_dir
            if include_system and system_name is None:
                config_file = os.path.join(experiment_path, 'config.txt')
                system_name = get_system_from_config(config_file)

            # Process each sub directory
            for sub_dir in sub_dirs:
                # Extract sub value
                match = re.match(r'sub(\d+)', sub_dir)
                if not match:
                    continue

                sub = int(match.group(1))
                sub_path = os.path.join(experiment_path, sub_dir)
                run_id = f"{experiment}_sub{sub}"

                # Look for evaluation results
                results_file = os.path.join(sub_path, 'evaluation_results.txt')
                results = parse_evaluation_results(results_file)

                result_dict = {
                    'run_id': run_id,
                    'experiment': experiment,
                    'sub': sub,
                }

                if system_name:
                    result_dict['system'] = system_name

                if results is None:
                    print(f"    WARNING: No results found for sub{sub}")
                    result_dict.update({
                        'status': 'missing',
                        'precision': None,
                        'recall': None,
                        'f1': None,
                        'num_attractors': None,
                        'num_trajectories': None,
                        'num_successful': None,
                        'num_unsuccessful': None
                    })
                else:
                    result_dict.update({
                        'status': 'success',
                        'precision': results.get('precision'),
                        'recall': results.get('recall'),
                        'f1': results.get('f1'),
                        'num_attractors': results.get('num_attractors'),
                        'num_trajectories': results.get('num_trajectories'),
                        'num_successful': results.get('num_successful'),
                        'num_unsuccessful': results.get('num_unsuccessful')
                    })
                    print(f"    sub{sub}: F1={results.get('f1', 'N/A'):.4f}")

                all_results.append(result_dict)

    # Write CSV
    if all_results:
        # Determine fieldnames based on whether system is included
        if include_system and any('system' in r for r in all_results):
            fieldnames = [
                'run_id', 'system', 'experiment', 'sub', 'status',
                'precision', 'recall', 'f1',
                'num_attractors', 'num_trajectories',
                'num_successful', 'num_unsuccessful'
            ]
        else:
            fieldnames = [
                'run_id', 'experiment', 'sub', 'status',
                'precision', 'recall', 'f1',
                'num_attractors', 'num_trajectories',
                'num_successful', 'num_unsuccessful'
            ]

        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\n{'='*80}")
        print(f"Aggregation complete!")
        print(f"Total runs: {len(all_results)}")
        print(f"Successful runs: {sum(1 for r in all_results if r['status'] == 'success')}")
        print(f"Missing results: {sum(1 for r in all_results if r['status'] == 'missing')}")
        print(f"\nResults saved to: {output_csv}")
        print(f"{'='*80}")

        # Print summary statistics
        successful_results = [r for r in all_results if r['status'] == 'success' and r['f1'] is not None]
        if successful_results:
            f1_scores = [r['f1'] for r in successful_results]
            print(f"\nF1 Score Statistics:")
            print(f"  Mean: {sum(f1_scores) / len(f1_scores):.4f}")
            print(f"  Min:  {min(f1_scores):.4f}")
            print(f"  Max:  {max(f1_scores):.4f}")

            # Find best configuration
            best_result = max(successful_results, key=lambda r: r['f1'])
            print(f"\nBest Configuration:")
            if 'system' in best_result:
                print(f"  System: {best_result['system']}")
            print(f"  Run ID: {best_result['run_id']}")
            print(f"  Experiment: {best_result['experiment']}")
            print(f"  Sub: {best_result['sub']}")
            print(f"  F1 Score: {best_result['f1']:.4f}")
            print(f"  Precision: {best_result['precision']:.4f}")
            print(f"  Recall: {best_result['recall']:.4f}")

            # Group by system if available
            if 'system' in best_result:
                print(f"\nBest Configuration by System:")
                systems = set(r.get('system') for r in successful_results if r.get('system'))
                for system in sorted(systems):
                    system_results = [r for r in successful_results if r.get('system') == system]
                    if system_results:
                        best_for_system = max(system_results, key=lambda r: r['f1'])
                        print(f"  {system}:")
                        print(f"    Experiment: {best_for_system['experiment']}, Sub: {best_for_system['sub']}")
                        print(f"    F1: {best_for_system['f1']:.4f}, Precision: {best_for_system['precision']:.4f}, Recall: {best_for_system['recall']:.4f}")

    else:
        print("No results found!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Aggregate grid search evaluation results into a CSV file'
    )
    parser.add_argument(
        '--grid_search_dir',
        type=str,
        default='output',
        help='Directory containing grid search results (default: output/)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: grid_search_results.csv in grid_search_dir)'
    )
    parser.add_argument(
        '--no_system',
        action='store_true',
        help='Do not extract system information from configs'
    )

    args = parser.parse_args()

    aggregate_results(args.grid_search_dir, args.output, include_system=not args.no_system)
