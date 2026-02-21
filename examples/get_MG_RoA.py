from MORALS.systems.utils import get_system
from MORALS.models import *
from MORALS.dynamics_utils import DynamicsUtils
from MORALS.data_utils import DynamicsDataset
import argparse

import dytop.CMGDB_util as CMGDB_util
import dytop.RoA as RoA
import dytop.Grid as Grid
from dytop.PlotRoA import PlotRoA

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import shutil
import multiprocessing as mp
from functools import partial

from evaluate_RoA import evaluate_roa, TEST_SET_PATHS
from MORALS.grid_utils import valid_grid_gpu
import CMGDB


# ============================================================================
# PARALLEL F PRE-COMPUTATION
# ============================================================================

# Global worker state (initialized once per worker process)
_worker_dyn_utils = None
_worker_MG_util = None
_worker_config = None


class DynamicsUtilsCPU:
    """CPU-only version of DynamicsUtils for parallel workers."""

    def __init__(self, config):
        import torch
        from MORALS.systems.utils import get_system
        from MORALS.models import Encoder, Decoder, LatentDynamics

        self.system = get_system(config['system'])

        assert os.path.exists(config['model_dir']), "model expected"

        if config.get("use_limits", False):
            raise NotImplementedError
        else:
            self.X_min = np.loadtxt(os.path.join(config['model_dir'], 'X_min.txt'), delimiter=',')
            self.X_max = np.loadtxt(os.path.join(config['model_dir'], 'X_max.txt'), delimiter=',')

        # Force CPU device
        self.device = torch.device('cpu')

        # Load models directly to CPU
        self.encoder = torch.load(
            os.path.join(config['model_dir'], 'encoder.pt'),
            map_location='cpu', weights_only=False
        )
        self.decoder = torch.load(
            os.path.join(config['model_dir'], 'decoder.pt'),
            map_location='cpu', weights_only=False
        )
        self.dynamics = torch.load(
            os.path.join(config['model_dir'], 'dynamics.pt'),
            map_location='cpu', weights_only=False
        )

        self.encoder.eval()
        self.decoder.eval()
        self.dynamics.eval()

    def f_iterate_batch(self, z_batch, n_steps=1):
        """Batched dynamics iteration on CPU."""
        import torch
        with torch.no_grad():
            z = torch.from_numpy(z_batch.astype(np.float32))
            for _ in range(int(n_steps)):
                z = self.dynamics(z)
            return z.numpy()


def cell_id_to_rect(cell_id, grid):
    """
    Convert a cell ID to its rectangle bounds.

    Args:
        cell_id: Integer cell ID
        grid: dytop.Grid object

    Returns:
        List [lower_d0, ..., lower_dn, upper_d0, ..., upper_dn]
    """
    dim = grid.dim
    subdiv = grid.subdiv  # list of per-dimension subdivisions
    lower_bounds = grid.lower_bounds
    size_of_box = grid.size_of_box

    # Decode cell_id to coordinates using the bit-interleaving scheme.
    # This is the inverse of Grid.get_id_vertex, which encodes by iterating
    # count from subdivision down to 1, extracting the current LSB of
    # position_temp[j] and shifting right. The first extracted bit (LSB of
    # the coordinate) ends up as the MSB of the id. To invert, we extract
    # bits from the id LSB-first and place them at reversed bit positions
    # within each coordinate.
    subdivision = grid.subdivision
    bits_per_dim = [0] * dim
    for count in range(1, subdivision + 1):
        j = (count - 1) % dim
        bits_per_dim[j] += 1

    coords = [0] * dim
    visit_count = [0] * dim
    temp_id = cell_id
    for count in range(1, subdivision + 1):
        j = (count - 1) % dim
        bit = temp_id & 1
        coords[j] = coords[j] | (bit << (bits_per_dim[j] - 1 - visit_count[j]))
        visit_count[j] += 1
        temp_id >>= 1

    # Now coords[j] contains the cell coordinate for dimension j
    # Convert to rect bounds
    rect = []
    for d in range(dim):
        rect.append(lower_bounds[d] + coords[d] * size_of_box[d])
    for d in range(dim):
        rect.append(lower_bounds[d] + (coords[d] + 1) * size_of_box[d])

    return rect


def _init_worker(config_dict, number_of_steps):
    """Initialize worker process with its own dynamics model (CPU only)."""
    global _worker_dyn_utils, _worker_MG_util, _worker_config

    # Force CPU-only mode to avoid GPU memory conflicts between workers
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # Suppress output in workers
    import warnings
    warnings.filterwarnings("ignore")

    _worker_config = {
        'number_of_steps': number_of_steps,
    }

    # Create a CPU-only dynamics utils
    _worker_dyn_utils = DynamicsUtilsCPU(config_dict)
    _worker_MG_util = CMGDB_util.CMGDB_util()


def _compute_F_for_cell(args):
    """
    Worker function to compute F for a single cell.

    Args:
        args: tuple of (cell_id, rect, K_list, dim, attractor_type, redirect_target)
              attractor_type: 0=dynamics, 1=identity attractor, 2=redirect to target

    Returns:
        tuple of (cell_id, F_result)
    """
    global _worker_dyn_utils, _worker_MG_util, _worker_config

    cell_id, rect, K_list, dim, attractor_type, redirect_target = args
    K_array = np.array(K_list, dtype=np.float32)
    number_of_steps = _worker_config['number_of_steps']

    # Get all face points and batch evaluate
    X = _worker_MG_util.FacePoints(rect, 2)
    X_array = np.array(X, dtype=np.float32)

    if attractor_type == 1:
        # Identity attractor (success cells + nominal failure cell)
        # K expansion causes neighboring attractor cells to overlap,
        # merging them into a single Morse set
        Y_array = X_array
    elif attractor_type == 2:
        # Redirect attractor: all face points map to the nominal failure cell center
        target = np.array(redirect_target, dtype=np.float32)
        Y_array = np.tile(target, (X_array.shape[0], 1))
    else:
        # Non-attractor cell: run actual dynamics
        Y_array = _worker_dyn_utils.f_iterate_batch(X_array, n_steps=number_of_steps)

    # Compute bounds with K expansion
    Y_min = Y_array.min(axis=0)
    Y_max = Y_array.max(axis=0)
    rect_size = np.array([rect[d + dim] - rect[d] for d in range(dim)], dtype=np.float32)
    Y_l_bounds = (Y_min - K_array * rect_size).tolist()
    Y_u_bounds = (Y_max + K_array * rect_size).tolist()

    return (cell_id, Y_l_bounds + Y_u_bounds)


def precompute_F_parallel(config, cmgdb_grid, validity_grid, valid_list, K, number_of_steps, num_workers=None,
                          attractor_cell_ids=None, redirect_cell_ids=None, redirect_target=None):
    """
    Pre-compute F(rect) for all valid cells using multiprocessing.

    Args:
        config: Configuration dict for DynamicsUtils
        cmgdb_grid: dytop.Grid object with actual CMGDB bounds
        validity_grid: dytop.Grid object with [-1, 1] bounds for validity checking
        valid_list: List of booleans indicating valid cells (from validity_grid)
        K: List of Lipschitz constants per dimension
        number_of_steps: Number of dynamics iterations
        num_workers: Number of worker processes (default: CPU count)
        attractor_cell_ids: Set of cell IDs that use identity F (success + nominal failure)
        redirect_cell_ids: Set of cell IDs that redirect to the nominal failure cell
        redirect_target: Center coordinates of the nominal failure cell (list of floats)

    Returns:
        Dict mapping cell_id -> F_result (using cmgdb_grid cell IDs)
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    dim = cmgdb_grid.dim

    # Collect all cells that are valid (checking validity using validity_grid)
    print(f"       Generating rectangles for valid cells...")
    valid_cells = []
    total_cells = 2 ** cmgdb_grid.subdivision

    for cell_id in range(total_cells):
        # Get rect in CMGDB coordinate system
        rect = cell_id_to_rect(cell_id, cmgdb_grid)

        # Check validity using the validity_grid (matches original F behavior)
        center = [(rect[d] + rect[d + dim]) / 2 for d in range(dim)]
        validity_cell_id = validity_grid.point2cell(center)

        if validity_cell_id < len(valid_list) and valid_list[validity_cell_id]:
            if attractor_cell_ids and cell_id in attractor_cell_ids:
                valid_cells.append((cell_id, rect, K, dim, 1, None))
            elif redirect_cell_ids and cell_id in redirect_cell_ids:
                valid_cells.append((cell_id, rect, K, dim, 2, redirect_target))
            else:
                valid_cells.append((cell_id, rect, K, dim, 0, None))

    num_valid = len(valid_cells)
    print(f"       Found {num_valid:,} valid cells to pre-compute (out of {total_cells:,})")
    if attractor_cell_ids or redirect_cell_ids:
        num_identity = sum(1 for c in valid_cells if c[4] == 1)
        num_redirect = sum(1 for c in valid_cells if c[4] == 2)
        num_dynamics = num_valid - num_identity - num_redirect
        print(f"       Identity attractor cells: {num_identity:,}, redirect cells: {num_redirect:,}, dynamics cells: {num_dynamics:,}")

    if num_valid == 0:
        return {}

    # Create worker pool and compute
    print(f"       Starting {num_workers} worker processes...")
    start_time = datetime.now()

    # Use spawn context for clean process creation (avoids CUDA issues)
    ctx = mp.get_context('spawn')

    F_cache = {}
    completed = 0
    last_print_time = datetime.now()
    print_interval = 5.0

    with ctx.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(config, number_of_steps)
    ) as pool:
        # Use imap_unordered for better progress reporting
        for cell_id, F_result in pool.imap_unordered(_compute_F_for_cell, valid_cells, chunksize=100):
            F_cache[cell_id] = F_result
            completed += 1

            # Progress reporting
            now = datetime.now()
            if (now - last_print_time).total_seconds() >= print_interval:
                elapsed = (now - start_time).total_seconds()
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (num_valid - completed) / rate if rate > 0 else 0
                pct = 100 * completed / num_valid
                print(f"\r       Progress: {completed:,}/{num_valid:,} ({pct:.1f}%) - "
                      f"{rate:.0f} cells/sec - ETA: {remaining:.0f}s", end="", flush=True)
                last_print_time = now

    elapsed = (datetime.now() - start_time).total_seconds()
    rate = num_valid / elapsed if elapsed > 0 else 0
    print(f"\n       Pre-computed {num_valid:,} cells in {elapsed:.1f}s ({rate:.0f} cells/sec)")

    return F_cache


def create_lookup_F(F_cache, cmgdb_grid, dim, invalid_return):
    """
    Create a lookup-based F function for CMGDB.

    Args:
        F_cache: Dict mapping cell_id -> F_result (keyed by cmgdb_grid cell IDs)
        cmgdb_grid: dytop.Grid object with CMGDB bounds
        dim: Number of dimensions
        invalid_return: Return value for invalid cells

    Returns:
        Function F(rect) that does fast dict lookup
    """
    # Stats for monitoring
    stats = {'hits': 0, 'misses': 0}

    def F_lookup(rect):
        # Get center point and cell ID using CMGDB grid
        center = [(rect[d] + rect[d + dim]) / 2 for d in range(dim)]
        cell_id = cmgdb_grid.point2cell(center)

        if cell_id in F_cache:
            stats['hits'] += 1
            return F_cache[cell_id]
        else:
            stats['misses'] += 1
            return invalid_return

    F_lookup.stats = stats
    return F_lookup


def _patched_assign_morse_nodes2tiles(self):
    """
    Patched version of RoA.assign_morse_nodes2tiles that uses a set instead of list.
    Original has O(n²) complexity due to list operations; this is O(n).
    """
    # clone self.tiles_in_morse_sets we dont have to recompute
    self.dict_tiles = dict(self.tiles_in_morse_sets)

    # Use SET instead of list for O(1) membership test and removal
    self.S = self.vertices() - set(self.tiles_in_morse_sets.keys())

    while self.S:  # loop: assign morse node to a tile and remove it
        v = next(iter(self.S))  # get an arbitrary element
        # propagate to determine which morse node should be assign to tile v
        morse_node = self.propagate(v, self.map_graph.adjacencies(v))

    return self.dict_tiles


def _patched_propagate(self, u, adjacencies):
    """Patched propagate that works with set-based self.S"""
    adjacencies_morse_node = set()
    for w in adjacencies:
        morse_node = self.dict_tiles.get(w, None)
        if morse_node is None:
            morse_node = self.propagate(w, self.map_graph.adjacencies(w))
        adjacencies_morse_node.add(morse_node)

    max_nodes = list(self.MG.maximal(adjacencies_morse_node - {-1}))
    morse_node = max_nodes[0] if max_nodes else -1

    self.dict_tiles[u] = morse_node
    self.S.discard(u)  # O(1) set operation instead of O(n) list operation
    return morse_node


# Apply monkey patches to fix O(n²) -> O(n)
RoA.RoA.assign_morse_nodes2tiles = _patched_assign_morse_nodes2tiles
RoA.RoA.propagate = _patched_propagate

def get_mg_run_dir(base_output_dir, sub, Lips, validation_type, use_known_attractors=False, attractor_expansion=0):
    """Generate a unique directory name for this MG run based on parameters."""
    run_name = f"MG_sub{sub}_Lips{Lips}_{validation_type}"
    if use_known_attractors:
        run_name += "_attractors"
        if attractor_expansion > 0:
            run_name += f"_exp{attractor_expansion}"
    run_dir = os.path.join(base_output_dir, "mg_runs", run_name)
    return run_dir


def expand_cell_ids(cell_ids, grid, expansion=1):
    """Expand a set of cell IDs to include neighbors within `expansion` steps.

    For each cell, finds all cells within a Manhattan distance of `expansion`
    in grid coordinates.
    """
    if expansion <= 0 or not cell_ids:
        return cell_ids

    dim = grid.dim
    subdivision = grid.subdivision
    total_cells = 2 ** subdivision

    # Decode each cell_id to grid coordinates (inverse of get_id_vertex)
    bits_per_dim = [0] * dim
    for count in range(1, subdivision + 1):
        j = (count - 1) % dim
        bits_per_dim[j] += 1

    def cell_id_to_coords(cell_id):
        coords = [0] * dim
        visit_count = [0] * dim
        temp_id = cell_id
        for count in range(1, subdivision + 1):
            j = (count - 1) % dim
            bit = temp_id & 1
            coords[j] = coords[j] | (bit << (bits_per_dim[j] - 1 - visit_count[j]))
            visit_count[j] += 1
            temp_id >>= 1
        return coords

    def coords_to_cell_id(coords):
        position_temp = list(coords)
        cell_id = 0
        for count in range(subdivision, 0, -1):
            j = (count - 1) % dim
            cell_id = (cell_id << 1) | (position_temp[j] & 1)
            position_temp[j] >>= 1
        return cell_id

    # Get number of cells per dimension
    cells_per_dim = [0] * dim
    for count in range(1, subdivision + 1):
        j = (count - 1) % dim
        cells_per_dim[j] += 1
    max_coord = [2**b - 1 for b in cells_per_dim]

    expanded = set()
    for cell_id in cell_ids:
        base_coords = cell_id_to_coords(cell_id)
        # Generate all neighbors within expansion distance
        import itertools
        offsets = [range(-expansion, expansion + 1)] * dim
        for offset in itertools.product(*offsets):
            neighbor = [base_coords[d] + offset[d] for d in range(dim)]
            # Clamp to grid bounds
            if all(0 <= neighbor[d] <= max_coord[d] for d in range(dim)):
                expanded.add(coords_to_cell_id(neighbor))

    return expanded


class BoxMapStats:
    """Track statistics for BoxMap calls with progress indication."""
    def __init__(self, total_cells=0):
        self.total_calls = 0
        self.valid_calls = 0
        self.invalid_calls = 0
        self.total_cells = total_cells
        self.start_time = None
        self.last_print_time = None
        self.print_interval = 5.0  # Print every 5 seconds

    def start(self):
        self.start_time = datetime.now()
        self.last_print_time = self.start_time

    def update(self):
        self.total_calls += 1
        now = datetime.now()
        if self.last_print_time and (now - self.last_print_time).total_seconds() >= self.print_interval:
            self._print_progress()
            self.last_print_time = now

    def _print_progress(self):
        if self.total_cells > 0 and self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            pct = 100 * self.total_calls / self.total_cells
            rate = self.total_calls / elapsed if elapsed > 0 else 0
            remaining = (self.total_cells - self.total_calls) / rate if rate > 0 else 0
            print(f"\r       Progress: {self.total_calls:,}/{self.total_cells:,} ({pct:.1f}%) - "
                  f"{rate:.0f} cells/sec - ETA: {remaining:.0f}s", end="", flush=True)

    def print_stats(self):
        if self.total_calls > 0:
            print()  # New line after progress
            valid_pct = 100 * self.valid_calls / self.total_calls
            elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            print(f"       BoxMap calls: {self.total_calls:,} total, {self.valid_calls:,} valid ({valid_pct:.1f}%)")
            if elapsed > 0:
                print(f"       Average: {1000*elapsed/self.total_calls:.3f} ms/cell")


# Global stats tracker and pre-allocated arrays for invalid return
_box_map_stats = BoxMapStats()
_invalid_return_2d = [30000, 30000, 30000, 30000]  # Pre-allocated for 2D


import time

# Profiling accumulators
_profile_times = {'center': 0, 'point2cell': 0, 'facepoints': 0, 'f_batch': 0, 'bounds': 0, 'total_valid': 0}
_profile_count = 0

def print_profile_stats():
    """Print profiling statistics."""
    global _profile_count, _profile_times
    if _profile_count > 0:
        print(f"\n       Profiling ({_profile_count} valid cells):")
        for name, total in _profile_times.items():
            avg_ms = 1000 * total / _profile_count
            print(f"         {name}: {avg_ms:.3f} ms/cell")

def reset_profile_stats():
    global _profile_times, _profile_count
    _profile_times = {'center': 0, 'point2cell': 0, 'facepoints': 0, 'f_batch': 0, 'bounds': 0, 'total_valid': 0}
    _profile_count = 0


def BoxMapK_valid_batched(f_batch, rect, K, valid_list, point2cell, face_points_func, K_array, dim):
    """
    Batched version of BoxMapK_valid that evaluates all face points at once.
    Optimized to reduce per-call overhead.
    """
    global _profile_count, _profile_times

    _box_map_stats.update()

    # Fast validity check
    t0 = time.perf_counter()
    center = CMGDB.CenterPoint(rect)[0]
    t1 = time.perf_counter()
    cell_id = point2cell(center)
    t2 = time.perf_counter()

    if not valid_list[cell_id]:
        _box_map_stats.invalid_calls += 1
        return _invalid_return_2d if dim == 2 else [30000] * 2 * dim

    _box_map_stats.valid_calls += 1

    # Get all face points and batch evaluate
    t3 = time.perf_counter()
    X = face_points_func(rect, 2)
    X_array = np.array(X, dtype=np.float32)
    t4 = time.perf_counter()
    Y_array = f_batch(X_array)
    t5 = time.perf_counter()

    # Compute bounds with pre-allocated K_array
    Y_min = Y_array.min(axis=0)
    Y_max = Y_array.max(axis=0)
    rect_size = np.array([rect[d + dim] - rect[d] for d in range(dim)], dtype=np.float32)
    Y_l_bounds = (Y_min - K_array * rect_size).tolist()
    Y_u_bounds = (Y_max + K_array * rect_size).tolist()
    t6 = time.perf_counter()

    # Accumulate profiling (only for first 10000 valid cells to avoid overhead)
    if _profile_count < 10000:
        _profile_times['center'] += t1 - t0
        _profile_times['point2cell'] += t2 - t1
        _profile_times['facepoints'] += t4 - t3
        _profile_times['f_batch'] += t5 - t4
        _profile_times['bounds'] += t6 - t5
        _profile_times['total_valid'] += t6 - t0
        _profile_count += 1

    return Y_l_bounds + Y_u_bounds


def reset_box_map_stats(total_cells=0):
    """Reset the global stats tracker."""
    global _box_map_stats
    _box_map_stats = BoxMapStats(total_cells)
    _box_map_stats.start()


def get_box_map_stats():
    """Get the global stats tracker."""
    return _box_map_stats


def save_mg_params(output_dir, params):
    """Save Morse Graph computation parameters to a file for reproducibility."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fname = os.path.join(output_dir, "mg_params.txt")
    with open(fname, "w") as f:
        f.write(f"# Morse Graph Parameters\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    return fname


def write_experiments(morse_graph, experiment_name, output_dir, name="MG_attractors.txt"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    name = f"{output_dir}/{name}"

    with open(name, "w") as file:
        file.write(experiment_name)

        counting_attractors = 0
        list_attractors = []
        if morse_graph:
            S = set(morse_graph.vertices())
            while len(S) != 0:
                v = S.pop()
                if len(morse_graph.adjacencies(v)) == 0:
                    counting_attractors += 1
                    list_attractors.append(v)

        file.write(f":{list_attractors},{counting_attractors}\n")


def compute_roa(map_graph, morse_graph, lower_bounds, upper_bounds, output_dir, low_dims=2):
    print("       Building RoA data structure...")
    startTime = datetime.now()

    roa = RoA.RoA(map_graph, morse_graph)
    print(f"       RoA built ({datetime.now() - startTime})")

    roa.dir_path = os.path.join(os.getcwd(), output_dir)

    print("       Saving RoA to file...")
    roa.save_file("MG")

    if low_dims == 2:
        print("       Generating 2D RoA plot...")
        fig, ax = PlotRoA(lower_bounds, upper_bounds,
                    from_file="MG", dir_path=output_dir)

        out_pic = os.path.join(os.getcwd(), output_dir, "MG_RoA_")
        plt.savefig(out_pic, bbox_inches='tight')
        print(f"       Plot saved to {out_pic}")

def main(config, experiment_name, sub, Lips, validation_type, name_out, num_workers=None, use_known_attractors=False, attractor_expansion=0):
    total_start = datetime.now()

    # Create a unique directory for this MG run
    base_output_dir = config['output_dir'].rstrip('/')
    mg_run_dir = get_mg_run_dir(base_output_dir, sub, Lips, validation_type, use_known_attractors, attractor_expansion)

    if not os.path.exists(mg_run_dir):
        os.makedirs(mg_run_dir)

    # Create a modified config for this run directory
    # Update output_dir to point to mg_run_dir so MorseGraphOutputProcessor can find files
    run_config = config.copy()
    run_config['output_dir'] = mg_run_dir + '/'
    run_config['base_model_dir'] = base_output_dir  # Keep reference to original model

    config_dst = os.path.join(mg_run_dir, 'config.txt')
    with open(config_dst, 'w') as f:
        f.write(repr(run_config))

    print("\n" + "=" * 60)
    print("MORSE GRAPH COMPUTATION")
    print("=" * 60)
    print(f"System: {config['system']}")
    print(f"Subdivision: {sub}")
    print(f"Validation type: {validation_type}")
    print(f"Lipschitz increase: {Lips}%")
    print(f"Base model dir: {base_output_dir}")
    print(f"MG run output: {mg_run_dir}")
    print("=" * 60)

    # Step 1: Load dynamics model
    print("\n[1/6] Loading dynamics model...")
    step_start = datetime.now()
    dyn_utils = DynamicsUtils(config)
    print(f"       Done ({datetime.now() - step_start})")

    MG_util = CMGDB_util.CMGDB_util()
    system = get_system(config['system'], config['high_dims'])

    sb = sub
    number_of_steps = np.ceil(12 / config['step'])
    if config['system'] == "discrete_map":
        number_of_steps = 1
    print(f"       Dynamics iterations per step: {int(number_of_steps)}")

    subdiv_init = subdiv_min = subdiv_max = sb

    # Step 2: Get bounds
    print("\n[2/6] Computing state space bounds...")
    step_start = datetime.now()
    if type(system.get_true_bounds()) == type(NotImplementedError):
        a = np.loadtxt(os.path.join(config['model_dir'], 'X_min.txt'), delimiter=',').tolist()
        b = np.loadtxt(os.path.join(config['model_dir'], 'X_max.txt'), delimiter=',').tolist()
        system.state_bounds = np.array([a,b]).T

    lower_bounds_original_space = system.get_true_bounds()[:,0].tolist()
    upper_bounds_original_space = system.get_true_bounds()[:,1].tolist()
    print(f"       Original space bounds: {lower_bounds_original_space} to {upper_bounds_original_space}")

    dim_latent_space = config['low_dims']
    lower_bounds = [-1]*dim_latent_space
    upper_bounds = [1]*dim_latent_space
    print(f"       Latent space dims: {dim_latent_space}")
    print(f"       Done ({datetime.now() - step_start})")

    grid = Grid.Grid(lower_bounds, upper_bounds, sb)

    # Step 3: Generate validation samples
    print(f"\n[3/6] Generating validation samples (type={validation_type})...")
    step_start = datetime.now()
    if validation_type == 'uniform':
        grid_original_space = Grid.Grid(lower_bounds_original_space, upper_bounds_original_space, sb + 2)
        original_space_sample = grid_original_space.uniform_sample()
        print(f"       Generated {len(original_space_sample)} uniform grid samples")
        original_space_sample = system.transform(original_space_sample)

    elif validation_type == 'random':
        num_samples = 2**(sb + 8)
        print(f"       Generating {num_samples} random samples...")
        original_space_sample = system.sample_state(num_samples)

    else:
        print("       Loading trajectory dataset...")
        dataset = DynamicsDataset(config)
        latent_space_sample = dyn_utils.encode(dataset.Xt.numpy(), normalize=False)
        original_space_sample = None
        print(f"       Loaded {len(latent_space_sample)} samples from trajectories")

    print(f"       Done ({datetime.now() - step_start})")

    # Step 4: Encode samples to latent space
    print("\n[4/6] Encoding samples to latent space...")
    step_start = datetime.now()
    if original_space_sample is not None:
        latent_space_sample = dyn_utils.encode(original_space_sample)

    lower_bounds = np.min(latent_space_sample, axis=0).tolist()
    upper_bounds = np.max(latent_space_sample, axis=0).tolist()
    print(f"       Encoded {latent_space_sample.shape[0]} samples")
    print(f"       Latent bounds: [{lower_bounds}] to [{upper_bounds}]")
    print(f"       Done ({datetime.now() - step_start})")

    # Step 5: Build valid grid (GPU-accelerated)
    # The validity_grid uses [-1, 1] bounds for validity checking
    print("\n[5/6] Building valid grid for latent space (GPU)...")
    step_start = datetime.now()
    print(f"       Processing {len(latent_space_sample)} samples...")
    validity_grid = grid  # grid was created with [-1, 1] bounds
    valid_grid_latent_space = valid_grid_gpu(latent_space_sample, validity_grid, batch_size=10_000_000)
    num_valid = np.sum(valid_grid_latent_space)
    num_total = len(valid_grid_latent_space)
    print(f"       Valid cells: {num_valid}/{num_total} ({100*num_valid/num_total:.1f}%)")
    print(f"       Done ({datetime.now() - step_start})")

    phase_periodic = [False, False]
    K = [1 + Lips/100] * dim_latent_space
    K_array = np.array(K, dtype=np.float32)  # Pre-compute for efficiency

    # Total cells and invalid return value
    total_cells = 2 ** sb
    invalid_return = [30000] * 2 * dim_latent_space

    base_name = f"{mg_run_dir}/MG"
    MG_util.dir_path = ""

    # Create CMGDB grid with actual encoded bounds
    cmgdb_grid = Grid.Grid(lower_bounds, upper_bounds, sb)

    # Step 5b: Build known attractor cell sets (if enabled)
    identity_attractor_ids = set()
    redirect_cell_ids = set()
    redirect_target = None
    if use_known_attractors:
        from collections import Counter
        print("\n[5b] Building known attractor cell sets from trajectory final states...")
        step_start = datetime.now()

        # Load trajectory filenames
        if 'shuffled_indices_fname' in config:
            with open(config['shuffled_indices_fname'], 'r') as f_idx:
                all_files = [line.strip() for line in f_idx.readlines()]
        else:
            all_files = [f for f in os.listdir(config['data_dir']) if f.endswith('.txt')]

        # Load labels
        if 'labels_fname' in config:
            labels_data = np.loadtxt(config['labels_fname'], delimiter=',')
            if labels_data.ndim == 1:
                labels = labels_data.astype(int)
            else:
                labels = labels_data[:, -1].astype(int)
        else:
            # No labels: treat all as success (identity attractors only)
            labels = np.ones(len(all_files), dtype=int)

        # Extract final state from each trajectory
        subsample = config.get('subsample', 1)
        final_states_raw = []
        valid_labels = []
        for i, fname in enumerate(all_files):
            fpath = os.path.join(config['data_dir'], fname)
            if not os.path.isfile(fpath):
                continue
            data = np.loadtxt(fpath, delimiter=',')
            if len(data) == 0:
                continue
            indices = np.arange(data.shape[0])
            subsampled = data[indices % subsample == 0]
            final_states_raw.append(subsampled[-1])
            valid_labels.append(labels[i] if i < len(labels) else 0)

        final_states_raw = np.array(final_states_raw)
        valid_labels = np.array(valid_labels)
        print(f"       Loaded {len(final_states_raw)} trajectory final states "
              f"({np.sum(valid_labels == 1)} success, {np.sum(valid_labels == 0)} failure)")

        # Transform and encode to latent space
        final_states_transformed = system.transform(final_states_raw)
        final_states_latent = dyn_utils.encode(final_states_transformed)

        # Success cells: identity attractors
        success_cell_ids = set()
        for pt in final_states_latent[valid_labels == 1]:
            cell_id = cmgdb_grid.point2cell(pt.tolist())
            success_cell_ids.add(cell_id)

        # Failure cells: find most common cell (nominal failure attractor)
        failure_cell_counts = Counter()
        for pt in final_states_latent[valid_labels == 0]:
            cell_id = cmgdb_grid.point2cell(pt.tolist())
            failure_cell_counts[cell_id] += 1

        if failure_cell_counts:
            nominal_failure_cell_id, nominal_count = failure_cell_counts.most_common(1)[0]
            # Compute center of nominal failure cell
            nominal_rect = cell_id_to_rect(nominal_failure_cell_id, cmgdb_grid)
            redirect_target = [(nominal_rect[d] + nominal_rect[d + dim_latent_space]) / 2
                               for d in range(dim_latent_space)]

            # Identity attractors: success cells + nominal failure cell
            identity_attractor_ids = success_cell_ids | {nominal_failure_cell_id}
            # Redirect cells: all other failure cells
            redirect_cell_ids = set(failure_cell_counts.keys()) - {nominal_failure_cell_id}

            print(f"       Success attractor cells: {len(success_cell_ids)}")
            print(f"       Nominal failure cell: {nominal_failure_cell_id} "
                  f"({nominal_count} trajectories, center={[f'{x:.4f}' for x in redirect_target]})")
            print(f"       Failure redirect cells: {len(redirect_cell_ids)}")
        else:
            identity_attractor_ids = success_cell_ids
            print(f"       Success attractor cells: {len(success_cell_ids)} (no failure data)")

        # Expand attractor regions to include neighbors
        if attractor_expansion > 0:
            pre_expand = len(identity_attractor_ids)
            identity_attractor_ids = expand_cell_ids(identity_attractor_ids, cmgdb_grid, attractor_expansion)
            print(f"       Expanded identity cells: {pre_expand} -> {len(identity_attractor_ids)} (expansion={attractor_expansion})")

        print(f"       Done ({datetime.now() - step_start})")

    # Step 6: Pre-compute F values in parallel, then compute Morse Graph
    print("\n[6/6] Computing Morse Graph...")
    print(f"       Subdivision levels: min={subdiv_min}, max={subdiv_max}, init={subdiv_init}")
    print(f"       Lipschitz constant K={K}")
    print(f"       Total cells: {total_cells:,}")
    print(f"       CMGDB bounds: {lower_bounds} to {upper_bounds}")

    # Determine number of workers
    actual_workers = num_workers if num_workers is not None else mp.cpu_count()
    print(f"\n       [6a] Pre-computing F values with {actual_workers} workers...")
    step_start = datetime.now()

    F_cache = precompute_F_parallel(
        config, cmgdb_grid, validity_grid, valid_grid_latent_space, K, number_of_steps, actual_workers,
        attractor_cell_ids=identity_attractor_ids, redirect_cell_ids=redirect_cell_ids,
        redirect_target=redirect_target
    )
    precompute_time = datetime.now() - step_start
    print(f"       Pre-computation done ({precompute_time})")

    # Create lookup-based F function using CMGDB grid
    F = create_lookup_F(F_cache, cmgdb_grid, dim_latent_space, invalid_return)

    print(f"\n       [6b] Running CMGDB with cached F values...")
    step_start = datetime.now()
    morse_graph, map_graph = MG_util.run_CMGDB(
        subdiv_min, subdiv_max, lower_bounds, upper_bounds, phase_periodic, F, base_name, subdiv_init)
    cmgdb_time = datetime.now() - step_start
    print(f"       CMGDB completed ({cmgdb_time})")
    print(f"       F lookup stats: {F.stats['hits']:,} hits, {F.stats['misses']:,} misses")

    # Save results
    print("\n" + "-" * 60)
    print("Saving results...")
    write_experiments(morse_graph, experiment_name, mg_run_dir)

    # Save parameters for reproducibility
    mg_params = {
        'base_model_dir': base_output_dir,
        'sub': sub,
        'Lips': Lips,
        'validation_type': validation_type,
        'number_of_steps': int(number_of_steps),
        'K': K,
        'lower_bounds_latent': lower_bounds,
        'upper_bounds_latent': upper_bounds,
        'lower_bounds_original': lower_bounds_original_space,
        'upper_bounds_original': upper_bounds_original_space,
        'num_valid_cells': int(num_valid),
        'total_cells': int(num_total),
        'valid_cell_percent': f"{100*num_valid/num_total:.2f}",
        'use_known_attractors': use_known_attractors,
        'num_identity_attractor_cells': len(identity_attractor_ids),
        'num_redirect_attractor_cells': len(redirect_cell_ids),
    }
    if morse_graph:
        mg_params['num_morse_vertices'] = morse_graph.num_vertices()
    params_file = save_mg_params(mg_run_dir, mg_params)
    print(f"       Parameters saved to: {params_file}")

    if morse_graph:
        num_vertices = morse_graph.num_vertices()
        print(f"       Morse Graph vertices: {num_vertices}")

    if RoA:
        print("\nComputing Region of Attraction...")
        compute_roa(map_graph, morse_graph, lower_bounds, upper_bounds, mg_run_dir, config['low_dims'])

    print("\n" + "=" * 60)
    print(f"MORSE GRAPH COMPLETED in {datetime.now() - total_start}")
    print(f"Output saved to: {mg_run_dir}")
    print("=" * 60)

    # Run evaluation if test set is available for this system
    if config['system'] in TEST_SET_PATHS:
        print("\nRunning RoA evaluation...")
        eval_start = datetime.now()
        results = evaluate_roa(mg_run_dir)
        if results is not None:
            # Save evaluation results
            results_file = os.path.join(mg_run_dir, "evaluation_results.txt")
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
            print(f"\nEvaluation completed in {datetime.now() - eval_start}")
            print(f"Results saved to: {results_file}")
    else:
        print(f"\nSkipping evaluation: no test set available for system '{config['system']}'")
        print(f"Supported systems: {list(TEST_SET_PATHS.keys())}")

    print("\n" + "=" * 60)
    print(f"ALL DONE in {datetime.now() - total_start}")
    print("=" * 60 + "\n")

if __name__ == "__main__":

    sub = 14
    parser = argparse.ArgumentParser(description='Compute Morse Graph and Region of Attraction')
    parser.add_argument('--output_path', help='Output path containing config.txt', type=str, required=True)
    parser.add_argument('--name_out', help='Name of the output file', type=str, default='out_exp')
    parser.add_argument('--sub', help='Subdivision level (default: 14)', type=int, default=sub)
    parser.add_argument('--validation_type', help='Validation type: uniform, random, or trajectories', type=str, default='random')
    parser.add_argument('--Lips', help='Lipschitz constant increase percentage (default: 10)', type=float, default=10)
    parser.add_argument('--workers', help='Number of parallel workers (default: CPU count)', type=int, default=None)
    parser.add_argument('--use_known_attractors',
                        help='Use trajectory final states as known attractor cells',
                        action='store_true', default=False)
    parser.add_argument('--attractor_expansion', help='Expand attractor cells by N neighbors (default: 0)', type=int, default=0)

    args = parser.parse_args()

    print(f"\nLoading config from: {os.path.join(args.output_path, 'config.txt')}")
    with open(os.path.join(args.output_path, 'config.txt')) as f:
        config = eval(f.read())

    experiment_name = f"{config['experiment']}&{config['num_layers']}&{config['data_dir'][5::]}&{config['step']}&{args.sub}"

    main(config, experiment_name, args.sub, args.Lips, args.validation_type, args.name_out, args.workers, args.use_known_attractors, args.attractor_expansion)

    # if os.path.exists(config['model_dir']):
    #     main(args, config, experiment_name)
    # else:
    #     write_experiments(False, experiment_name, config['output_dir'])