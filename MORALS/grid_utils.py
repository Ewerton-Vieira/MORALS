"""
GPU-accelerated grid utilities for valid_grid computation.
Produces identical results to dytop.Grid.valid_grid but much faster.
"""

import torch
import numpy as np


def valid_grid_gpu(data, grid, batch_size=10_000_000, device=None):
    """
    GPU-accelerated version of grid.valid_grid with neighbors=True.

    Args:
        data: numpy array of shape (N, dim) - points in latent space
        grid: dytop.Grid object with lower_bounds, upper_bounds, subdiv, subdivision
        batch_size: number of samples to process per batch (default 10M for ~2GB memory)
                    Memory per batch ≈ batch_size * 3^dim * dim * 8 bytes (for neighbor coords)
                    For 2D: 10M batch ≈ 1.4GB, 40M batch ≈ 5.6GB
        device: torch device (default: cuda if available, else cpu)

    Returns:
        list of booleans of length 2^subdivision, identical to grid.valid_grid(data)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dim = grid.dim
    subdivision = grid.subdivision
    subdiv = grid.subdiv  # per-dimension subdivisions

    # Grid parameters as tensors
    lower_bounds = torch.tensor(grid.lower_bounds, dtype=torch.float32, device=device)
    size_of_box = torch.tensor(grid.size_of_box, dtype=torch.float32, device=device)
    subdiv_tensor = torch.tensor(subdiv, dtype=torch.int64, device=device)
    max_coords = (2 ** subdiv_tensor - 1).to(torch.int64)

    # Output: valid cell flags
    num_cells = 2 ** subdivision
    valid_flags = torch.zeros(num_cells, dtype=torch.bool, device=device)

    # Generate neighbor offsets: all combinations of {-1, 0, 1} for each dimension
    # For 2D: [[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]]
    offsets = torch.cartesian_prod(*[torch.tensor([-1, 0, 1], device=device) for _ in range(dim)])
    num_neighbors = offsets.shape[0]  # 3^dim

    # Process in batches
    n_samples = data.shape[0]

    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_data = torch.tensor(data[batch_start:batch_end], dtype=torch.float32, device=device)

        # Compute cell coordinates: (point - lower) / size_of_box, then floor to int
        cell_coords = ((batch_data - lower_bounds) / size_of_box).to(torch.int64)

        # Clamp to valid range (handle boundary cases)
        cell_coords = torch.clamp(cell_coords, torch.zeros(dim, dtype=torch.int64, device=device), max_coords)

        # Expand for all neighbors: (batch, 1, dim) + (1, num_neighbors, dim)
        # Result: (batch, num_neighbors, dim)
        neighbor_coords = cell_coords.unsqueeze(1) + offsets.unsqueeze(0)

        # Clamp neighbor coordinates to valid range
        neighbor_coords = torch.clamp(neighbor_coords,
                                       torch.zeros(dim, dtype=torch.int64, device=device),
                                       max_coords)

        # Convert coordinates to cell IDs using get_id_vertex logic
        # This implements the bit-interleaving scheme from Grid.get_id_vertex
        cell_ids = coords_to_cell_ids(neighbor_coords.reshape(-1, dim), dim, subdivision, subdiv_tensor, device)

        # Mark all these cells as valid
        valid_flags.scatter_(0, cell_ids, torch.ones_like(cell_ids, dtype=torch.bool))

    # Convert to Python list of bools (matching original output format)
    return valid_flags.cpu().tolist()


def coords_to_cell_ids(coords, dim, subdivision, subdiv_tensor, device):
    """
    Convert cell coordinates to cell IDs using the CMGDB bit-interleaving scheme.

    This implements Grid.get_id_vertex:
        position_temp = list(position)
        count = subdivision
        id = 0
        while any(position_temp) or count:
            id <<= 1
            j = (count - 1) % dim
            id += position_temp[j] & 1
            position_temp[j] >>= 1
            count -= 1
        return id

    Args:
        coords: tensor of shape (N, dim) with integer cell coordinates
        dim: number of dimensions
        subdivision: total subdivision level
        subdiv_tensor: tensor of per-dimension subdivisions
        device: torch device

    Returns:
        tensor of shape (N,) with cell IDs
    """
    n = coords.shape[0]
    ids = torch.zeros(n, dtype=torch.int64, device=device)
    coords_temp = coords.clone()

    for count in range(subdivision, 0, -1):
        ids = ids << 1
        j = (count - 1) % dim
        ids = ids + (coords_temp[:, j] & 1)
        coords_temp[:, j] = coords_temp[:, j] >> 1

    return ids


def valid_grid_cpu_original(data, grid):
    """
    Wrapper around the original grid.valid_grid for comparison.
    """
    return grid.valid_grid(data, neighbors=True)
