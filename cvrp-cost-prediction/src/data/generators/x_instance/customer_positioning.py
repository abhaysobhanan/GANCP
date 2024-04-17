import numpy as np
from scipy.spatial.distance import cdist

from src.data.generators.x_instance.utils import idx_to_rowcol


def get_random_customer_pos_idx(n, depot_idx, grid):
    nrow = ncol = grid[1] - grid[0] + 1
    idx_list = np.delete(np.arange(nrow * ncol), depot_idx)
    customer_idx = np.random.choice(idx_list, size=n, replace=False)
    return customer_idx


def get_clustered_customer_pos_idx(n, excluding_idx, grid):
    nrow = ncol = grid[1] - grid[0] + 1
    idx_list = np.setdiff1d(np.arange(nrow * ncol), excluding_idx)

    # sample the seeds - the centers of clusters
    s = np.random.randint(3, 8 + 1)
    seed_idx = np.random.choice(idx_list, size=s, replace=False)
    idx_list = np.setdiff1d(idx_list, seed_idx)

    seed_row, seed_col = idx_to_rowcol(seed_idx, grid)
    candidate_row, candidate_col = idx_to_rowcol(idx_list, grid)
    seed_pos = np.stack((seed_row, seed_col), axis=-1)
    candidate_pos = np.stack((candidate_row, candidate_col), axis=-1)
    dist_cs = cdist(candidate_pos, seed_pos)  # [candidates x seed]
    odd = np.exp(-dist_cs / 40.0).sum(axis=-1)

    customer_idx = []
    for _ in range(n - s):
        prob = odd / odd.sum()
        customer_id = int(np.random.choice(prob.shape[0], 1, replace=False, p=prob))
        customer_idx.append(idx_list[customer_id])
        odd[customer_id] = 0.0

    customer_idx = seed_idx.tolist() + customer_idx
    return customer_idx


def get_customer_idx(mode, n, depot_idx, grid):
    if mode == 'random':
        pos_idx = get_random_customer_pos_idx(n, depot_idx, grid)
    elif mode == 'clustered':
        pos_idx = get_clustered_customer_pos_idx(n, depot_idx, grid)
    elif mode == 'random-clustered':
        random_n = int(n * 0.5)
        random_customer_idx = get_random_customer_pos_idx(random_n, depot_idx, grid)
        excluding_idx = [depot_idx] + random_customer_idx.tolist()
        clustered_customer_idx = get_clustered_customer_pos_idx(n - random_n, excluding_idx, grid)
        pos_idx = random_customer_idx.tolist() + clustered_customer_idx
    else:
        raise NotImplementedError("Given customer generation mode '{}' is not implemented.".format(mode))
    return pos_idx
