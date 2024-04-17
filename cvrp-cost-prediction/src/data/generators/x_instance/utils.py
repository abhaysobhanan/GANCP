from os.path import join

import numpy as np


def idx_to_rowcol(idx, grid):
    row, col = np.divmod(idx, grid[1] + 1)
    return row, col


def rowcol_to_idx(rows, cols, grid):
    idx = rows * (grid[1] + 1) + cols
    return idx