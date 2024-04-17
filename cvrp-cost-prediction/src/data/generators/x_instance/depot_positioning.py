import numpy as np

from src.data.generators.x_instance.utils import rowcol_to_idx


def get_depot_idx(mode, grid):
    if mode == 'center':
        center_pos = int(0.5 * (grid[1] - grid[0]))
        idx = rowcol_to_idx(center_pos, center_pos, grid)
    elif mode == 'eccentric':
        idx = rowcol_to_idx(0, 0, grid)
    elif mode == 'random':
        row = int(np.random.randint(grid[0], grid[1], size=1))
        col = int(np.random.randint(grid[0], grid[1], size=1))
        idx = rowcol_to_idx(row, col, grid)
    elif mode == 'quadrant':
        center_pos = int(0.5 * (grid[1] - grid[0]))
        row = int(np.random.randint(grid[0], center_pos, size=1))
        col = int(np.random.randint(grid[0], center_pos, size=1))
        idx = rowcol_to_idx(row, col, grid)
    else:
        raise NotImplementedError("Given depot generation mode '{}' is not implemented.".format(mode))
    return idx
