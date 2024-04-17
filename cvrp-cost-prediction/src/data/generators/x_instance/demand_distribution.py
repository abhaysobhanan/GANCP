import numpy as np

from src.data.generators.x_instance.utils import idx_to_rowcol


def get_demands(mode,
                customer_idx,
                grid,
                **kwargs):
    n = len(customer_idx)
    if mode == 'U':
        demands = np.array([1] * n)
    elif mode == 'CV':
        demands = np.random.randint(*kwargs['CV_range'], size=n)
    elif mode == 'Q':
        rows, cols = idx_to_rowcol(customer_idx, grid)
        center_pos = int(0.5 * (grid[1] - grid[0]))

        demands = np.zeros(shape=n)
        # even quadrants
        quad1 = np.logical_and(rows >= center_pos, cols >= center_pos)
        quad3 = np.logical_and(rows <= center_pos, cols <= center_pos)
        even_quad = np.logical_or(quad1, quad3)

        # odd quadrants
        quad2 = np.logical_and(rows >= center_pos, cols <= center_pos)
        quad4 = np.logical_and(rows <= center_pos, cols >= center_pos)
        odd_quad = np.logical_or(quad2, quad4)

        demands[even_quad] = np.random.randint(1, 50 + 1, size=sum(even_quad))
        demands[odd_quad] = np.random.randint(51, 101 + 1, size=sum(odd_quad))
    elif mode == 'SL':
        n_small = int(n * np.random.uniform(0.7, 0.95))
        n_large = n - n_small
        demands = np.concatenate((np.random.randint(1, 10 + 1, size=n_small),
                                  np.random.randint(50, 100 + 1, size=n_large)), axis=0)
    else:
        raise NotImplementedError("Given demand generation mode '{}' is not implemented.".format(mode))

    return demands