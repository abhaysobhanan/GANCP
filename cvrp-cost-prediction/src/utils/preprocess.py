import numpy as np


def get_normed_xy_and_scaler(coords,
                             depot_center: bool = False):
    x, y = coords[:, 0], coords[:, 1]

    bot, left = min(y), min(x)
    if left < 0:
        x += left
    if bot < 0:  
        y += bot

    top, right = max(y), max(x)
    scaler = np.sqrt(top ** 2 + right ** 2)
    normed_x, normed_y = x / right, y / top

    if depot_center:
        normed_x -= normed_x[0]
        normed_y -= normed_y[0]

    metadata = {
        'right': right,
        'top': top,
        'depot': coords[0],
        'scaler': scaler,
    }
    return normed_x, normed_y, metadata
