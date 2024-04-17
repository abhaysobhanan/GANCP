from math import ceil

import numpy as np

from src.data.generators.x_instance.customer_positioning import get_customer_idx
from src.data.generators.x_instance.demand_distribution import get_demands
from src.data.generators.x_instance.depot_positioning import get_depot_idx
from src.data.generators.x_instance.utils import idx_to_rowcol


def generate_x_instance(n,
                        depot_positioning: str = 'random',
                        customer_positioning: str = 'random',
                        demand_distribution: str = 'CV',
                        r: float = 5.0,
                        grid=[0, 1000],
                        **kwargs):
    """
    Generate x-like instances from
    "New benchmark instances for the Capacitated Vehicle Routing Problem".
    """

    assert n > 2
    assert depot_positioning in ['center', 'eccentric', 'random', 'quadrant']
    assert customer_positioning in ['random', 'clustered', 'random-clustered']
    assert demand_distribution in ['U', 'CV', 'Q', 'SL']

    if kwargs.get('CV_range') is None:
        kwargs['CV_range'] = [0, 100]

    depot_pos_idx = get_depot_idx(depot_positioning, grid)
    customer_pos_idx = get_customer_idx(customer_positioning, n, depot_pos_idx, grid)
    demands = get_demands(demand_distribution, customer_pos_idx, grid, **kwargs)
    demands = np.concatenate([[0], demands])
    Q = ceil(r * demands.sum() / n)

    depot_coord = np.stack(idx_to_rowcol(depot_pos_idx, grid), axis=-1).reshape(1, 2)
    customer_coords = np.stack(idx_to_rowcol(customer_pos_idx, grid), axis=-1)

    # assume the first city is the depot
    coords = np.concatenate((depot_coord, customer_coords), axis=0)

    data = dict()
    data['x_coordinates'] = coords[:, 0]
    data['y_coordinates'] = coords[:, 1]
    data['service_times'] = np.zeros(n + 1)
    data['vehicle_capacity'] = Q
    data['depot'] = 0
    data['demands'] = demands
    return data
