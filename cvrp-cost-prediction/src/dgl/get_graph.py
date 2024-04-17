import dgl
import numpy as np
import torch

from src.utils.preprocess import get_normed_xy_and_scaler
from src.utils.vrp_utils import parse_vrp_file, parse_sol_file


def compute_dists(edges):
    src_coord, dst_coord = edges.src['coord'], edges.dst['coord']
    src_normed_coord, dst_normed_coord = edges.src['normed_coord'], edges.dst['normed_coord']
    dist = ((src_coord - dst_coord) ** 2).sum(dim=-1, keepdim=True).sqrt()
    normed_dist = ((src_normed_coord - dst_normed_coord) ** 2).sum(dim=-1, keepdim=True).sqrt()
    return {'dist': dist, 'normed_dist': normed_dist}


def get_knn_graph(instance_path=None,
                  coord=None, demand=None, q=None, cost=None,
                  k: int = 5,
                  depot_center: bool = True): 
    if instance_path is not None:
        coord, demand, q, problem = parse_vrp_file(instance_path)
        _, cost, _ = parse_sol_file(instance_path)

    if coord.shape[0] != demand.shape[0]:
        # append virtual depot demand as 0
        demand = np.concatenate([np.array([0]), demand])

    normed_x, normed_y, metadata = get_normed_xy_and_scaler(coord, depot_center)
    scaler = metadata['scaler']
    normed_demand = demand / q

    # generate graph
    n = coord.shape[0]
    g = dgl.knn_graph(torch.tensor(coord), k=k)

    # meta data
    g.ndata['coord'] = torch.tensor(coord).view(-1, 2)
    g.ndata['normed_coord'] = torch.cat([torch.tensor(normed_x).view(-1, 1),
                                         torch.tensor(normed_y).view(-1, 1)],
                                        dim=-1).float()

    g.ndata['demand'] = torch.tensor(demand).view(-1, 1)
    g.ndata['q'] = q * torch.ones(n, 1)
    # g.ndata['normed_demand'] = g.ndata['demand'] / g.ndata['q']
    g.ndata['scaler'] = scaler * torch.ones(n, 1)

    g.apply_edges(compute_dists)  # compute distances

    # depot masking
    is_depot = torch.zeros(n, 1)
    is_depot[0, :] = 1.0
    g.ndata['is_depot'] = is_depot

    # prepare node features
    # normed_x, normed_y, normed_demand, depot_mask
    node_feat = [torch.tensor(normed_x).view(-1, 1),
                 torch.tensor(normed_y).view(-1, 1),
                 torch.tensor(normed_demand).view(-1, 1),
                 is_depot]
    g.ndata['feat'] = torch.cat(node_feat, dim=-1).float()

    g.ndata['invariant_feat'] = torch.cat([torch.tensor(normed_demand).view(-1, 1),
                                           is_depot], dim=-1)

    cost_scaler = scaler * n
    metadata['cost_scaler'] = cost_scaler

    if cost is not None:
        g.ndata['cost'] = cost * torch.ones(n, 1)  # original cost
        g.ndata['label'] = cost / (scaler * n) * torch.ones(n, 1)  # normalized cost
        normed_cost = cost / cost_scaler
        metadata['normed_cost'] = normed_cost
        metadata['cost'] = cost

    return g, metadata
