import sys;

import dgl
import torch
from math import floor
import numpy as np

from time import perf_counter
from tqdm.auto import tqdm
import hygese as hgs
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.utils.test_util import set_seed
from src.data import generate_x_instance
from src.data.generators.cvrp_instance.cvrp_from_mdvrp import generate_cvrp_instances
from src.dgl.get_graph import get_knn_graph
from src.dgl.dataset import GraphDataset, GraphDataLoader
from src.dgl import Transformer

ap = hgs.AlgorithmParameters(timeLimit=1)
hgs_solver = hgs.Solver(parameters=ap, verbose=False)

n_problem = 100000 #000 #100000
n_range = [15, 45]

instances = []
problem_count = 0
while (problem_count < n_problem):
    cvrp_list = generate_cvrp_instances(n_range)
    for i in range(len(cvrp_list)):
        problem = cvrp_list[i]
        result = hgs_solver.solve_cvrp(problem, rounding=False)

        instance = dict()
        instance.update(problem)
        instance['cost'] = float(result.cost)
        instance['routes'] = result.routes
        instances.append(instance)
        problem_count += 1
        if problem_count%100 == 0:
            print("Train data generated: ", problem_count , "\n")


# gs, ys, metadata = [], [], []
# for ins in instances:
#     g, y, md = get_knn_graph(ins)
#     gs.append(g)
#     ys.append(y)
#     metadata.append(md)
def convert_to_train_data(data):
    coord = np.stack([data['x_coordinates'], data['y_coordinates']], axis=1)  # [N, 2]
    demand = data['demands']
    q = data['vehicle_capacity']
    cost = data.get('cost')
    return coord, demand, q, cost
    
gs, ys, metadata = [], [], []
for ins in instances:
    coord, demand, q, cost = convert_to_train_data(ins)
    g, y, md = get_knn_graph(coord=coord, demand=demand, q=q, cost=cost)
    gs.append(g)
    ys.append(y)
    metadata.append(md)

ys = torch.tensor(ys).float().view(-1, 1)
labels = {'ys': ys}
dgl.save_graphs('test.dgldat', gs, labels)

device = 'cpu' #torch.cuda.is_available() 
set_seed(2022, use_cuda='cpu' in device) #'cuda'

# set up dataset and dataloader
train_val_split = floor(n_problem * 0.8)
train_gs, val_gs = gs[:train_val_split], gs[train_val_split:]
train_ys, val_ys = ys[:train_val_split], ys[train_val_split:]

train_ds = GraphDataset(train_gs, train_ys)
train_dl = GraphDataLoader(train_ds, batch_size=4, shuffle=True)
val_dl = GraphDataLoader(GraphDataset(val_gs, val_ys), batch_size=128)

model = Transformer(in_dim=4, latent_dim=128, n_layers=4).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingWarmRestarts(opt, T_0=32)
loss_fn = torch.nn.MSELoss()
test_fn = torch.nn.MSELoss()

def MAPE(pred, target):
    return ((target - pred).abs() / target.abs()).mean() * 100


def evaluate(model, data_loader, eval_fns, device):
    model.eval()

    with torch.no_grad():
        preds, ys = [], []
        for g, y in data_loader:
            g, y = g.to(device), y.to(device)
            pred = model(g, g.ndata['feat'])
            preds.append(pred)
            ys.append(y)

        preds = torch.cat(preds, dim=0)
        ys = torch.cat(ys, dim=0)

        losses = [eval_fn(preds, ys).item() for eval_fn in eval_fns]

    model.train()
    return losses


def format_log(dict):
    msg = ''
    for k, v in dict.items():
        if isinstance(v, int):
            msg += '{}: {} | '.format(k, v)
        else:
            msg += '{}: {:.3e} | '.format(k, v)
    return msg


n_epoch = 10 #20
n_update = 0
eval_every = 5

best_test_mse = float('inf')
for i in range(n_epoch):
    for train_g, train_y in train_dl:
        train_g, train_y = train_g.to(device), train_y.to(device).float()

        start = perf_counter()
        train_pred_y = model(train_g, train_g.ndata['feat'].float())
        loss = loss_fn(train_pred_y, train_y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        fit_time = perf_counter() - start
        n_update += 1

        log_dict = {
            'loss': loss.item(),
            'fit_time': fit_time,
            'lr': opt.param_groups[0]['lr'],
            'epoch': i
        }

    if n_update % eval_every == 0:
        start = perf_counter()
        train_perf = evaluate(model, train_dl,
                              [test_fn, MAPE], device)
        val_perf = evaluate(model, val_dl,
                            [test_fn, MAPE], device)
        eval_time = perf_counter() - start
        log_dict['train_mse'] = train_perf[0]
        log_dict['train_mape'] = train_perf[1]
        log_dict['val_mse'] = val_perf[0]
        log_dict['val_mape'] = val_perf[1]
        log_dict['eval_time'] = eval_time
        print('{} th iter | '.format(i) + format_log(log_dict))

        if val_perf[0] < best_test_mse:
            best_test_mse = val_perf[0]
            torch.save(model.state_dict(), "model_best.pt".format(n_update))