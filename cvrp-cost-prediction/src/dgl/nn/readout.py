import dgl

import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_softmax


class BasicReadout(nn.Module):
    """
    a NN module wrapper class for graph readout
    """

    def __init__(self, op):
        super(BasicReadout, self).__init__()
        self.op = op

    def forward(self, g, x):
        with g.local_scope():
            g.ndata['feat'] = x
            rd = dgl.readout_nodes(g, 'feat', op=self.op)
            return rd


class GeneralizedReadout(nn.Module):
    '''
    https://arxiv.org/pdf/2009.09919.pdf
    https://github.com/hypnopump/generalized-readout-phase/blob/master/molhiv_power_good_beta_p10_beta0.ipynb
    '''

    def __init__(self, family="softmax", p=1.0, beta=1.0,
                 trainable_p=False, trainable_beta=False):
        super(GeneralizedReadout, self).__init__()

        self.family = family
        self.base_p = p
        self.base_beta = beta
        self.trainable_p = trainable_p
        self.trainable_beta = trainable_beta
        # define params
        self.p = torch.nn.Parameter(torch.tensor([p]), requires_grad=trainable_p)
        self.beta = torch.nn.Parameter(torch.tensor([beta]), requires_grad=trainable_beta)

    def forward(self, graph, x, bsize=None):
        r"""Args:
            x (Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
            batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
                B-1\}}^N`, which assigns each node to a specific example.
            size (int, optional): Batch-size :math:`B`.
                Automatically calculated if not given. (default: :obj:`None`)
        :rtype: :class:`Tensor`
        """
        batch = torch.repeat_interleave(torch.arange(graph.batch_size, device=x.device),
                                        graph.batch_num_nodes())
        bsize = int(batch.max().item() + 1) if bsize is None else bsize
        n_nodes = graph.batch_num_nodes()
        if self.family == "softmax":
            out = scatter_softmax(self.p * x.detach(), batch, dim=0)
            return scatter_add(x * out,
                               batch, dim=0, dim_size=bsize) * n_nodes.view(-1, 1) / (1 + self.beta * (n_nodes - 1)).view(-1, 1)

        elif self.family == "power":
            # numerical stability - avoid powers of large numbers or negative ones
            min_x, max_x = 1e-7, 1e+3
            torch.clamp_(x, min_x, max_x)
            out = scatter_add(torch.pow(x, self.p),
                              batch, dim=0, dim_size=bsize) / (1 + self.beta * (n_nodes - 1))
            torch.clamp_(out, min_x, max_x)
            return torch.pow(out, 1 / self.p)

    def reset_parameters(self):
        if self.p and torch.is_tensor(self.p):
            self.p.data.fill_(self.base_p)
        if self.beta and torch.is_tensor(self.beta):
            self.beta.data.fill_(self.base_beta)

    def __repr__(self):
        return "Generalized Aggr-Mean-Max global pooling layer with params:" + \
               str({"family": self.family,
                    "base_p": self.base_p,
                    "base_beta": self.base_beta,
                    "trainable_p": self.trainable_p,
                    "trainable_beta": self.trainable_beta})