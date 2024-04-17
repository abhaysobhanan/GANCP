import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl.nn.functional import edge_softmax


def get_norm_layer(norm_method, dim):
    if norm_method == 'batch':
        norm = nn.BatchNorm1d(dim)
    elif norm_method == 'layer':
        norm = nn.LayerNorm(dim)
    elif norm_method == 'none':
        norm = nn.Identity()  # kinda placeholder
    return norm


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 d_model: int = 128,
                 num_heads: int = 8,
                 d_head: int = 128,
                 d_ff: int = 512,
                 logit_clamp: float = 5.0,
                 normalization: str = 'layer'):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.proj_q = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.proj_k = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.proj_v = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.proj_o = nn.Linear(num_heads * d_head, d_model, bias=False)

        self.norm1 = get_norm_layer(normalization, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = get_norm_layer(normalization, d_model)
        self.logit_clamp = logit_clamp

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor):
        with g.local_scope():
            # MHA part
            g.ndata['h'] = feat
            g.apply_edges(self.update_edges)
            g.edata['av'] = edge_softmax(g, g.edata['u']) * g.edata['v']
            g.update_all(dgl.function.copy_e('av', 'av'), self.reduce_func)

            # Normalization and FFN
            uh = self.norm1(feat + g.ndata['uh'])
            ffn_uh = self.ffn(uh)
            uh = self.norm2(uh + ffn_uh)
            return uh

    def update_edges(self, edges):
        q = self.proj_q(edges.dst['h'])
        k = self.proj_k(edges.src['h'])
        u = (q * k).reshape(-1, self.num_heads, self.d_head)  # [#. edges x num_heads x d_head]
        u = u.sum(dim=-1, keepdim=True) / np.sqrt(self.d_head)  # [#. edges x num_heads x 1]
        u = u.clamp(min=-self.logit_clamp, max=self.logit_clamp)
        v = self.proj_v(edges.src['h']).reshape(-1, self.num_heads, self.d_head)  # [#. edges x num_heads x d_head]
        return {'u': u, 'v': v}

    def reduce_func(self, nodes):
        sum_av = nodes.mailbox['av'].sum(dim=1)  # [#. nodes x num_heads x d_head]
        sum_av = sum_av.flatten(start_dim=1)  # [#. nodes x (num_heads x d_head)]
        uh = self.proj_o(sum_av)
        return {'uh': uh}
