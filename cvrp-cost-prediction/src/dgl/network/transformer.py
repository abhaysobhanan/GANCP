import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch.utils import Sequential

from src.dgl.nn.mha import MultiHeadAttention
from src.dgl.nn.readout import BasicReadout, GeneralizedReadout


class Transformer(nn.Module):

    def __init__(self,
                 in_dim: int,
                 latent_dim: int,
                 n_layers: int,
                 readout: str = 'mean',
                 layer_share: bool = False,
                 mha_params: dict = {}):
        super(Transformer, self).__init__()
        self.encoder = nn.Linear(in_dim, latent_dim)

        self.layer_share = layer_share
        if self.layer_share:
            self.mha = MultiHeadAttention(d_model=latent_dim, **mha_params)
        else:
            self.mha = Sequential(
                *tuple([MultiHeadAttention(d_model=latent_dim,
                                           **mha_params) for _ in range(n_layers)])
            )

        self.decoder = nn.Linear(latent_dim, 1)
        self.readout_method = readout
        if readout == 'generalized':
            self.readout = GeneralizedReadout()
        else:
            self.readout = BasicReadout(readout)
        self.n_layers = n_layers

    def forward(self, graph: dgl.graph, nf: torch.tensor) -> torch.tensor:
        unf = self.encoder(nf)

        if self.layer_share:
            for _ in range(self.n_layers):
                unf = self.mha(graph, unf)
        else:
            unf = self.mha(graph, unf)

        if self.readout_method == 'generalized':
            unf = self.readout(graph, unf)
            pred = self.decoder(unf)
        else:
            unf = self.decoder(unf)
            pred = self.readout(graph, unf)
        return pred
