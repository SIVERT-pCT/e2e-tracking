
import torch
from torch_geometric.data import Data
from src.utils.graph import MaskedHitGraph
from torch_geometric.nn import MessagePassing
from src.supervised.gnn.layers import RelationalModel, ObjectModel


class LGInteractionNetwork(MessagePassing):
    def __init__(self, node_features: int, edge_features: int, hidden_size: int, **kwargs):
        super(LGInteractionNetwork, self).__init__(aggr="max")
        self.R1 = RelationalModel(2*node_features + edge_features, hidden_size, hidden_size)
        self.O = ObjectModel(node_features + hidden_size, hidden_size, hidden_size)
        self.R2 = RelationalModel(hidden_size, 1, hidden_size)
        self.E = torch.Tensor()
        
    def forward(self, G: Data) -> torch.Tensor:
        x_tilde = self.propagate(G.edge_index, x=G.x, edge_attr=G.edge_attr, size=None)
        return self.R2(x_tilde).squeeze()

    def message(self, x_i, x_j, edge_attr):   
        m1 = torch.cat([x_i, x_j, edge_attr], dim=1)
        self.E = self.R1(m1)
        return self.E
    
    def update(self, aggr_out, x):
        c = torch.cat([x, aggr_out], dim=1)
        return self.O(c).squeeze(dim=-1).squeeze(dim=-1)
    
class HGInteractionNetwork(MessagePassing):
    def __init__(self, node_features: int, edge_features: int, hidden_size: int, **kwargs):
        super(HGInteractionNetwork, self).__init__(aggr="max")
        self.R1 = RelationalModel(2*node_features + edge_features, hidden_size, hidden_size)
        self.O = ObjectModel(node_features + hidden_size, hidden_size, hidden_size)
        self.R2 = RelationalModel(3*hidden_size, 1, hidden_size)
        self.E = torch.Tensor()

    def forward(self, G: MaskedHitGraph) -> torch.Tensor:

        # propagate_type: (x: Tensor, edge_attr: Tensor)
        x_tilde = self.propagate(G.masked_edge_index, x=G.x, edge_attr=G.masked_edge_attr, size=None)

        m2 = torch.cat([x_tilde[G.masked_edge_index[1]],
                        x_tilde[G.masked_edge_index[0]],
                        self.E], dim=1)
        return torch.sigmoid(self.R2(m2)).squeeze()

    def message(self, x_i, x_j, edge_attr):
        m1 = torch.cat([x_i, x_j, edge_attr], dim=1)
        self.E = self.R1(m1)
        return self.E

    def update(self, aggr_out, x):
        c = torch.cat([x, aggr_out], dim=1)
        return self.O(c) 