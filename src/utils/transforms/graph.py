
from typing import Any, Union, Tuple


import torch
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.data import Data
from torch_scatter import scatter_add
from torch_geometric.utils import degree
from torch_geometric.transforms import BaseTransform, ToUndirected
from src.supervised.gnn.encoding import SimilarityPE
from src.utils.graph import MaskedHitGraph, HitGraph

@torch.jit.script
def create_line_graph_edge_index(row: torch.Tensor, col: torch.Tensor, N: int):
    count = scatter_add(torch.ones_like(row), row, dim=0, dim_size=N)
    cumsum = torch.cat([count.new_zeros(1), count.cumsum(0)], dim=0)
    
    start = cumsum[col]
    n = cumsum[col + 1] - start

    t = torch.arange(sum(n), device=n.device)
    c = t + start.repeat_interleave(n) - F.pad(torch.cumsum(n, dim=0), [1, 0])[:-1].repeat_interleave(n)
    r = torch.arange(len(n), device=n.device).repeat_interleave(n)
    return r, c

class ToSegmentLineGraph(BaseTransform):
    def __init__(self, similarity_encoding: Any) -> None:
        super().__init__()
        self.similarity_encoding = similarity_encoding

    def __call__(self, data: Union[HitGraph, MaskedHitGraph]) -> Data:
        N = data.num_nodes
        (row, col) = data.masked_edge_index \
            if isinstance(data, MaskedHitGraph) \
            else data.edge_index

        if not data.is_directed():
            raise Exception()
        
        #New node features (dE_i, dE_j, pos_i, pos_j)
        x = torch.cat((data.edep[row].unsqueeze(dim=-1), 
                       data.edep[col].unsqueeze(dim=-1), 
                       data.pos[row], data.pos[col]), dim=-1)
        
        r, c = create_line_graph_edge_index(row, col, N)

        dist = data.pos[row] - data.pos[col]
        sim = torch.cosine_similarity(dist[r], dist[c])
        theta = torch.nan_to_num(torch.acos(sim))

        line_data = Data(x=x, edge_index=torch.stack([r, c], dim=0)).to(r.device)
        line_data.edge_attr = self.similarity_encoding(theta)
        line_data.dir_degree = degree(line_data.edge_index[0], line_data.num_nodes)
        line_data.theta = theta

        return line_data 
    
class ToMaskedTGraph(BaseException):
    def __init__(self, stage1_theta_dd: float, stage1_theta_dt: float) -> None:
        super().__init__()
        self.stage1_theta_dd = stage1_theta_dd
        self.stage1_theta_dt = stage1_theta_dt

    def __call__(self, data: HitGraph) -> Any:
        return MaskedHitGraph.from_graph(data,self.stage1_theta_dd, self.stage1_theta_dt)
    

class CustomGraphTransform:
    def __init__(self,
                 stage1_theta_dd: float, 
                 stage1_theta_dt: float,
                 d_encoding: float = 64, scaling: float = 5000,
                 to_line_graph: bool = True) -> None:
        
        self.to_line_graph = to_line_graph
        self.to_undirected = ToUndirected()
        self.to_masked = ToMaskedTGraph(stage1_theta_dd, stage1_theta_dt)
        self.to_linegraph = ToSegmentLineGraph(
            SimilarityPE(d_encoding, scaling
        ))

    def __call__(self, G) -> Tuple[MaskedHitGraph, Data]:
        HG = self.to_masked(G)
        
        if not self.to_line_graph:
            return HG, None
        
        LG = self.to_linegraph(HG)
        LG = self.to_undirected(LG)
        return HG, LG