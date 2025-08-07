
import torch
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from src.utils.detector import position_to_pixel
from src.utils.pandas import edep_to_cluster_size

class InvariantNorm(BaseTransform):
    def __init__(self, pixel: bool = True, cluster: bool = True):
        super().__init__()
        self.edep_index = 0
        self.pos_x_index = 1
        self.pos_y_index = 2
        assert pixel and cluster, \
            "Currently only supported for cluster and pixel definitions"
        
    def __normalize_edep(self, x: torch.Tensor):
        x[:,self.edep_index] /= edep_to_cluster_size(0.06)
        return x
    
    def __normalize_pos(self, x: torch.Tensor, spot_x: int, spot_y: int):
        spot_x_px = 0 if spot_x is None else position_to_pixel(spot_x, x_dim=True)
        spot_y_px = 0 if spot_x is None else position_to_pixel(spot_y, x_dim=False)
        
        x[:,self.pos_x_index] = (x[:,self.pos_x_index] - spot_x_px)/1000
        x[:,self.pos_y_index] = (x[:,self.pos_y_index] - spot_y_px)/1000
        return x
    
    def __call__(self, data: Data):
        assert hasattr(data, "spot_x"), \
            "Invariant norm is only available for TGraph definitions"
        assert hasattr(data, "spot_y"), \
            "Invariant norm is only available for TGraph definitions"
            
        data.x = self.__normalize_edep(data.x)
        data.x = self.__normalize_pos(data.x, data.spot_x, data.spot_y)
        return data
    