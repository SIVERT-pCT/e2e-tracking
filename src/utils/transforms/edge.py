
from math import pi as PI

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

class SphericalInverse(BaseTransform):
    """Uses pytorch geometric implementation (https://pytorch-geometric.readthedocs.io/
    en/latest/_modules/torch_geometric/transforms/spherical.html) for reference.
    """
    def __init__(self, norm=True, max_value=None, cat=True):
        self.norm = norm
        self.max = max_value
        self.cat = cat

    def __call__(self, data: Data):
        (from_index, to_index), pos, pseudo = data.edge_index, data.pos, data.edge_attr
        assert pos.dim() == 2 and pos.size(1) == 3

        cart =  pos[from_index] - pos[to_index]

        rho = torch.norm(cart, p=2, dim=-1).view(-1, 1)
        theta = torch.atan2(cart[..., 1], cart[..., 0]).view(-1, 1)
        phi = torch.acos(cart[..., 2] / rho.view(-1)).view(-1, 1)

        if self.norm:
            rho = rho / (rho.max() if self.max is None else self.max)
            theta = theta / (2 * PI)
            phi = phi / PI

        spherical = torch.cat([rho, theta, phi], dim=-1)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, spherical.type_as(pos)], dim=-1)
        else:
            data.edge_attr = spherical

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')