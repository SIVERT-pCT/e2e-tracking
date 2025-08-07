
from torch import nn

class RelationalModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, m):
        return self.layers(m)

class ObjectModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(output_size),
            nn.Linear(hidden_size, output_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(output_size),
        )

    def forward(self, C):
        return self.layers(C)