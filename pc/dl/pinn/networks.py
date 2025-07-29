# dl/pinn/networks.py


import torch
import torch.nn as nn

from torch_geometric.nn import SAGEConv


class MLP(nn.Module):
    def __init__(self, layer_size):
        super().__init__()
        self.mlp = self._build_mlp(layers=layer_size, activation=nn.Tanh())
        self.apply(self._init_weights)
    
    def _build_mlp(self, layers: list, activation) -> nn.Sequential:
        modules: list[nn.Module] = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(activation)
        return nn.Sequential(*modules)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)
    
    def initiate_weights(self) -> None:
        self.apply(self._init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class GraphSAGE(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=9):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.act   = nn.GELU()
        self.lin = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.act(x)
        
        x = self.conv2(x, edge_index)
        x = self.act(x)
        
        x = self.conv3(x, edge_index)
        x = self.act(x)
        
        x = self.lin(x)
        return x