# dl/pinn/networks.py


import torch
import torch.nn as nn


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