# dl/pinn/networks.py


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, ChebConv, NNConv, GraphNorm,  GATConv, global_mean_pool


class MLP(nn.Module):
    def __init__(self, layer_size):
        super().__init__()
        self.mlp = self._build_mlp(layers=layer_size, activation=nn.GELU())
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
    def __init__(self, input_dim=3, hidden_dim=3, output_dim=9):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.act   = nn.GELU()
        self.proj_in = (nn.Linear(input_dim, hidden_dim)
                        if input_dim != hidden_dim else nn.Identity())
        self.head  = nn.Linear(hidden_dim, output_dim)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)
        elif isinstance(m, SAGEConv):
            # SAGEConv has 'lin_l' and 'lin_r' inside
            nn.init.xavier_uniform_(m.lin_l.weight)
            if m.lin_l.bias is not None:
                nn.init.constant_(m.lin_l.bias, 0.01)
            nn.init.xavier_uniform_(m.lin_r.weight)
            if m.lin_r.bias is not None:
                nn.init.constant_(m.lin_r.bias, 0.01)
    
    def initiate_weights(self) -> None:
        self.apply(self._init_weights)
        
    def forward(self, x, edge_index):
        # input -> hidden skip (with projection if needed)
        s0 = self.proj_in(x)
        
        # pre-norm residual block 1
        h = self.conv1(self.norm(x), edge_index)
        h = self.act(h)
        
        # pre-norm residual block 2
        h = self.conv2(self.norm1(h), edge_index)
        h = self.act(h)
        
        # add input skip
        h = h + s0
        return self.head(h)

class PhysicsInformedGraphSAGE(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=16, output_dim=9, num_layers=2, 
                dropout=0.1, use_attention=True, physics_constraint=True):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.physics_constraint = physics_constraint
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Initial projection
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if use_attention:
                conv = GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout)
            else:
                conv = SAGEConv(hidden_dim, hidden_dim)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Activation and dropout
        self.activation = nn.GELU()  # Often works better than Tanh
        self.dropout = nn.Dropout(dropout)
        
        # Multi-scale feature aggregation
        self.global_pool = global_mean_pool
        self.local_to_global = nn.Linear(hidden_dim, hidden_dim // 4)
        self.global_to_local = nn.Linear(hidden_dim // 4, hidden_dim)
        
        # Output head with physics-informed structure
        if physics_constraint:
            # Separate heads for different physical quantities
            self.stress_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 6)  # 6 stress components (3D)
            )
            self.displacement_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 3)   # 3 displacement components
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim)
            )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Use He initialization for GELU
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)
        elif isinstance(m, (SAGEConv, GATConv)):
            # Initialize graph conv layers
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif 'bias' in name:
                    nn.init.constant_(param, 0.01)
    
    def forward(self, x, edge_index, batch=None):
        # Input normalization and projection
        x = self.input_norm(x)
        x = self.proj_in(x)
        x = self.activation(x)
        
        # Store input for skip connection
        x_skip = x
        
        # Graph convolution layers with residual connections
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_res = x
            
            # Graph convolution
            x = conv(x, edge_index)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)
            
            # Residual connection every 2 layers
            if i % 2 == 1 and i > 0:
                x = x + x_res
        
        # Multi-scale feature aggregation
        if batch is not None:
            # Global context
            global_feat = self.global_pool(x, batch)
            global_feat = self.local_to_global(global_feat)
            
            # Broadcast global features back to nodes
            global_broadcast = global_feat[batch]
            global_broadcast = self.global_to_local(global_broadcast)
            
            # Combine local and global features
            x = x + global_broadcast
        
        # Final skip connection from input
        x = x + x_skip
        
        # Output prediction
        if self.physics_constraint:
            stress = self.stress_head(x)
            displacement = self.displacement_head(x)
            return torch.cat([displacement, stress], dim=-1)
        else:
            return self.head(x)

    def compute_physics_loss(self, pred, x, edge_index, material_props):
        """
        Compute physics-informed loss for linear elasticity
        pred: predictions [stress (6) + displacement (3)]
        material_props: dictionary with 'E' (Young's modulus), 'nu' (Poisson's ratio)
        """
        if not self.physics_constraint:
            return 0.0
        
        stress = pred[:, :6]  # σxx, σyy, σzz, τxy, τyz, τxz
        displacement = pred[:, 6:]  # ux, uy, uz
        
        # Material constants
        E = material_props.get('E', 200e9)  # Pa
        nu = material_props.get('nu', 0.3)
        
        # Hooke's law matrix (simplified for demonstration)
        lambda_lame = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        mu_lame = E / (2 * (1 + nu))
        
        # Physics loss terms would go here
        # This is a placeholder - you'd need to implement proper strain calculation
        # from displacement gradients and stress-strain relationships
        physics_loss = 0.0
        
        return physics_loss
    
class ChebNetLite(nn.Module):
    """
    Very small: two ChebConv layers with small hidden; control locality via K (Chebyshev order).
    """
    def __init__(self, in_dim=3, out_dim=9, hidden=10, K=3):
        super().__init__()
        self.conv1 = ChebConv(in_dim, hidden, K=K, normalization='sym')
        self.conv2 = ChebConv(hidden, out_dim, K=K, normalization='sym')
        self.act = nn.GELU()

    def forward(self, x, edge_index):
        x = self.act(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)  # linear head
        return x