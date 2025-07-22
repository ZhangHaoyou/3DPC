# dl/graph/scaler.py


import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph


from dl.pinn.scaler import Scaler


class Scaler_Graph(Scaler):
    
    def __init__(self, net: nn.Module, mins: torch.Tensor, maxs: torch.Tensor) -> None:
        """Initialize the scaler with a network and physical bounds.
        
        Args:
            net (nn.Module):  The GNN model for displacements & stresses.
            mins: Tensor of shape (3,) giving the physical minimum in each axis.
            maxs: Tensor of shape (3,) giving the physical maximum in each axis.
        """
        self.net = net
        self.mins = mins
        self.maxs = maxs
    
    def calculate_original_displacement(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the physical displacement field at points x.
        
        The network’s output at normalized corner [1,1,1] is assumed to be
        the “max‐displacement” disp1 = net([1,1,1])[:3].  We then assume
        a linear variation so that
            u_phys(x) = (x - mins) ⊙ disp1.
            
        Args:
            x: Tensor of shape (N,3) in physical coordinates.
            
        Returns:
            Tensor of shape (N,3) giving the displacement [u,v,w] at each x.
        """
        N = 16
        xy = torch.linspace(0.0, 1.0, N, device=x.device, dtype=x.dtype)
        z = torch.ones(1, device=x.device, dtype=x.dtype)
        top_points = torch.cartesian_prod(xy, xy, z)
        edge_index = knn_graph(x=top_points, k=8, loop=False)
        data_top = Data(x=top_points, edge_index=edge_index).to(x.device)
        
        self.net.eval()
        with torch.no_grad():
            disp1 = self.net(x=data_top.x, edge_index=data_top.edge_index)[-1, :3]
        return (x - self.mins) * disp1
    
    def calculate_original_stress(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute the physical stress components at points x.
        
        1) Normalize x into [0,1]^3  
        2) Run the network to predict stresses on normalized inputs  
        3) Extract the six stress components [σₓₓ, σᵧᵧ, σ𝓏𝓏, σₓᵧ, σᵧ𝓏, σ𝓏ₓ]
        
        Args:
            x: Tensor of shape (N,3) in physical coordinates.
            
        Returns:
            Tensor of shape (N,6) with network‐predicted stress components
            in the order [σₓₓ, σᵧᵧ, σ𝓏𝓏, σₓᵧ, σᵧ𝓏, σ𝓏ₓ].
        """
        x_norm = self.normalize_coordinates(x=x)
        self.net.eval()
        with torch.no_grad():
            stress = self.net(x=x_norm, edge_index=edge_index)[:, 3:9]
        return stress
    
    def calculate_elastic_strain(self, x_norm: torch.Tensor, edge_index: torch.Tensor,
                delta: float = 1e-5) -> torch.Tensor:
        """Compute small-strain components from network-predicted displacements.
        
        Uses automatic differentiation to recover
        ε_ij = ½ (∂u_i/∂x_j + ∂u_j/∂x_i) at each of the N input points.
        
        Args:
            x_norm (torch.Tensor): FloatTensor of shape (N, 3), the
                normalized spatial coordinates at which to evaluate the strain.
                
        Returns:
            torch.Tensor: FloatTensor of shape (N, 6), containing
            the small-strain components in Voigt order
            `[eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_xz]`.
        """
        N = x_norm.shape[0]
        device = x_norm.device
        eps_xx, eps_yy, eps_zz = torch.zeros(N, device=device), torch.zeros(N, device=device), torch.zeros(N, device=device)
        eps_xy, eps_yz, eps_xz = torch.zeros(N, device=device), torch.zeros(N, device=device), torch.zeros(N, device=device)
        # Perturb coordinates in each spatial direction
        for i in range(3):  # loop over x, y, z
            perturb = torch.zeros_like(x_norm, device=device)
            perturb[:, i] = delta / 2
            
            u_plus = self.net(x_norm + perturb, edge_index)[:, :3]
            u_minus = self.net(x_norm - perturb, edge_index)[:, :3]
            
            du_dxi = (u_plus - u_minus) / delta
            
            if i == 0:  # derivative wrt x
                eps_xx = du_dxi[:, 0]
                eps_xy += 0.5 * du_dxi[:, 1]
                eps_xz += 0.5 * du_dxi[:, 2]
            elif i == 1:  # derivative wrt y
                eps_yy = du_dxi[:, 1]
                eps_xy += 0.5 * du_dxi[:, 0]
                eps_yz += 0.5 * du_dxi[:, 2]
            elif i == 2:  # derivative wrt z
                eps_zz = du_dxi[:, 2]
                eps_yz += 0.5 * du_dxi[:, 1]
                eps_xz += 0.5 * du_dxi[:, 0]
                
        return torch.stack([eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_xz], dim=1)
    
    def calculate_original_strain(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Convert raw physical coords → normalized coords → small-strain.
        
        1. Normalize physical coordinates `x` into [0,1]^3 via `self.normalize_coordinates`.  
        2. Compute the small-strain components at each normalized point.
        
        Args:
            x (torch.Tensor): FloatTensor of shape (N, 3), raw physical coordinates.
            
        Returns:
            torch.Tensor: FloatTensor of shape (N, 6), the small-strain components
            `[eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_xz]` in physical units.
        """
        x_norm = self.normalize_coordinates(x=x)
        strain = self.calculate_elastic_strain(x_norm=x_norm, edge_index=edge_index)
        return strain
    