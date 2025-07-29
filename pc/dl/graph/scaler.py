# dl/graph/scaler.py


import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn import knn_graph


from dl.pinn.scaler import Scaler


class Scaler_Graph(Scaler):
    """Helper for mapping between physical and normalized coordinates on a graph and recovering original model outputs.
    
    Wraps a pretrained graph neural network that operates on normalized inputs
    in [0,1]^3 and provides methods to:
        - normalize physical node coordinates into [0,1]^3
        - reverse that mapping for model predictions
        - recover the physical displacement or stress outputs from the normalized model,
            using graph connectivity
            
    Attributes:
        net (nn.Module): GNN model mapping normalized node coords and edge_index to
            [u, v, w, sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_zx].
        mins (torch.Tensor): 3-element tensor of minimum physical coordinates.
        maxs (torch.Tensor): 3-element tensor of maximum physical coordinates.
    """
    def __init__(self, net: nn.Module, mins: torch.Tensor, maxs: torch.Tensor) -> None:
        """Initialize the scaler with a graph network and physical bounds.
        
        Args:
            net (nn.Module): Graph neural network for displacements and stresses.
            mins (torch.Tensor): Tensor of shape (3,) giving the physical
                minimum in each axis.
            maxs (torch.Tensor): Tensor of shape (3,) giving the physical
                maximum in each axis.
        """
        self.net = net
        self.mins = mins
        self.maxs = maxs
    
    def calculate_original_displacement(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the physical displacement field at points x.
        
        The network's output at normalized corner [1,1,1] is assumed to be
        the "max-displacement" disp1 = net([1,1,1])[:3].  We then assume
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
        """Compute the network‑predicted physical stress components at given points.
        
        This method performs the following steps:
        1. Normalizes the input coordinates from physical space into [0,1]^3.
        2. Runs the GNN in evaluation mode to predict displacements and stresses.
        3. Extracts and returns the six stress components.
        
        Args:
            x (torch.Tensor): Tensor of shape (N, 3) with node coordinates in physical units.
            edge_index (torch.Tensor): LongTensor of shape (2, E) defining graph connectivity.
            
        Returns:
            torch.Tensor: Tensor of shape (N, 6) containing the stress components
                [sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_zx]
                in physical units.
        """
        x_norm = self.normalize_coordinates(x=x)
        self.net.eval()
        with torch.no_grad():
            stress = self.net(x=x_norm, edge_index=edge_index)[:, 3:9]
        return stress
    
    def calculate_elastic_strain(self, x_norm: torch.Tensor, edge_index: torch.Tensor,
                delta: float = 1e-5) -> torch.Tensor:
        """Compute small-strain tensor components from model-predicted displacements.
        
        Uses central finite differences on the GNN’s displacement output to
        evaluate the small-strain definition
        ε_ij = ½ (∂u_i/∂x_j + ∂u_j/∂x_i) at each input point.
        
        Args:
            x_norm (torch.Tensor): Tensor of shape (N, 3) giving normalized
                node coordinates in [0,1]^3.
            edge_index (torch.Tensor): LongTensor of shape (2, E) defining
                graph connectivity.
            delta (float): Finite-difference step size for numerical derivatives.
            
        Returns:
            torch.Tensor: Tensor of shape (N, 6) containing the six small-strain
                components in Voigt order:
                [eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_xz].
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
            edge_index (torch.Tensor): LongTensor of shape (2, E) defining graph connectivity.
            
        Returns:
            torch.Tensor: FloatTensor of shape (N, 6), the small-strain components
            `[eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_xz]` in physical units.
        """
        x_norm = self.normalize_coordinates(x=x)
        strain = self.calculate_elastic_strain(x_norm=x_norm, edge_index=edge_index)
        return strain
    