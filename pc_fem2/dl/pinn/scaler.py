# dl/pinn/scaler.py


import torch
import torch.nn as nn


class Scaler():
    """Helper for mapping between physical and normalized coordinates and retrieving original model outputs.
    
    This wraps a pretrained neural network `net` that operates on normalized
    inputs in [0,1]^3, and provides methods to:
        - normalize physical points into [0,1]^3
        - reverse that mapping
        - recover the physical displacement or stress outputs from the normalized model.
        
    Attributes:
        net (nn.Module): Network mapping normalized coordinates → [u,v,w, σ…].
        mins (torch.Tensor): 3‐vector of minimum physical coordinates.
        maxs (torch.Tensor): 3‐vector of maximum physical coordinates.
    """
    def __init__(self, net: nn.Module, mins: torch.Tensor, maxs: torch.Tensor) -> None:
        """Initialize the scaler with a network and physical bounds.
        
        Args:
            net: A PyTorch module that takes inputs in normalized [0,1]^3.
            mins: Tensor of shape (3,) giving the physical minimum in each axis.
            maxs: Tensor of shape (3,) giving the physical maximum in each axis.
        """
        self.net = net
        self.mins = mins
        self.maxs = maxs
    
    def normalize_coordinates(self, x: torch.Tensor) -> torch.Tensor:
        """Linearly map physical points into the unit cube [0,1]^3.
        
        Args:
            x: Tensor of shape (N,3) in physical coordinates.
            
        Returns:
            Tensor of shape (N,3) where
            x_norm[i,j] = (x[i,j] - mins[j]) / (maxs[j] - mins[j]).
        """
        scale = self.maxs - self.mins           # (3,)
        x_norm = (x - self.mins[None, :]) / scale[None, :]
        return x_norm
    
    def scale_back_from_standard_coordinates(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Map normalized [0,1]^3 points back to physical coordinates.
        
        Args:
            x_norm: Tensor of shape (N,3) in normalized coordinates.
            
        Returns:
            Tensor of shape (N,3) in physical coordinates:
            x_phys = x_norm * (maxs - mins) + mins.
        """
        scale = self.maxs - self.mins           # (3,)
        x = x_norm * scale[None, :] + self.mins[None, :]
        return x
    
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
        self.net.eval()
        with torch.no_grad():
            disp1 = self.net(torch.tensor([1.0, 1.0, 1.0]).to(self.mins.device))[:3]
        return (x - self.mins) * disp1
    
    def calculate_original_stress(self, x: torch.Tensor) -> torch.Tensor:
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
            stress = self.net(x_norm)[:, 3:9]
        return stress
    
    def calculate_elastic_strain(self, x_norm: torch.Tensor) -> torch.Tensor:
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
        def uvw_fn(x):
            # x has shape (N,3); we need to re-compute u,v,w from the graph
            # The easiest is to re-slice outputs, but torch.functional.jacobian
            # requires us to recompute outputs via the network:
            return self.net(x)[:3]
        # Compute the full Jacobian J_ijk = ∂(u,v,w)_i / ∂x_j
        J  = torch.func.vmap(torch.func.jacrev(uvw_fn))(x_norm)  # shape: (N, 3, 3)
        # Shear strains
        eps_xx = J[:, 0, 0]
        eps_yy = J[:, 1, 1]
        eps_zz = J[:, 2, 2]
        eps_xy = 0.5 * (J[:, 1, 0] + J[:, 0, 1])
        eps_yz = 0.5 * (J[:, 2, 1] + J[:, 1, 2])
        eps_xz = 0.5 * (J[:, 2, 0] + J[:, 0, 2])
        return torch.stack([eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_xz], dim=1)
    
    def calculate_original_strain(self, x: torch.Tensor) -> torch.Tensor:
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
        strain = self.calculate_elastic_strain(x_norm=x_norm)
        return strain


