# dl/graph/pde.py


import torch
import torch.nn as nn

from typing import Tuple


class PDE_Graph:
    """Physics‑informed PDE wrapper for nondimensionalized linear elasticity.
    
    Couples a graph‑based neural network (`net`) with material parameters
    to compute strains, stresses, and PINN residuals on a 3D mesh represented
    as a graph.
    
    
            
    Attributes:
        net (nn.Module):  The GNN model for displacements & stresses.
        E (float):        Young’s modulus inside the PINN.
        nu (float):       Poisson’s ratio of the material.
    """
    def __init__(self, net: nn.Module, E: float, nu: float) -> None:
        """Initialize the PDE with a network and material properties.
        
        Args:
            net (nn.Module):
                A PyTorch GNN module that maps node features (x, y, z) and
                `edge_index` connectivity to a tensor of shape [N, 9],
                containing:
                    [u, v, w, σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_zx].
            E (float):
                Young’s modulus (may be nondimensionalized).
            nu (float):
                Poisson’s ratio of the elastic material.
        """
        self.net = net
        self.E   = E
        self.nu  = nu
    
    def calculate_strain_central_diff(self, x: torch.Tensor, edge_index: torch.Tensor, delta: float = 1e-5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the small‐strain components via central differences.
        
        Perturbs the input node coordinates along each axis by ±delta/2,
        evaluates the GNN displacements, and forms the symmetric strain tensor.
        
        Args:
            x (torch.Tensor): Node coordinate tensor of shape [N, 3].
            edge_index (torch.Tensor):
                Graph connectivity of shape [2, E], as used by the GNN.
            delta (float, optional):
                Finite‐difference step size (default: 1e-5).
                
        Returns:
            eps_xx (torch.Tensor): Normal strain in x, shape [N].
            eps_yy (torch.Tensor): Normal strain in y, shape [N].
            eps_zz (torch.Tensor): Normal strain in z, shape [N].
            eps_xy (torch.Tensor): Shear strain γ_xy = ½(∂u/∂y + ∂v/∂x), shape [N].
            eps_yz (torch.Tensor): Shear strain γ_yz = ½(∂v/∂z + ∂w/∂y), shape [N].
            eps_xz (torch.Tensor): Shear strain γ_xz = ½(∂u/∂z + ∂w/∂x), shape [N].
        """
        N = x.shape[0]
        device = x.device
        eps_xx, eps_yy, eps_zz = torch.zeros(N, device=device), torch.zeros(N, device=device), torch.zeros(N, device=device)
        eps_xy, eps_yz, eps_xz = torch.zeros(N, device=device), torch.zeros(N, device=device), torch.zeros(N, device=device)
        # Perturb coordinates in each spatial direction
        for i in range(3):  # loop over x, y, z
            perturb = torch.zeros_like(x, device=device)
            perturb[:, i] = delta / 2
            
            u_plus = self.net(x + perturb, edge_index)[:, :3]
            u_minus = self.net(x - perturb, edge_index)[:, :3]
            
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
                
        return eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_xz
    
    def calculate_stress_tensor(self, x: torch.Tensor, edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Cauchy stress components from the small‐strain tensor.
        
        Applies the isotropic linear‐elastic constitutive law
        σ = C : ε using Young’s modulus `E` and Poisson’s ratio `nu`.
        
        Args:
            x (torch.Tensor): Node coordinates, shape [N, 3].
            edge_index (torch.Tensor):
                Graph connectivity, shape [2, E].
                
        Returns:
            sigma_xx (torch.Tensor): Normal stress in x, shape [N].
            sigma_yy (torch.Tensor): Normal stress in y, shape [N].
            sigma_zz (torch.Tensor): Normal stress in z, shape [N].
            sigma_xy (torch.Tensor): Shear stress τ_xy, shape [N].
            sigma_yz (torch.Tensor): Shear stress τ_yz, shape [N].
            sigma_zx (torch.Tensor): Shear stress τ_zx, shape [N].
        """
        # first compute small‐strain components via central diff
        eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_zx = self.calculate_strain_central_diff(x=x, edge_index=edge_index)
        # isotropic constitutive factor: E / [(1+ν)(1−2ν)]
        factor = self.E / ((1 + self.nu) * (1 - 2 * self.nu))
        # normal stress components
        sigma_xx = factor * ((1 - self.nu) * eps_xx + self.nu * eps_yy + self.nu * eps_zz)
        sigma_yy = factor * ((1 - self.nu) * eps_yy + self.nu * eps_xx + self.nu * eps_zz)
        sigma_zz = factor * ((1 - self.nu) * eps_zz + self.nu * eps_xx + self.nu * eps_yy)
        sigma_xy = factor * (1 - 2 * self.nu) * eps_xy
        sigma_yz = factor * (1 - 2 * self.nu) * eps_yz
        sigma_zx = factor * (1 - 2 * self.nu) * eps_zx
        return sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_zx
    
    def calculate_stress_coupling(self, x: torch.Tensor, edge_index: torch.Tensor, outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Form physics‐coupling residuals between model stresses and network outputs.
        
        Computes σ_ij(x) via the constitutive law, then subtracts the
        corresponding stress predictions that the GNN returned in `outputs`.
        
        Args:
            x (torch.Tensor): Node coordinates, shape [N, 3].
            edge_index (torch.Tensor):
                Graph connectivity, shape [2, E].
            outputs (torch.Tensor):
                Network predictions of shape [N, 9], ordered
                [u, v, w, σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_zx].
                
        Returns:
            term_xx (torch.Tensor): σ_xx_model – σ_xx_net, shape [N].
            term_yy (torch.Tensor): σ_yy_model – σ_yy_net, shape [N].
            term_zz (torch.Tensor): σ_zz_model – σ_zz_net, shape [N].
            term_xy (torch.Tensor): σ_xy_model – σ_xy_net, shape [N].
            term_yz (torch.Tensor): σ_yz_model – σ_yz_net, shape [N].
            term_zx (torch.Tensor): σ_zx_model – σ_zx_net, shape [N].
        """
        sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_zx = self.calculate_stress_tensor(x=x, edge_index=edge_index)
        term_xx = sigma_xx - outputs[:, 3]
        term_yy = sigma_yy - outputs[:, 4]
        term_zz = sigma_zz - outputs[:, 5]
        term_xy = sigma_xy - outputs[:, 6]
        term_yz = sigma_yz - outputs[:, 7]
        term_zx = sigma_zx - outputs[:, 8]
        return term_xx, term_yy, term_zz, term_xy, term_yz, term_zx
    
    def pde_mixed_central_diff(self, x: torch.Tensor, edge_index: torch.Tensor, outputs: torch.Tensor, delta: float = 1e-5):
        """Compute momentum balance and stress‐coupling residuals via central diff.
        
        For a graph with N nodes, perturbs each coordinate direction by ±δ/2,
        estimates ∂σ/∂x, ∂σ/∂y, ∂σ/∂z from the network’s stress outputs,
        assembles the three momentum equations, and then forms the PDE residual
        between model stress and network‐predicted stress.
        
        Args:
            x (torch.Tensor): Node coordinate tensor of shape [N, 3].
            edge_index (torch.Tensor): Graph connectivity [2, E].
            outputs (torch.Tensor):
                GNN predictions of shape [N, 9] = [u, v, w,
                σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_zx].
            delta (float, optional): Finite‐difference step (default: 1e-5).
            
        Returns:
            momentum_x (torch.Tensor):
                ∂σ_xx/∂x + ∂σ_xy/∂y + ∂σ_xz/∂z, shape [N].
            momentum_y (torch.Tensor):
                ∂σ_xy/∂x + ∂σ_yy/∂y + ∂σ_yz/∂z, shape [N].
            momentum_z (torch.Tensor):
                ∂σ_xz/∂x + ∂σ_yz/∂y + ∂σ_zz/∂z, shape [N].
            term_xx (torch.Tensor):
                σ_xx_model - σ_xx_net, shape [N].
            term_yy (torch.Tensor):
                σ_yy_model - σ_yy_net, shape [N].
            term_zz (torch.Tensor):
                σ_zz_model - σ_zz_net, shape [N].
            term_xy (torch.Tensor):
                σ_xy_model - σ_xy_net, shape [N].
            term_yz (torch.Tensor):
                σ_yz_model - σ_yz_net, shape [N].
            term_zx (torch.Tensor):
                σ_zx_model - σ_zx_net, shape [N].
        """
        N = x.shape[0]
        device = x.device
        momentum_x, momentum_y, momentum_z = torch.zeros(N, device=device), torch.zeros(N, device=device), torch.zeros(N, device=device)
        # Compute derivatives for stress components
        for i in range(3):  # x, y, z
            perturb = torch.zeros_like(x, device=device)
            perturb[:, i] = delta / 2
            
            sigma_plus = self.net(x + perturb, edge_index)[:, 3:9]
            sigma_minus = self.net(x - perturb, edge_index)[:, 3:9]
            
            dsigma_dxi = (sigma_plus - sigma_minus) / delta
            
            if i == 0:
                sigma_xx_x = dsigma_dxi[:, 0]
                sigma_xy_x = dsigma_dxi[:, 3]
                sigma_zx_x = dsigma_dxi[:, 5]
            elif i == 1:
                sigma_xy_y = dsigma_dxi[:, 3]
                sigma_yy_y = dsigma_dxi[:, 1]
                sigma_yz_y = dsigma_dxi[:, 4]
            elif i == 2:
                sigma_zx_z = dsigma_dxi[:, 5]
                sigma_yz_z = dsigma_dxi[:, 4]
                sigma_zz_z = dsigma_dxi[:, 2]
                
        # Assemble momentum equations
        momentum_x = sigma_xx_x + sigma_xy_y + sigma_zx_z
        momentum_y = sigma_xy_x + sigma_yy_y + sigma_yz_z
        momentum_z = sigma_zx_x + sigma_yz_y + sigma_zz_z
        
        # Material law
        term_xx, term_yy, term_zz, term_xy, term_yz, term_zx = self.calculate_stress_coupling(x=x, edge_index=edge_index, outputs=outputs)
        return momentum_x, momentum_y, momentum_z, term_xx, term_yy, term_zz, term_xy, term_yz, term_zx
    