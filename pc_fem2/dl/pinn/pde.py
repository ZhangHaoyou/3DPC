# dl/pinn/pde.py


import torch
import torch.nn as nn

from typing import Tuple


class PDE:
    """Physics‐informed PDE wrapper for nondimensionalized linear elasticity.
    
    This class couples a neural network approximation (`net`) with material
    parameters provided directly as Young’s modulus `E` and Poisson’s ratio `nu`,
    and provides methods to compute strains, stresses, and PDE residuals for PINN training.
    
    Attributes:
        net (nn.Module):
            Neural network mapping spatial inputs (x, y, z) to a vector of
            displacements and stress components [u, v, w, σₓₓ, σᵧᵧ, σ𝓏𝓏, σₓᵧ, σᵧ𝓏, σ𝓏ₓ].
        E (float):
            Nondimensionalized Young’s modulus used inside the PINN.
        nu (float):
            Poisson’s ratio of the elastic material.
    """
    def __init__(self, net: nn.Module, E: float, nu: float) -> None:
        """Initialize the PDE with a network and material properties.
        
        Args:
            net: A PyTorch module that takes spatial coordinates (x, y, z)
                    and returns displacements and stress components.
            E:   Young’s modulus to use in the PINN (may be nondimensionalized).
            nu:  Poisson’s ratio of the material.
        """
        self.net = net
        self.E   = E
        self.nu  = nu
    
    def calculate_elastic_strain(self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute small‐strain components from network‐predicted displacements.
        
        This method uses automatic differentiation to obtain the displacement
        gradient ∂(u,v,w)/∂(x,y,z) at each of the N input points, then assembles
        the symmetric small‐strain tensor components:
        
            ε_xx = ∂u/∂x
            ε_yy = ∂v/∂y
            ε_zz = ∂w/∂z
            ε_xy = ½(∂u/∂y + ∂v/∂x)
            ε_yz = ½(∂v/∂z + ∂w/∂y)
            ε_xz = ½(∂u/∂z + ∂w/∂x)
        
        Args:
            inputs: FloatTensor of shape (N, 3), the spatial coordinates
                at which to evaluate the strain.
                
        Returns:
            A 6‐tuple of FloatTensors, each of shape (N,), in the order:
            (eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_xz).
        """
        def uvw_fn(x):
            # x has shape (N,3); we need to re-compute u,v,w from the graph
            # The easiest is to re-slice outputs, but torch.functional.jacobian
            # requires us to recompute outputs via the network:
            return self.net(x)[:3]
        # Compute the full Jacobian J_ijk = ∂(u,v,w)_i / ∂x_j
        J  = torch.func.vmap(torch.func.jacrev(uvw_fn))(inputs)  # shape: (N, 3, 3)
        # Shear strains
        eps_xx = J[:, 0, 0]
        eps_yy = J[:, 1, 1]
        eps_zz = J[:, 2, 2]
        eps_xy = 0.5 * (J[:, 1, 0] + J[:, 0, 1])
        eps_yz = 0.5 * (J[:, 2, 1] + J[:, 1, 2])
        eps_xz = 0.5 * (J[:, 2, 0] + J[:, 0, 2])
        return eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_xz
    
    def calculate_stress_tensor(self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Cauchy stress components from small‐strain via Hooke’s law.
        
        Given the displacement field predicted by `self.net`, this method first
        computes the small‐strain components ε using automatic differentiation,
        then applies the linear elastic constitutive relation for an isotropic
        material with Young’s modulus E and Poisson’s ratio ν:
            σₓₓ = λ tr(ε) + 2μ εₓₓ,   etc.
        but here written in the common “factor” form.
        
        Args:
            inputs: FloatTensor of shape (N,3), the spatial coordinates
                at which to evaluate the stress.
        
        Returns:
            A tuple of six FloatTensors, each of shape (N,):
            - sigma_xx: normal stress σₓₓ
            - sigma_yy: normal stress σᵧᵧ
            - sigma_zz: normal stress σ𝓏𝓏
            - sigma_xy: shear stress σₓᵧ = σᵧₓ
            - sigma_yz: shear stress σᵧ𝓏 = σ𝓏ᵧ
            - sigma_zx: shear stress σ𝓏ₓ = σₓ𝓏
        """
        eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_zx = self.calculate_elastic_strain(inputs=inputs)
        # calculate stress terms (constitutive law)
        factor = self.E / ((1 + self.nu) * (1 - 2 * self.nu))
        sigma_xx = factor * ((1 - self.nu) * eps_xx + self.nu * eps_yy + self.nu * eps_zz)
        sigma_yy = factor * ((1 - self.nu) * eps_yy + self.nu * eps_xx + self.nu * eps_zz)
        sigma_zz = factor * ((1 - self.nu) * eps_zz + self.nu * eps_xx + self.nu * eps_yy)
        sigma_xy = factor * (1 - 2 * self.nu) * eps_xy
        sigma_yz = factor * (1 - 2 * self.nu) * eps_yz
        sigma_zx = factor * (1 - 2 * self.nu) * eps_zx
        return sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_zx
    
    def calculate_stress_coupling(self, inputs: torch.Tensor, outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the residual between constitutive and network‐predicted stress components.
        
        For a mixed‐formulation PINN, we enforce that the stress predicted by the
        network matches the stress obtained from Hooke’s law applied to the
        displacement field.  This method returns the six component‐wise differences:
        
            term_xx = σₓₓ(ε) − σₓₓ^net,
            term_yy = σᵧᵧ(ε) − σᵧᵧ^net,
            …,
            term_zx = σ𝓏ₓ(ε) − σ𝓏ₓ^net.
        
        Args:
            inputs:  FloatTensor of shape (N,3), the spatial coordinates.
            outputs: FloatTensor of shape (N,9), network predictions in the order
                    [u, v, w, σₓₓ^net, σᵧᵧ^net, σ𝓏𝓏^net, σₓᵧ^net, σᵧ𝓏^net, σ𝓏ₓ^net].
                    
        Returns:
            A 6‐tuple of FloatTensors, each of shape (N,):
            - term_xx: Residual in σₓₓ.
            - term_yy: Residual in σᵧᵧ.
            - term_zz: Residual in σ𝓏𝓏.
            - term_xy: Residual in σₓᵧ.
            - term_yz: Residual in σᵧ𝓏.
            - term_zx: Residual in σ𝓏ₓ.
        """
        sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_zx = self.calculate_stress_tensor(inputs=inputs)
        term_xx = sigma_xx - outputs[:, 3]
        term_yy = sigma_yy - outputs[:, 4]
        term_zz = sigma_zz - outputs[:, 5]
        term_xy = sigma_xy - outputs[:, 6]
        term_yz = sigma_yz - outputs[:, 7]
        term_zx = sigma_zx - outputs[:, 8]
        return term_xx, term_yy, term_zz, term_xy, term_yz, term_zx
    
    def pde_mixed(self, inputs: torch.Tensor, outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the mixed‐formulation PDE residuals for 3D linear elasticity.
        
        This assembles:
        1. The equilibrium (momentum‐balance) residuals ∇·σ = 0 in each spatial
            direction (x, y, z) by differentiating the predicted stresses.
        2. The constitutive coupling residuals enforcing σ = Hooke(ε) via
            `calculate_stress_coupling`.
            
        Args:
            inputs:  FloatTensor of shape (N,3), the spatial coordinates (x,y,z).
            outputs: FloatTensor of shape (N,9), the network’s predictions in order
                    [u, v, w, σₓₓ, σᵧᵧ, σ𝓏𝓏, σₓᵧ, σᵧ𝓏, σ𝓏ₓ].
            
        Returns:
            A 9‐tuple of FloatTensors, each of shape (N,):
            - momentum_x: ∂σₓₓ/∂x + ∂σₓᵧ/∂y + ∂σₓ𝓏/∂z
            - momentum_y: ∂σₓᵧ/∂x + ∂σᵧᵧ/∂y + ∂σᵧ𝓏/∂z
            - momentum_z: ∂σₓ𝓏/∂x + ∂σᵧ𝓏/∂y + ∂σ𝓏𝓏/∂z
            - term_xx … term_zx: the six constitutive residuals
                (σᵢⱼ(ε) – σᵢⱼ^net) from `calculate_stress_coupling`.
        """
        # Helper to extract only the stress components from the network
        def stress_fn(x):
            # returns (6,) vector [σₓₓ, σᵧᵧ, σ𝓏𝓏, σₓᵧ, σᵧ𝓏, σ𝓏ₓ] for a single point
            return self.net(x)[3:9]
        # Compute the full Jacobian J[k,i,j] = ∂stresses[k,i] / ∂inputs[k,j]
        J  = torch.func.vmap(torch.func.jacrev(stress_fn))(inputs)
        # Now pull out exactly the partials needed
        sigma_xx_x = J[:, 0, 0]
        sigma_xy_y = J[:, 3, 1]
        sigma_zx_z = J[:, 5, 2]
        
        sigma_xy_x = J[:, 3, 0]
        sigma_yy_y = J[:, 1, 1]
        sigma_yz_z = J[:, 4, 2]
        
        sigma_zx_x = J[:, 5, 0]
        sigma_yz_y = J[:, 4, 1]
        sigma_zz_z = J[:, 2, 2]
        # Momentum residuals (∇·σ)
        momentum_x = sigma_xx_x + sigma_xy_y + sigma_zx_z
        momentum_y = sigma_xy_x + sigma_yy_y + sigma_yz_z
        momentum_z = sigma_zx_x + sigma_yz_y + sigma_zz_z
        # Stress coupling residuals (σ_pred - σ_from_disp)
        term_xx, term_yy, term_zz, term_xy, term_yz, term_zx = self.calculate_stress_coupling(inputs=inputs, outputs=outputs)
        return momentum_x, momentum_y, momentum_z, term_xx, term_yy, term_zz, term_xy, term_yz, term_zx
    
    