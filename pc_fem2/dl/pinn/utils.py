# dl/pinn/utils.py


import torch
import torch.nn as nn

from torch.autograd.functional import jacobian

E = 30e3 # psi
nu = 0.3
lam = E * nu / (1 - nu**2)
mu = E / (2 * (1 + nu))

def calculate_strain(func: callable, x: torch.Tensor) -> torch.Tensor:
    def calculate_strain_at_single(func: callable, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True) # (d,)
        J = jacobian(func, x, create_graph=True) # (d, d)
        return 0.5 * (J + J.transpose(0, 1))
    
    if x.dim() == 1:
        return calculate_strain_at_single(func=func, x=x)
    return torch.stack([calculate_strain_at_single(func=func, x=x_i)
                for x_i in x], dim=0)

def calculate_stress(eps: torch.Tensor, lam: float = lam, mu: float = mu) -> torch.Tensor:
    trace = torch.einsum('...ii->...', eps)
    
    d = eps.shape[-1]
    I = torch.eye(d, device=eps.device, dtype=eps.dtype)

    sig = lam * trace[..., None, None] * I + 2 * mu * eps
    
    return sig

def calculate_divergence(func: callable, x: torch.Tensor) -> torch.Tensor:
    def calculate_divergence_at_singe(x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)

        def stress_fn(x: torch.Tensor) -> torch.Tensor:
            eps = calculate_strain(func=func, x=x)
            return calculate_stress(eps=eps)

        # J[i,j,k] = ∂σ_ij / ∂x_k at this single point → shape (d, d, d)
        J = jacobian(stress_fn, x, create_graph=True)
        
        # divergence_i = ∑_j ∂σ_ij/∂x_j = J[i, j, j]
        return torch.einsum('ijj->i', J) # → (d,)
    
    if x.dim == 1:
        return calculate_divergence_at_singe(x=x)
    
    return torch.stack([calculate_divergence_at_singe(x=x_i) for x_i in x], dim=0)



def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp