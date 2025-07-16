# dl/pinn/contact_boundary_conditions.py

import torch

from typing import Tuple


from dl.pinn.geometry import Single_Contact_Geometry


class Contact_Boundary_Condition:
    """Encapsulates contact boundary data for traction/pressure application.
    
    This object holds the contact geometry (face or patch) and the
    uniform contact pressure.  Downstream methods can compute boundary
    tractions, tangential components, gap functions, etc., using this data.
    
    Attributes:
        geom: Geometry of the contact surface.
        pressure: Applied normal pressure (tensor scalar on geom.device).
    """
    def __init__(self, geom: Single_Contact_Geometry, pressure: float):
        """Initialize a contact boundary condition.
        
        Args:
            geom: A `Single_Contact_Geometry` describing the contact patch.
            pressure: Scalar contact pressure (positive in compression),
                        will be converted to a tensor on `geom.device`.
        """
        self.geom = geom
        self.pressure = torch.tensor(pressure, device=geom.device)
    
    def calculate_traction_from_stress(self, sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_zx, normals, t1, t2):
        """Compute traction and its normal/tangential components on a contact surface.
        
        Given the stress components at N boundary points and the local normal
        and two tangent directions, this builds the full stress tensor, computes
        the traction vector T = σ · n, and then projects T onto n, t1, and t2.
        
        Args:
            sigma_xx: Tensor of shape (N,) for σₓₓ.
            sigma_yy: Tensor of shape (N,) for σᵧᵧ.
            sigma_zz: Tensor of shape (N,) for σ𝓏𝓏.
            sigma_xy: Tensor of shape (N,) for σₓᵧ = σᵧₓ.
            sigma_yz: Tensor of shape (N,) for σᵧ𝓏 = σ𝓏ᵧ.
            sigma_zx: Tensor of shape (N,) for σ𝓏ₓ = σₓ𝓏.
            normals: Tensor of shape (N,3) with unit outward normals on each point.
            t1:      Tensor of shape (N,3) first tangent direction at each point.
            t2:      Tensor of shape (N,3) second tangent direction at each point.
            
        Returns:
            Tx:  (N,) x‐component of the traction vector.
            Ty:  (N,) y‐component of the traction vector.
            Tz:  (N,) z‐component of the traction vector.
            Tn:  (N,) normal component of the traction (T ⋅ n).
            Tt1: (N,) first tangential component (T ⋅ t1).
            Tt2: (N,) second tangential component (T ⋅ t2).
        """
        # 1) Build the symmetric stress tensor (N,3,3)
        #    [ [σ_xx, σ_xy, σ_zx],
        #      [σ_xy, σ_yy, σ_yz],
        #      [σ_zx, σ_yz, σ_zz] ]
        sigma = torch.stack([
            torch.stack([sigma_xx, sigma_xy, sigma_zx], dim=1),
            torch.stack([sigma_xy, sigma_yy, sigma_yz], dim=1),
            torch.stack([sigma_zx, sigma_yz, sigma_zz], dim=1),
        ], dim=1)  # now shape (N,3,3)
        # 2) Compute traction vector T = σ ⋅ n  -> shape (N,3,1), squeeze to (N,3)
        #    (we treat normals as (...,3,1) so matmul works)
        n_vec = normals.unsqueeze(-1)        # (N,3,1)
        T_vec = torch.matmul(sigma, n_vec).squeeze(-1)  # (N,3)
        # 3) Split into components
        Tx, Ty, Tz = T_vec.unbind(dim=1)      # each (N,)
        # 4) Compute Tn = T ⋅ n, and Tt1 = T ⋅ t1, Tt2 = T ⋅ t2
        #    simple elementwise dot:
        Tn  = (T_vec * normals).sum(dim=1)    # (N,)
        Tt1 = (T_vec * t1)     .sum(dim=1)
        Tt2 = (T_vec * t2)     .sum(dim=1)
        return Tx, Ty, Tz, Tn, Tt1, Tt2
    
    def calculate_traction_mixed_formulation(self, inputs: torch.Tensor, outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute boundary traction components from a mixed‐formulation PINN output.
        
        This method takes the network’s outputs (including stress components) at
        a set of boundary points, builds local normals and tangents, and then
        computes the full traction vector and its normal/tangential projections.
        
        Args:
            inputs: FloatTensor of shape (M,3), the physical coordinates of
                M boundary points where traction is to be evaluated.
            outputs: FloatTensor of shape (M,9), the network’s predictions in
                the order `[u, v, w, σₓₓ, σᵧᵧ, σ𝓏𝓏, σₓᵧ, σᵧ𝓏, σ𝓏ₓ]`.
                
        Returns:
            A 6‐tuple of FloatTensors, each of shape (M,):
            - Tx:  x‐component of the traction vector.
            - Ty:  y‐component of the traction vector.
            - Tz:  z‐component of the traction vector.
            - Tn:  normal component T ⋅ n.
            - Tt1: first tangential component T ⋅ t₁.
            - Tt2: second tangential component T ⋅ t₂.
        """
        stress = outputs[:, 3:9]  # (M,6) [xx,yy,zz,xy,yz,zx]
        # compute normals & tangents
        normals, t1, t2 = self.geom.calculate_boundary_normals_tangents(inputs)
        # split out the six stress columns
        sx, sy, sz, sxy, syz, szx = stress.unbind(dim=1)
        # compute traction *only* on boundary
        Tx, Ty, Tz, Tn, Tt1, Tt2 = self.calculate_traction_from_stress(
            sigma_xx=sx, sigma_yy=sy, sigma_zz=sz,
            sigma_xy=sxy, sigma_yz=syz, sigma_zx=szx,
            normals=normals, t1=t1, t2=t2
        )
        return Tx, Ty, Tz, Tn, Tt1, Tt2
    
    def calculate_tangential_traction_component1(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Compute the first tangential component of the traction on boundary points.
        
        This extracts Tₜ₁ from the full traction vector computed by the mixed‐
        formulation routine.
        
        Args:
            inputs:  FloatTensor of shape (M,3), the boundary point coordinates.
            outputs: FloatTensor of shape (M,9), network predictions in order
                    [u, v, w, σₓₓ, σᵧᵧ, σ𝓏𝓏, σₓᵧ, σᵧ𝓏, σ𝓏ₓ].
        
        Returns:
            FloatTensor of shape (M,), the traction component T ⋅ t₁ at each point.
        """
        _, _, _, _, Tt1, _ = self.calculate_traction_mixed_formulation(inputs=inputs, outputs=outputs)
        return Tt1
    
    def calculate_tangential_traction_component2(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Compute the second tangential component of the traction on boundary points.
        
        This extracts Tₜ₂ from the full traction vector computed by the mixed‐
        formulation routine.
        
        Args:
            inputs:  FloatTensor of shape (M,3), the boundary point coordinates.
            outputs: FloatTensor of shape (M,9), network predictions in order
                    [u, v, w, σₓₓ, σᵧᵧ, σ𝓏𝓏, σₓᵧ, σᵧ𝓏, σ𝓏ₓ].
                    
        Returns:
            FloatTensor of shape (M,), the traction component T ⋅ t₂ at each point.
        """
        _, _, _, _, _, Tt2 = self.calculate_traction_mixed_formulation(inputs=inputs, outputs=outputs)
        return Tt2
    
    def calculate_gap_in_normal_direction(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Compute the normal gap (penetration) at contact interface points.
        
        The normal gap gₙ is defined as the negative of the normal component
        of the displacement: gₙ = - (u · n).  Non‐zero only where the model
        predicts contact.
        
        Args:
            inputs:  FloatTensor of shape (M,3), the boundary point coordinates.
            outputs: FloatTensor of shape (M,9), network predictions [u, v, w, …].
        
        Returns:
            FloatTensor of shape (M,), the normal gap gₙ at each input point.
        """
        disp = outputs[:, :3]                # (M,3) [u,v,w]
        normals, _, _ = self.geom.calculate_boundary_normals_tangents(x=inputs) # normals: (M,3)
        # Compute gap = -(u ⋅ n) for each contact point
        #    batch‐dot via elementwise*sum
        gap_n = -(disp * normals).sum(dim=1)        # (M,)
        return gap_n
    
    def calculate_complementarity_based_fisher_burmeister(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Compute the Fisher–Burmeister complementarity residual for contact.
        
        The residual is defined as:
            FB(gₙ, Pₙ) = gₙ + (–Pₙ) – √(gₙ² + Pₙ² + ε)
        and measures how well the gap `gₙ` and the normal traction `Pₙ`
        satisfy the contact complementarity conditions.
        
        Only points that lie on the boundary *and* are in contact
        should yield nonzero residuals; elsewhere this will naturally
        evaluate to zero if `gₙ` and `Pₙ` are zero.
        
        Args:
            inputs:  FloatTensor of shape (N,3), the physical coordinates
                    where contact conditions are enforced.
            outputs: FloatTensor of shape (N,9), network predictions
                    [u, v, w, σₓₓ, σᵧᵧ, σ𝓏𝓏, σₓᵧ, σᵧ𝓏, σ𝓏ₓ].
                    
        Returns:
            FloatTensor of shape (N,), the Fisher–Burmeister residual
            at each input point.
        """
        # Compute normal gap and normal traction over all points
        gn = self.calculate_gap_in_normal_direction(inputs=inputs, outputs=outputs)   # (N,)
        _, _, _, Pn, _, _ = self.calculate_traction_mixed_formulation(inputs=inputs, outputs=outputs)  # Pn_full: (N,)
        # Fisher–Burmeister residual on those M points
        eps = torch.tensor(1e-9, dtype=inputs.dtype, device=inputs.device)
        sq_sum = gn * gn + Pn * Pn
        under  = torch.maximum(sq_sum, eps)
        fb  = gn - Pn - torch.sqrt(under)   # note: b = -Pn, so a + b = gn - Pn
        return fb

