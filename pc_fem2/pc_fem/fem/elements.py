# pc_fem/fem/elements.py

import numpy as np

from typing import Tuple, Optional, List, NamedTuple
from abc import ABC

from pc_fem.parameters import Contact_Parameter
from pc_fem.fem.materials import Material
from pc_fem.fem.geo_mesh import Mesh
from pc_fem.fem.damage import Damage

from pc_fem.utils.utils import  local_to_global_index, calculate_principal_values, calculate_equivalent_strain_from_voigt
from pc_fem.utils.log_config import setup_logger


class Element(ABC):
    """Abstract base class for 3D finite elements in structural analysis.

    This base class provides the standard interface and shared functionalities 
    for hexahedral and contact elements. It supports strain computation, 
    stiffness matrix assembly, and damage evaluation.

    Attributes:
        mat (Material): Material model associated with the element.
        dmg (Damage): Damage model used for material degradation.
        mesh (Mesh): Mesh containing nodal coordinates and connectivity.
    """
    def __init__(self, mat: Material, mesh: Mesh, dmg: Damage) -> None:
        """Initializes the base finite element with material, mesh, and damage models.

        Args:
            mat (Material): Object defining elastic and constitutive properties.
            mesh (Mesh): Mesh object holding geometry and topology data.
            dmg (Damage): Object defining the material damage evolution.
        """
        self.mat = mat
        self.mesh = mesh
        self.dmg = dmg
    
    def calculate_single_eps_plastic_eq_ansys(self, eps_p: np.ndarray, nu_prime: float) -> float:
        """Computes equivalent plastic strain from 3 principal strains using ANSYS definition.

        Args:
            eps_p (np.ndarray): Principal strain vector of shape (3,).
            nu_prime (float): Modified Poisson's ratio (usually 0.5 for incompressible plasticity).

        Returns:
            float: Equivalent plastic strain (scalar).
        """
        eps1, eps2, eps3 = eps_p
        return (1 / (1 + nu_prime)) * np.sqrt(
            0.5 * ((eps1 - eps2) ** 2 + (eps2 - eps3) ** 2 + (eps3 - eps1) ** 2)
        )
    
    def calculate_eps_plastic_eq_ansys(self, eps: np.ndarray, nu_prime: float = 0.5) -> np.ndarray:
        """Computes equivalent plastic strain from engineering strain using ANSYS plasticity theory.

        This function supports both single strain vectors and multiple strain states.

        Args:
            eps (np.ndarray): Engineering strain vector(s), shape (6,) or (6, N),
                where the 6 components are [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_zx].
            nu_prime (float, optional): Modified Poisson's ratio. Default is 0.5.

        Returns:
            np.ndarray: Equivalent plastic strain(s), scalar or array of shape (N,).

        Raises:
            ValueError: If `eps` is not of shape (6,) or (6, N).
        """
        if eps.ndim == 1 and eps.shape[0] == 6:
            eps_p = calculate_principal_values(vec=eps, quantity='strain')
            return self.calculate_single_eps_plastic_eq_ansys(eps_p, nu_prime)

        elif eps.ndim == 2 and eps.shape[0] == 6:
            return np.array([
                self.calculate_single_eps_plastic_eq_ansys(
                    calculate_principal_values(vec=eps[:, i], quantity='strain'),
                    nu_prime
                ) for i in range(eps.shape[1])
            ])

        raise ValueError("Input `eps` must have shape (6,) or (6, N).")
    
    def smooth_result(self, n_nodes: int, elements: np.ndarray, values: np.ndarray, avg_threshold: float = 0.75, merge: bool = False) -> np.ndarray:
        """Smooths element-based quantities (e.g., stress, strain) by averaging at shared nodes.

        This function supports two strategies:
            - merge=True: Node-wise averaging from connected element values.
            - merge=False: Local element smoothing if variation is low.

        Args:
            n_nodes (int): Total number of mesh nodes.
            elements (np.ndarray): Element-to-node connectivity, shape (n_elem, 8).
            values (np.ndarray): Element-based data, shape (n_elem, 8).
            avg_threshold (float): Threshold for local smoothing. Defaults to 0.75.
            merge (bool): Whether to merge values to nodes (True) or smooth locally (False).

        Returns:
            np.ndarray:
                - If merge=True: Node-wise averaged results, shape (n_nodes,).
                - If merge=False: Smoothed element values, shape (n_elem, 8).

        Raises:
            ValueError: If shape mismatch between `elements` and `values`.
        """
        # Validate input shapes
        if elements.shape != values.shape:
            raise ValueError("`elements` and `values` must have the same shape.")

        # Flatten the element connectivity and corner values so they align in one dimension.
        # 'ele_flat[i]' is the node index for the i-th corner in the mesh
        # 'val_flat[i]' is the corresponding corner value.
        ele_flat = elements.ravel()    # (n_elem * 8,)
        val_flat = values.ravel()       # (n_elem * 8,)
        
        if merge:
            node_vals = np.zeros(n_nodes, dtype=values.dtype)
            node_counts = np.zeros(n_nodes, dtype=np.int32)

            # Accumulate values at nodes using safe in-place addition
            np.add.at(node_vals, ele_flat, val_flat)
            np.add.at(node_counts, ele_flat, 1)

            # Avoid division by zero
            node_counts[node_counts == 0] = 1

            return node_vals / node_counts
        else:
            # Compute global range for relative variation threshold
            global_range = np.max(val_flat) - np.min(val_flat)

            for node_id in range(n_nodes):
                mask = (ele_flat == node_id)
                if not np.any(mask):
                    continue

                local_vals = val_flat[mask]
                local_range = np.max(local_vals) - np.min(local_vals)

                if 0 < local_range <= avg_threshold * global_range:
                    val_flat[mask] = np.mean(local_vals)

            return val_flat.reshape(values.shape)

class Hexahedral_Element(Element):
    """3D hexahedral finite element for nonlinear structural analysis.

    This class represents an 8-node hexahedral element used in 3D FEA. It supports 
    element-level operations including shape function evaluation, stiffness matrix 
    assembly, and damage-integrated stress updates.

    Attributes:
        natural_coords (np.ndarray): Reference coordinates in the natural coordinate system (8×3).
        logger (logging.Logger): Logger instance for tracking element behavior.
    """
    
    def __init__(self, mat: Material, mesh: Mesh, dmg: Damage, natural_coords: Optional[np.ndarray] = None) -> None:
        """Initializes a Hexahedral_Element instance.

        Args:
            mat (Material): Material model containing constitutive properties.
            mesh (Mesh): Global mesh structure.
            dmg (Damage): Damage model for material degradation.
            natural_coords (Optional[np.ndarray], optional): 
                Natural coordinate locations of nodes in the [-1, 1] cube. 
                If None, uses the standard 8-node layout. Defaults to None.
        """
        super().__init__(mat, mesh, dmg)
        # Note that the order of nodes must be keep identical with node connection of an element
        #      ^ z
        #      |
        #      8---------7
        #     /|        /|
        #    / |       / |
        #   5---------6  |
        #   |  |      |  |
        #   |  4------|--3 ------> y
        #   | /       | /
        #   |/        |/
        #   1---------2
        #  / x
        # v
        # Natural coordinates of hexahedral element (in [-1,1] space)
        # Use provided natural coordinates or default to standard hexahedral layout
        self.natural_coords = (
            natural_coords if natural_coords is not None else np.array([
                [ 1, -1, -1],  # Node 1
                [ 1,  1, -1],  # Node 2
                [-1,  1, -1],  # Node 3
                [-1, -1, -1],  # Node 4
                [ 1, -1,  1],  # Node 5
                [ 1,  1,  1],  # Node 6
                [-1,  1,  1],  # Node 7
                [-1, -1,  1],  # Node 8
            ])
        )
        
        # Setup logger
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info("Initialized Hexahedral_Element.")
        self.logger.info(
            "Natural coordinates shape = %s | Custom input = %s",
            self.natural_coords.shape,
            natural_coords is not None
        )

    def calculate_shape_function_derivatives(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Computes the derivatives of shape functions with respect to natural coordinates 
        (ξ, η, ζ) for an 8-node hexahedral element.

        This function evaluates the partial derivatives ∂N/∂ξ, ∂N/∂η, and ∂N/∂ζ 
        at a given integration (Gauss) point in the element's reference (natural) coordinate system.

        Args:
            xi (float): Natural coordinate ξ ∈ [-1, 1].
            eta (float): Natural coordinate η ∈ [-1, 1].
            zeta (float): Natural coordinate ζ ∈ [-1, 1].

        Returns:
            np.ndarray: A (3 × 8) matrix containing shape function derivatives with respect to
                ξ, η, and ζ. Each row corresponds to a natural direction:
                    - Row 0: ∂N_i/∂ξ
                    - Row 1: ∂N_i/∂η
                    - Row 2: ∂N_i/∂ζ
        """
        dN_dxi = []
        dN_deta = []
        dN_dzeta = []
        
        for xi_i, eta_i, zeta_i in self.natural_coords:
            # Chain-rule derivatives for trilinear shape functions
            dN_dxi.append(0.125 * xi_i * (1 + eta_i * eta) * (1 + zeta_i * zeta))
            dN_deta.append(0.125 * eta_i * (1 + xi_i * xi) * (1 + zeta_i * zeta))
            dN_dzeta.append(0.125 * zeta_i * (1 + xi_i * xi) * (1 + eta_i * eta))
        
        # Stack as a 3x8 matrix: (∂N/∂ξ, ∂N/∂η, ∂N/∂ζ)
        dN_dnatural = np.vstack((dN_dxi, dN_deta, dN_dzeta))  # 3 x 8
        
        return dN_dnatural
    
    def shape_function(self, xi: float, eta: float, zeta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Computes shape functions and their derivatives for a hexahedral element.

        Shape function:
            Ni = (1 + xi * xi_i) * (1 + eta * eta_i) * (1 + zeta * zeta_i) / 8
        
        Args:
            xi (float): Natural coordinate in x direction.
            eta (float): Natural coordinate in y direction.
            zeta (float): Natural coordinate in z direction.

        Returns:
            tuple:
                - Ni (np.ndarray): Shape functions (8,).
                - dN_dxi (np.ndarray): Shape function derivatives (3, 8).
        """
        
        coords = self.natural_coords # Access natural coordinates
        
        # Compute shape functions
        # Ni = (1+xi*xi_i)*(1+eta*eta_i)*(1+zeta*zeta_i)/8 (11.3.3), p548
        Ni = np.array([
            (1 + xi * xi_i) * (1 + eta * eta_i) * (1 + zeta * zeta_i) / 8
            for xi_i, eta_i, zeta_i in coords
        ])
        
        # Compute derivatives of shape functions (Jacobian matrix J)
        dN_dxi = np.array([ # this is the Jacobian matrix J
            [   (1-eta)*(1-zeta), (1+eta)*(1-zeta), -(1+eta)*(1-zeta), -(1-eta)*(1-zeta),
                (1-eta)*(1+zeta), (1+eta)*(1+zeta), -(1+eta)*(1+zeta), -(1-eta)*(1+zeta)],
            [   -(1+xi)*(1-zeta),  (1+xi)*(1-zeta),   (1-xi)*(1-zeta),  -(1-xi)*(1-zeta),
                -(1+xi)*(1+zeta),  (1+xi)*(1+zeta),   (1-xi)*(1+zeta),  -(1-xi)*(1+zeta)],
            [    -(1+xi)*(1-eta),  -(1+xi)*(1+eta),   -(1-xi)*(1+eta),   -(1-xi)*(1-eta),
                  (1+xi)*(1-eta),   (1+xi)*(1+eta),    (1-xi)*(1+eta),    (1-xi)*(1-eta)]
        ])/8.0 # Shape derivative matrix (3x8)
        
        return Ni, dN_dxi
    
    def calculate_B(self, xi: float, eta: float, zeta: float, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the strain-displacement matrix B for a hexahedral element.

        The function calculates the strain-displacement matrix (B-matrix) and 
        the determinant of the Jacobian (detJ), which are used in finite element 
        stiffness calculations.

        Args:
            xi (float): Natural coordinate in x direction.
            eta (float): Natural coordinate in y direction.
            zeta (float): Natural coordinate in z direction.
            coords (np.ndarray): Physical coordinates of the element nodes (8,3).

        Returns:
            tuple:
                - B (np.ndarray): Strain-displacement matrix (6, 24).
                - detJ (float): Determinant of the Jacobian matrix.

        Raises:
            ValueError: If the determinant of the Jacobian matrix is non-positive.
        """
        
        assert coords.shape == (8, 3), "coords must be of shape (8, 3)"
        
        # Compute shape function and its derivatives
        Ni, dN_dxi = self.shape_function(xi, eta, zeta)  # (3x8)
        
        # Compute Jacobian matrix
        J = np.dot(dN_dxi, coords)
        detJ = np.linalg.det(J)
        
        if detJ <= 0:
            self.logger.error(
            "Non-positive Jacobian determinant encountered:\n"
            "  detJ       = %.6e\n"
            "  J          = \n%s\n"
            "  dN_dxi     = \n%s\n"
            "  coords     = \n%s\n"
            "  xi, eta, zeta     = (%.3f, %.3f, %.3f)",
            detJ,
            np.array2string(J, precision=4, suppress_small=True),
            np.array2string(dN_dxi, precision=4, suppress_small=True),
            np.array2string(coords, precision=4, suppress_small=True),
            xi, eta, zeta
            )
            raise ValueError("Jacobian determinant is non-positive!")
        
        condJ = np.linalg.cond(J)
        if condJ > 1e8:
            self.logger.debug(
            "Jacobian is ill-conditioned at Gauss point (xi, eta, zeta):\n"
            "  detJ       = %.6e\n"
            "  J          = \n%s\n"
            "  dN_dxi     = \n%s\n"
            "  coords     = \n%s\n"
            "  xi, eta, zeta     = (%.3f, %.3f, %.3f)",
            detJ,
            np.array2string(J, precision=4, suppress_small=True),
            np.array2string(dN_dxi, precision=4, suppress_small=True),
            np.array2string(coords, precision=4, suppress_small=True),
            xi, eta, zeta
            )
            print(f"⚠️ Warning: Jacobian is ill-conditioned at Gauss point ({xi}, {eta}, {zeta})")
        
        # Compute inverse of Jacobian
        J_inv = np.linalg.inv(J)
        
        # Compute derivatives of shape functions in physical space
        q = np.dot(J_inv, dN_dxi)
        qx, qy, qz = q[0, :], q[1, :], q[2, :]
        
        # Construct strain-displacement matrix (B-matrix)
        B = np.zeros((6, 24))
        # Box 2.2 page 36, Rene de Borst, Mike A. Crisfield, et al.,
        # Non-linear Finite Element Analysis of Solids and Structures, 2nd edition
        # Equation (11.2.14), page 543, Daryl L. Logan.
        B[0, 0::3] = qx  # ε_xx
        B[1, 1::3] = qy  # ε_yy
        B[2, 2::3] = qz  # ε_zz
        B[3, 0::3], B[3, 1::3] = qy, qx  # γ_xy
        B[4, 1::3], B[4, 2::3] = qz, qy  # γ_yz
        B[5, 0::3], B[5, 2::3] = qz, qx  # γ_zx
        
        return B, detJ
    
    def calculate_element_stiffness(self, node_coords: np.ndarray, u: np.ndarray, d0: np.ndarray, B_bar: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Computes the element stiffness matrix for a hexahedral element using 2x2x2 Gauss quadrature.

        Supports B-bar method to reduce volumetric locking in nearly incompressible materials.


        This function applies the finite element method (FEM) to compute the element stiffness matrix
        using 2x2x2 Gauss integration. The constitutive matrix is computed using Mazars' damage model.

        Reference:
            - https://github.com/YX-Song0526/SimpleC3D8/blob/main/stiffness_matrix.py

        Args:
            node_coords (np.ndarray): Node coordinates of the element, shape (8,3).
            u (np.ndarray, optional): Nodal displacement vector, shape (24,1). Defaults to a zero array.
            d0 (np.ndarray): Scalar damage values (8,) from the previous increment (0 ≤ d < 1).
            B_bar (bool, optional): Whether to apply the B-bar method for the volumetric strain. 

        Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - Ke (np.ndarray): The element stiffness matrix (24x24).
            - current_d (np.ndarray): Current scalar damage values (8,).

        Raises:
            AssertionError: If `node_coords` is not of shape (8,3).
        """
        
        # Ensure node coordinates have correct shape
        assert node_coords.shape == (8, 3), "Input `node_coords` must have shape (8,3)"
        assert d0.shape == (8, 3), "Input `d0` must have shape (8,3)"

        # Default displacement vector (zero)
        if u is None:
            u = np.zeros((24, 1))
        
        # Retrieve material properties
        E = self.mat.prop.E  # Young's modulus
        
        # Initialize stiffness matrix
        total_dof = 24  # 8 nodes * 3 DOFs per node
        Ke = np.zeros((total_dof, total_dof)) # 24x24
        
        # Gauss quadrature points and weights (2x2x2 integration)
        gauss_points = np.array([-np.sqrt(1 / 3), np.sqrt(1 / 3)])  # [-0.577, 0.577]
        gauss_weights = np.array([1, 1])  # Uniform weight for 2-point Gauss integration
        
        # Precompute B_vol sum for B-bar
        if B_bar:
            Bvol_sum = np.zeros((3, total_dof))
            for zeta in gauss_points:
                for eta in gauss_points:
                    for xi in gauss_points:
                        B, _ = self.calculate_B(xi, eta, zeta, node_coords)
                        Bvol_sum += B[:3, :]  # ε_xx, ε_yy, ε_zz rows
            Bvol_avg = Bvol_sum / 8
        
        current_d = []
        current_epsilon = []
        current_eps_eq = []
        gp_count = 0
        # Compute stiffness matrix using Gaussian quadrature
        for gp_zeta, wt in zip(gauss_points, gauss_weights):
            for gp_eta, wn in zip(gauss_points, gauss_weights):
                for gp_xi, ws in zip(gauss_points, gauss_weights):
                    # Compute strain-displacement matrix and determinant of Jacobian
                    B, detJ = self.calculate_B(gp_xi, gp_eta, gp_zeta, node_coords)
                    
                    if B_bar:
                        # Replace volumetric strain part with average
                        B_mod = B.copy()
                        B_mod[:3, :] = Bvol_avg
                    else:
                        B_mod = B
                    
                    # Compute strain and Mazars' damage
                    eps = B @ u # (6x24) @ (24x1) -> (6x1)
                    damage = self.dmg.calculate_damage(eps=eps, d0=d0[gp_count], clip=False) # (6,)
                    current_d.append(damage)
                    current_epsilon.append(eps.ravel())
                    
                    # Compute equivalent strain for each mode
                    eps_eq = calculate_equivalent_strain_from_voigt(vec=eps.ravel())
                    current_eps_eq.append(eps_eq)
                    
                    # Compute damaged elastic modulus
                    control = self.dmg.para.control.strip().lower()
                    valid_modes = {
                        "ct": 0, "tc": 0, "compression-tension": 0, "tension-compression": 0,
                        "t": 1, "tension": 1,
                        "c": 2, "compression": 2,
                    }
                    if control in valid_modes:
                        d = damage[valid_modes[control]]
                    else:
                        raise ValueError(
                            f"[Hexahedral_Element] Invalid control mode '{self.dmg.para.control}'.\n"
                            f"-> Expected one of: {', '.join(sorted(valid_modes.keys()))}"
                        )
                    
                    damaged_E = float((1-d)*E)
                    
                    # Compute constitutive matrix (D)
                    D = self.mat.calculate_D(E=damaged_E)
                    
                    # Compute element stiffness matrix using Gauss integration
                    Ke += B_mod.T @ D @ B_mod * detJ * ws * wn * wt # (24x6) @ (6x6) @ (6x24) -> (24x24)
                    
                    gp_count += 1
        
        return Ke, np.array(current_d), np.array(current_epsilon), np.array(current_eps_eq)

    def calculate_global_stiffness(self, nodes: np.ndarray, elements: np.ndarray, u: Optional[np.ndarray], d0: np.ndarray, B_bar: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Assembles the global stiffness matrix for the entire structure.

        This method loops through all elements, computes each element's stiffness matrix
        (which may depend on deformation `u` in nonlinear analysis), and assembles them
        into a global stiffness matrix using standard finite element procedures.

        Args:
            nodes (np.ndarray): Nodal coordinates array of shape (num_nodes, 3).
            elements (np.ndarray): Element connectivity array of shape (num_elements, 8),
                where each row contains the indices of the 8 nodes forming a hexahedral element.
            u (Optional[np.ndarray], optional): Global displacement vector of shape (total_dof, 1).
                Required for nonlinear problems. If None, linear stiffness is used. Defaults to None.
            d0 (np.ndarray): Scalar damage values (num_elements, 8) from the previous increment (0 ≤ d < 1).
            B_bar (bool, optional): Whether to apply the B-bar method for the volumetric strain. 

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - global_k (np.ndarray): Global stiffness matrix of shape (total_dof, total_dof), where 
                    `total_dof = num_nodes * 3` for 3D problems.
                - current_d (np.ndarray): Current scalar damage values (8,).
                np.ndarray: 

        Attributes Set:
            self.global_k (np.ndarray): The assembled global stiffness matrix.
        
        Raises:
            ValueError: If element stiffness computation fails.
        """
        # Default displacement vector (zero)
        if u is None:
            u = np.zeros((24, 1))
        
        self.logger.debug(
            "Starting computation of global stiffness matrix for %d elements and %d nodes...",
            elements.shape[0], nodes.shape[0]
        )
        
        n_nodes, ndof = nodes.shape # ndof = 3 for 3D elements
        total_dof = n_nodes * ndof

        global_k = np.zeros((total_dof, total_dof))  # Initialize global stiffness matrix
        
        eps = np.zeros((len(elements), 8, 6)) # 8 Gaussian integration points
        eps_eq = np.zeros((len(elements), 8)) # 8 Gaussian integration points

        for e, ele in enumerate(elements):
            # Get coordinates of the current element's nodes
            coords = nodes[ele]  # shape: (8, 3)
            
            # Convert element node indices to global DOF indices
            global_index = local_to_global_index(ele, dof=ndof)

            # Compute element stiffness matrix (24×24)
            ke, current_d, current_eps, current_eps_eq = self.calculate_element_stiffness(node_coords=coords, u=u[global_index], d0=d0[e], B_bar=B_bar)

            d0[e] = current_d
            eps[e] = current_eps
            eps_eq[e] = current_eps_eq
            
            # Assemble local ke into global_k
            for (i, j), k_val in np.ndenumerate(ke):
                global_k[global_index[i], global_index[j]] += k_val

        # Store in object
        self.global_k = global_k
        self.current_damage = d0
        self.current_eps_ep = eps_eq
        self.current_eps = eps
        
        return global_k, d0

    def calculate_element_stress_strain(self, node_coords: np.ndarray, u: np.ndarray, current_d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes element stress and strain at Gauss points using Mazars' damage model.

        This function calculates the stress and strain at each Gauss point in a hexahedral element.
        The stiffness degradation due to material damage is handled via Mazars' damage model.

        Args:
            node_coords (np.ndarray): Node coordinates, shape (8,3).
            u (np.ndarray): Nodal displacement vector, shape (24,1).
            current_d (np.ndarray): Scalar damage values (8,) from the previous increment (0 ≤ d < 1).
            B_bar (bool, optional): Whether to apply the B-bar method for the volumetric strain.

        Returns:
            Tuple:
                - stress (np.ndarray): Stress tensor at Gauss points, shape (6, 8).
                - strain (np.ndarray): Strain tensor at Gauss points, shape (6, 8).

        Raises:
            AssertionError: If `nodes_coordinates` is not of shape (8,3).
        """
        # Validate input
        assert node_coords.shape == (8, 3), "Input `node_coords` must have shape (8,3)"

        # Initialize stress and strain storage
        sig_list, eps_list = [], []
        
        # Loop over natural (Gauss) points
        gp_count = 0
        for xi, eta, zeta in self.natural_coords:
            # Compute strain-displacement matrix (B) at Gauss point
            B, _, = self.calculate_B(xi, eta, zeta, node_coords)
            
            # Compute strain at Gauss point
            eps = B @ u # (6x24) @ (24x1) -> (6x1)
            
            # Compute material damage using Mazars' model
            damage = current_d[gp_count]
            control = self.dmg.para.control.strip().lower()
            valid_modes = {
                "ct": 0, "tc": 0, "compression-tension": 0, "tension-compression": 0,
                "t": 1, "tension": 1,
                "c": 2, "compression": 2,
            }
            if control in valid_modes:
                d = damage[valid_modes[control]]
            else:
                raise ValueError(
                    f"[Hexahedral_Element] Invalid control mode '{self.dmg.para.control}'.\n"
                    f"→ Expected one of: {', '.join(sorted(valid_modes.keys()))}"
                )
            
            damaged_E = (1 - d) * self.mat.prop.E
            
            # Compute constitutive matrix with damaged elasticity
            D = self.mat.calculate_D(E=damaged_E)
            
            # Compute stress at Gauss point
            sig = D @ eps  # (6x6) @ (6x1) -> (6x1)
            
            # Store stress and strain
            sig_list.append(sig.reshape(-1, 1))
            eps_list.append(eps.reshape(-1, 1))
            
            gp_count += 1
        
        # Convert lists to NumPy arrays (6x8)
        stress = np.hstack(sig_list)  # (6,8)
        strain = np.hstack(eps_list)  # (6,8)
        
        return stress, strain

class ContactVectors(NamedTuple):
    basics: List[np.ndarray]                    # x̄, g_N, n̄, a1̄, a2̄, ξ̄
    B_s: np.ndarray                             # Shape matrix for stick contact
    E: np.ndarray                               # Tangential stiffness matrix
    N_vec: np.ndarray                           # Normal projection of shape functions
    N_alpha: List[np.ndarray]                   # First-order derivatives of N wrt ξ
    T_alpha: List[np.ndarray]                   # Tangent projections (T1, T2)
    T_tilde: List[np.ndarray]                   # Zero tensor for curvature (always 0)
    D_alpha: List[np.ndarray]                   # D1, D2 for contact stiffness
    E_alpha: List[np.ndarray]                   # Derivatives of E (E1, E2)

class ContactMatrices(NamedTuple):
    a_bar_alpha_beta: np.ndarray                # Surface metric tensor (covariant)
    a_alpha_beta: np.ndarray                    # Surface metric tensor (contravariant)
    b_bar_alpha_beta: np.ndarray                # Surface curvature tensor (zero here)
    H_alpha_beta: np.ndarray                    # Surface projection tensor
    T_alpha_beta: np.ndarray                    # Tangent second derivatives
    N_alpha_beta: np.ndarray                    # Normal second derivatives

class Contact_Element(Hexahedral_Element):
    """Contact element for simulating interfacial behavior in layered or bonded materials.

    This class extends the `Hexahedral_Element` to support contact mechanics, particularly 
    for modeling inter-layer contact in 3D printed concrete or similar materials. It introduces 
    penalty-based enforcement for normal and tangential constraints and will serve as the foundation 
    for implementing cohesive or frictional contact behavior.

    Attributes:
        hexa (Hexahedral_Element): Reference to the bulk element providing geometry, shape functions, 
            and material data.
        para (Contact_Parameter): Contact interface parameters, such as penalty stiffness and friction coefficient.
    """
    def __init__(self, hexa: Hexahedral_Element, para: Contact_Parameter) -> None:
        """Initializes a Contact_Element with geometry, material, and contact interface properties.

        Args:
            hexa (Hexahedral_Element): Underlying bulk hexahedral element used for computing geometry 
                and stiffness matrices.
            para (Contact_Parameter): Contact parameter object containing:
                - epsilon (list[float]): Penalty stiffness for [normal, tangential] directions.
                - mu (float): Friction coefficient.
                - (optional) Additional parameters for future extensions (e.g., cohesion, gap tolerance).
        """
        self.hexa = hexa
        self.para = para
        
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info("Initialized Contact_Element with properties: %s", vars(self.para))
    
    def _sort_points_by_distance(self, p: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Sorts a set of points based on their Euclidean distance to a reference point.

        This utility function is useful when aligning or reordering nodes on a contact surface
        relative to a target point for consistent normal and tangential direction calculations.

        Args:
            p (np.ndarray): A 1D array of shape (3,) representing the reference point [x, y, z].
            points (np.ndarray): A 2D array of shape (N, 3) representing N 3D points.

        Returns:
            np.ndarray: A new (N, 3) array with the input points sorted by increasing distance to `p`.

        Raises:
            ValueError: If input shapes are not valid 3D coordinates.
        """
        if p.shape != (3,) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Input `p` must be shape (3,) and `points` must be shape (N, 3)")

        # Compute Euclidean distances from p to each point
        distances = np.linalg.norm(points - p, axis=1)

        # Sort distances in ascending order and reorder points accordingly
        sorted_points = points[np.argsort(distances)]

        return sorted_points
    
    def _point_projection_to_plane(self, point: np.ndarray, plane_points: np.ndarray) -> Tuple[float, np.ndarray]:
        # Edited by ChatGPT so check is necessary
        """Projects a 3D point onto a plane defined by four points and returns the signed distance.

        The function first sorts the plane points by distance to the given point, computes
        the plane normal from two in-plane vectors, and calculates the projection of the point
        onto the plane along the direction of the normal.

        Args:
            point (np.ndarray): A 1D array of shape (3,) representing the 3D point to project [x, y, z].
            plane_points (np.ndarray): A (4, 3) array of four points on the plane.

        Returns:
            Tuple[float, np.ndarray]:
                - distance (float): Signed distance from the point to the plane (positive if above).
                - projected_point (np.ndarray): The 3D coordinates of the projection on the plane.

        Raises:
            ValueError: If the plane is degenerate (normal vector magnitude is zero),
                        or if input shapes are invalid.
        """
        # Validate input
        if point.shape != (3,) or plane_points.shape != (4, 3):
            raise ValueError("`point` must be shape (3,) and `plane_points` must be shape (4, 3)")

        # Sort plane points by proximity to `point` to ensure numerical robustness
        A, B, C, D = self._sort_points_by_distance(point, plane_points)

        # Construct two in-plane vectors
        AB = B - A
        AC = C - A

        # Compute plane normal using cross product
        normal = np.cross(AB, AC)
        normal_norm = np.linalg.norm(normal)

        # Check if the plane is well-defined (non-degenerate)
        if normal_norm == 0:
            raise ValueError("Degenerate plane: the provided points do not form a valid surface.")

        # Normalize normal vector
        normal = normal / normal_norm

        # Project point onto plane using dot product
        distance = np.dot(normal, point - A)
        projected_point = point - distance * normal

        return distance, projected_point

    def calculate_natural_coordinates(self, point: np.ndarray, plane_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Computes the natural coordinates of a point projected onto a quadrilateral surface.

        This is commonly used in contact mechanics to map a point on a physical surface
        into a local parametric (ξ₁, ξ₂) coordinate system. It also returns the two
        in-plane tangent vectors and the surface normal.

        Args:
            point (np.ndarray): A (3,) array representing the coordinates of the point in 3D space.
            plane_points (np.ndarray): A (4, 3) array of four 3D points defining a quadrilateral surface.
                The order should follow: [A, B, C, D] (counterclockwise recommended).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - natural_coords (np.ndarray): Local natural coordinates [ξ₁, ξ₂] ∈ [-1, 1].
                - xi1_tangent (np.ndarray): Unit vector along ξ₁ direction in 3D space.
                - xi2_tangent (np.ndarray): Unit vector along ξ₂ direction in 3D space.
                - normal (np.ndarray): Unit normal vector of the surface.

        Raises:
            ValueError: If the shape of inputs is invalid or normalization fails.
        """
        if point.shape != (3,) or plane_points.shape != (4, 3):
            raise ValueError("`point` must be shape (3,), and `plane_points` must be shape (4, 3)")

        # Unpack points defining the surface
        A, B, C, D = plane_points

        # Compute center of the quadrilateral (used as origin of natural coord system)
        center = (A + B + C + D) / 4.0

        # Define tangent directions (axes of natural coordinates)
        xi1_vec = (B + C) / 2 - (A + D) / 2  # Midpoint BC - midpoint AD
        xi2_vec = (C + D) / 2 - (A + B) / 2  # Midpoint CD - midpoint AB
        
        xi1_length = np.linalg.norm(xi1_vec) / 2
        xi2_length = np.linalg.norm(xi2_vec) / 2
        
        if xi1_length == 0 or xi2_length == 0:
            raise ValueError("Degenerate quadrilateral: zero length axis.")

        # Normalize the axes
        xi1_tangent = xi1_vec / np.linalg.norm(xi1_vec)
        xi2_tangent = xi2_vec / np.linalg.norm(xi2_vec)

        # Normal vector (cross product of two tangent directions)
        normal = np.cross(xi1_tangent, xi2_tangent)
        normal = normal / np.linalg.norm(normal)

        # Vector from center to the projected point
        local_vec = point - center

        # Compute local coordinates in the xi1-xi2 plane
        xi1_coord = np.dot(local_vec, xi1_tangent) / xi1_length
        xi2_coord = np.dot(local_vec, xi2_tangent) / xi2_length

        natural_coords = np.array([xi1_coord, xi2_coord])

        return natural_coords, xi1_tangent, xi2_tangent, normal

    def calculate_shape_function_N(self, inner_point: np.ndarray) -> List[np.ndarray]:
        """Computes the shape function N and its derivatives at a natural coordinate (ξ, η).

        This is used for surface integration in contact formulations, where
        the shape functions and their derivatives with respect to natural coordinates
        are needed for assembling contact stiffness matrices.

        Args:
            inner_point (np.ndarray): A (2,) array representing the natural coordinates (ξ, η)
                of a point on a 2D quadrilateral face.

        Returns:
            List[np.ndarray]: A list of shape function quantities as (5, 1) column vectors:
                - N (shape function values)
                - Nξ (1st derivative with respect to ξ)
                - Nη (1st derivative with respect to η)
                - Nξη (2nd mixed partial derivative)
                - Nηξ (identical to Nξη, due to symmetry)
                - Nξξ (second derivative w.r.t. ξ, always zero for bilinear)
                - Nηη (second derivative w.r.t. η, always zero for bilinear)
        """
        xi, eta = inner_point
        
        # Corner natural coordinates of 4-node quadrilateral in order: A, B, C, D
        node_coords = [(-1, -1), (1, -1), (1, 1), (-1, 1)] # (7.25) P164
        
        # N: shape function values at (ξ, η) (5x1)
        N = [0.5 * (1 + xi_i * xi) * 0.5 * (1 + eta_i * eta) for xi_i, eta_i in node_coords]  # (7.24) P164
        N = np.array([-1] + N).reshape(-1, 1)  # Add dummy -1 at index 0 for 1-based indexing
        
        # Nξ: partial derivatives with respect to ξ
        # N_{I,alpha} (alpha=1), i.e., N_{I,1} (5x1)
        dN_dxi = [0.25 * xi_i * (1 + eta_i * eta) for xi_i, eta_i in node_coords]
        dN_dxi = np.array([0] + dN_dxi).reshape(-1, 1) # Add dummy 0 at index 0 for 1-based indexing
        
        # Nη: partial derivatives with respect to η
        # N_{I,alpha} (alpha=2), i.e., N_{I,2} (5x1)
        dN_deta = [0.25 * eta_i * (1 + xi_i * xi) for xi_i, eta_i in node_coords]
        dN_deta = np.array([0] + dN_deta).reshape(-1, 1)
        
        # Second derivatives:
        # Nξξ and Nηη are zero for bilinear elements
        # N_(I, alpha beta) (alpha=1, beta=1), i.e., N_{I,11}
        d2N_dxi2 = np.zeros((5, 1))
        # N_(I, alpha beta) (alpha=2, beta=2), i.e., N_{I,22}
        d2N_deta2 = np.zeros((5, 1))
        
        # Mixed partial derivatives (Nξη and Nηξ are identical)
        # N_(I, alpha beta) (alpha=1, beta=2), i.e., N_{I,12} (5x1)
        # N_(I, alpha beta) (alpha=2, beta=1), i.e., N_{I,21} (5x1)
        d2N_dxideta = [0.25 * xi_i * eta_i for xi_i, eta_i in node_coords]
        d2N_dxideta = np.array([0] + d2N_dxideta).reshape(-1, 1)
        
        # (5x1)
        return [N, dN_dxi, dN_deta, d2N_dxi2, d2N_dxideta, d2N_dxideta, d2N_deta2]
    
    def calculate_vectors_and_matrices(self, slave_node_coord: np.ndarray, surface_node_coord: np.ndarray) -> Tuple[ContactVectors, ContactMatrices]:
        """Computes key vectors and matrices for contact tangential stiffness formulation.

        This method is used in the context of point-to-surface contact elements. It calculates
        quantities such as projection point, normal gap, tangential basis vectors, shape function
        derivatives, and geometric transformation terms needed for tangential contact stiffness.

        Args:
            slave_node_coord (np.ndarray): Coordinates of the slave node (3D vector).
            surface_node_coord (np.ndarray): Coordinates of the 4 surface nodes (4 x 3 array).

        Returns:
            Tuple[ContactVectors, ContactMatrices]: 
                - vectors (list): A list containing projection vectors, normal gap, 
                tangential shape function matrices (N, T), and intermediate vectors for stiffness.
                - matrices (list): A list of 2×2 or 2×2×n matrices, including metric tensors, curvature, etc.

        Raises:
            ValueError: If projection or natural coordinate mapping fails.
        """
        # === 1. Geometry setup ===
        # x_bar: projection point on the surface
        g_N, x_bar = self._point_projection_to_plane(slave_node_coord, surface_node_coord)
        xi_bar, a1_bar, a2_bar, n_bar = self.calculate_natural_coordinates(x_bar, surface_node_coord)
        basics = [x_bar, g_N, n_bar, a1_bar, a2_bar, xi_bar]
        
        # (5x1)
        N, N1, N2, N11, N12, N21, N22 = self.calculate_shape_function_N(xi_bar)
        
        # === 3. B_s and E matrix (for tangential stiffness, 15x3) ===
        # B_s for stick; (9.87) p244
        B_s = np.vstack([-Ni * np.eye(3) for Ni in N])
        
        E = B_s.copy() # E for K_T1; (9.101) p246
        # E_alpha for K_T1; (9.101) p246
        E1 = np.vstack([-Ni * np.eye(3) for Ni in N1]) # 15x3
        E2 = np.vstack([-Ni * np.eye(3) for Ni in N2]) # 15x3
        
        # === 4. Normal direction shape functions ===
        n_bar = n_bar.reshape(1, -1)   # 1x3
        
        def build_vector(coeff: np.ndarray, direction: np.ndarray) -> np.ndarray:
            return (-coeff @ direction).reshape(-1, 1) # (5x1) @ (1x3) -> (5x3) -> (15,1)
        
        # N (15x1)
        N_vec = build_vector(N, n_bar) # 15x1
        # N_alpha (15x1)
        N1_vec = build_vector(N1, n_bar) # 15x1
        N2_vec = build_vector(N2, n_bar) # 15x1
        # N_{alpha beta} (15x1)
        N11_vec = build_vector(N11, n_bar) # 15x1
        N12_vec = build_vector(N12, n_bar) # 15x1
        N21_vec = build_vector(N21, n_bar) # 15x1
        N22_vec = build_vector(N22, n_bar) # 15x1
        
        # Normal derivative terms
        N_alpha_beta = np.array([[N11_vec, N12_vec], [N21_vec, N22_vec]], dtype=object)
        
        # === 5. Metric tensors (a_alpha_beta, H_alpha_beta) ===
        # a^{alpha beta} a_{beta gamma} = delta^alpha_gamma (B.18) p480
        # a^{alpha beta} is also identity because 
        # a^{bar}_{alpha beta} is identity
        a_alpha_beta = np.identity(2)      # contravariant
        
        # a^{bar}_{alpha beta} = a^{bar}_alpha dot a^{bar}_beta  (B.17) p480
        # a^{bar}_alpha are actual orthogonal unit vectors, 
        # leading to a^{bar}_{alpha beta} is identity
        a_bar_alpha_beta = np.identity(2)  # covariant
        
        # H^{alpha beta} = H_{alpha beta} which is identity
        H_alpha_beta = np.identity(2)
        
        # b_bar_{alpha, beta} = 0 (B.17) p480, a1,1 = a2,2 = 0
        b_bar_alpha_beta = np.zeros((2, 2), dtype=object)  # zero curvature
        
        # === 6. T_alpha_beta derivatives (for curvature or update) ===
        a1_bar = a1_bar.reshape(1, -1) # 1x3
        a2_bar = a2_bar.reshape(1, -1) # 1x3
        
        # T_alpha (15x1)
        T1_vec = build_vector(N, a1_bar) # 15x1
        T2_vec = build_vector(N, a2_bar) # 15x1
        
        # T_{alpha beta} (15x1)
        T11 = build_vector(N1, a1_bar) # 15x1
        T12 = build_vector(N2, a1_bar) # 15x1
        T21 = build_vector(N1, a2_bar) # 15x1
        T22 = build_vector(N2, a2_bar) # 15x1
        T_alpha_beta = np.array([[T11, T12], [T21, T22]], dtype=object)
        
        # T_tilde_{alpha beta} = N_I * a_bar_{alpha, beta}
        # so the values are 0 because a_bar_{alpha, beta} = 0
        T_tilde = np.zeros_like(N_vec) # T_tilde is zero since curvature = 0
        
        # === 7. D_alpha = H^{alpha beta} * (T_beta - g_N * N_beta)
        # D_alpha = H^{alpha beta} [T_beta - g_N N_beta] # 15x1
        D1 = H_alpha_beta[0,0] * (T1_vec - g_N*N1_vec) # (9.93) p245 # H12 = 0 for beta = 2
        D2 = H_alpha_beta[1,1] * (T2_vec - g_N*N2_vec) # (9.93) p245 # H21 = 0 for beta = 2
        
        # === Pack output ===
        vectors = [
            basics,      # x̄, g_N, n̄, a1̄, a2̄, ξ̄
            B_s, E,
            N_vec,
            [N1_vec, N2_vec],
            [T1_vec, T2_vec],
            [T_tilde, T_tilde],
            [D1, D2],
            [E1, E2],
        ]
        matrices = [
            a_bar_alpha_beta,
            a_alpha_beta,
            b_bar_alpha_beta,
            H_alpha_beta,
            T_alpha_beta,
            N_alpha_beta,
        ]
        
        return ContactVectors(*vectors), ContactMatrices(*matrices)
    #TODO
    def calculate_tangential_vectors_and_matrices(self, slave_node_coord: np.ndarray, surface_node_coord : np.ndarray,
            delta_u: np.ndarray, epsilon: list, x_xi_0: Optional[np.ndarray] = None) -> List[np.ndarray]:
        # x_hat: projection point on the surface
        g_N, x_hat = self._point_projection_to_plane(
                slave_node_coord, surface_node_coord)
        xi_bar, a_bar_1, a_bar_2, n_bar_c = self.calculate_natural_coordinates(
                x_hat, surface_node_coord)
        N_I, N1_coef, N2_coef, N11_coef, N12_coef, N21_coef, N22_coef = self.calculate_shape_function_N(xi_bar)
        
        if x_xi_0 is None:
            x_xi_0 = x_hat
        # g_s = slave_node_coord - x_hat # (9.83) p243 # a vector instead of a scalar
        g_s = x_hat - x_xi_0 # a vector instead of a scalar
        
        # P_alpha # 15x1
        # slave_u = delta_u[:3] # the first displacement
        # xi_bar_tr, a_bar_1_tr, a_bar_2_tr, n_bar_c_tr = self.__compute_natural_coordinates(
        #         x_hat+slave_u, surface_node_coord)
        # delta_xi = xi_bar_tr - xi_bar
        # epsilon_N, epsilon_T = epsilon
        # t_tr_T1 = epsilon_T * (g_s + slave_u)
        # t_tr_T = epsilon_T * g_s # 1x3
        # P1 = -N1_coef*t_tr_T.reshape(1,-1) # 5x3
        # P1 = N1.reshape(-1,1) # 15x1
        # P2 = -N2_coef*t_tr_T.reshape(1,-1) # 5x3
        # P2 = N2.reshape(-1,1) # 15x1
        
        vectors = [x_xi_0, g_s]
        return vectors

    def calculate_slip(self, t_T: float, p_N: float, mu: float) -> float:
        """Computes the slip function value to determine stick or sliding status.

        This function evaluates the classical Coulomb friction condition:
            - If f_s < 0 → sticking (no sliding)
            - If f_s = 0 → transition to sliding
            - If f_s > 0 → sliding occurs

        It is used in contact mechanics to determine whether a contact point is sticking
        or sliding under given tangential traction and normal pressure.

        Args:
            t_T (float): Tangential traction at the contact interface (shear stress).
            p_N (float): Normal contact pressure (should be compressive; use absolute).
            mu (float, optional): Coefficient of friction. Default is 0.75 (typical for concrete-concrete contact).

        Returns:
            float: Slip function value `f_s`:
                - f_s < 0 → stick
                - f_s = 0 → transition
                - f_s > 0 → slip
        """
        # fs < 0, stick; fs = 0, sliding
        # mu = 0.5-1.0 for concrete-concrete material pairing Table 5.1 p78
        f_s = np.abs(t_T) - mu * np.abs(p_N)
        return float(f_s)

    def calculate_trial_tangential_stress(self, t_T_n: np.ndarray, a_n1: np.ndarray, epsilon_T: float, a_alpha_beta: np.ndarray, delta_xi_n1: np.ndarray) -> np.ndarray:
        """Computes the trial tangential stress at the new time step (n+1).

        This function implements the trial stress formula for contact problems with friction,
        particularly in the context of sticking or sliding analysis on contact interfaces.

        Formula reference: Equation (9.97) from the source material.

        Args:
            t_T_n (np.ndarray): Previous tangential traction vector (3D).
            a_n1 (np.ndarray): Unit tangent vector at current step n+1 (3D).
            epsilon_T (float): Tangential penalty parameter.
            a_alpha_beta (np.ndarray): 2×2 in-plane metric tensor (often identity).
            delta_xi_n1 (np.ndarray): Incremental natural coordinate change (2D vector).

        Returns:
            np.ndarray: Trial tangential stress (scalar or vector depending on formulation).
        """
        # (9.97) p245
        # Project previous tangential traction along new tangent vector
        projection = np.dot(t_T_n, a_n1)
        
        # Tangential stress trial: projection + penalty * directional increment
        t_tr_T_n1 = projection + epsilon_T * np.dot(a_alpha_beta, delta_xi_n1)
        
        return t_tr_T_n1

    def calculate_tangential_stress(self, f_tr_s: float, t_tr_T_n1: np.ndarray, t_tr_T: np.ndarray, mu: float, t_N: float) -> np.ndarray:
        """Evaluates the final tangential traction after sticking/sliding condition check.

        Based on the trial slip function `f_tr_s`, this function determines whether the contact
        condition is sticking or sliding and computes the updated tangential traction accordingly.

        Reference: Equation (9.96), page 245.

        Args:
            f_tr_s (float): Trial slip function value. 
                - If ≤ 0: sticking occurs.
                - If > 0: sliding occurs.
            t_tr_T_n1 (np.ndarray): Trial tangential traction at time step n+1 (vector).
            t_tr_T (np.ndarray): Direction of the trial tangential traction at time step n+1 (vector).
            mu (float): Friction coefficient.
            t_N (float): Normal contact stress (positive in compression).

        Returns:
            np.ndarray: Updated tangential traction vector after checking stick/slip status.

        Raises:
            ValueError: If sliding occurs but trial tangential direction has zero norm (undefined).
        """
        
        if f_tr_s <= 0:
            # Sticking: use trial traction directly
            return t_tr_T_n1
        else:
            norm_t_tr_T = np.linalg.norm(t_tr_T)
            norm_t_tr_T = np.linalg.norm(t_tr_T)
            if norm_t_tr_T == 0:
                raise ValueError("Tangential traction direction is undefined (zero norm) during sliding.")
            
            # Sliding: project along direction of trial tangential traction
            t_T_updated = mu * np.abs(t_N) * (t_tr_T_n1 / norm_t_tr_T)
            return t_T_updated

    def calculate_contact_stiffness(self, slave_node_coord: np.ndarray, surface_node_coord: np.ndarray, delta_u: np.ndarray) -> Tuple[str, np.ndarray, List]:
        """Computes the contact stiffness matrix for a slave node contacting a surface defined by 4 nodes.

        This function implements contact formulation based on normal and tangential behavior using
        penalty methods and friction models as described by Simo & Laursen (1992) and Bandeira et al. (2004).

        Args:
            slave_node_coord (np.ndarray): Coordinate of the slave node (3,).
            surface_node_coord (np.ndarray): Coordinates of the master surface nodes (4, 3).
            delta_u (np.ndarray): Relative displacement increment (15,).

        Returns:
            Tuple[str, np.ndarray, List]:
                - status (str): 'stick' or 'slip'.
                - K (np.ndarray): Computed contact stiffness matrix (15x15).
                - contact_variables (List): List of basic variables used (only for stick case).
        """
        # Retrieve penalty parameters and friction coefficient from contact properties
        epsilon_N, epsilon_T = self.para.epsilon # [epsilon_N, epsilon_T] # [1e8, 1e4] # Simo and Laursen, 1992
        mu = self.para.mu # 0.75 # 0.5-1.0 for concrete-concrete material pairing in Table 5.1 p78
        
        # Compute key contact vectors and matrices from geometry
        vectors, matrices = self.calculate_vectors_and_matrices(
                slave_node_coord=slave_node_coord,
                surface_node_coord=surface_node_coord)
        basics, B_s, E, N, N_alpha, T_alpha, T_tilde_alpha, D_alpha, E_alpha = vectors
        
        # g_N is a scalar instead of a vector
        # g_s is a vector instead of a scalar
        x_bar, g_N, n_bar_c, a_bar_1, a_bar_2, xi_bar = basics
        a_bar_alpha_beta, a_alpha_beta, b_bar_alpha_beta, H_alpha_beta, T_alpha_beta, N_alpha_beta = matrices
        
        # Tangential vectors and matrices
        vectors_t = self.calculate_tangential_vectors_and_matrices(
            slave_node_coord=slave_node_coord,
            surface_node_coord=surface_node_coord,
            delta_u=delta_u, epsilon=self.para.epsilon,
        )
        x_xi_0, g_s = vectors_t
        # K_T1 (9.104) p246 or (34)
        # t_T_0 = t_T1_0*np.array([1, 0, 0]) + t_T2_0*np.array([0, 1, 0])
        t_T = epsilon_T * g_s
        t_T_norm = np.linalg.norm(t_T)
        t_N = epsilon_N * np.abs(g_N)
        
        # mu = 0.5-1.0 for concrete-concrete material pairing Table 5.1 p78
        f_s = self.calculate_slip(t_T=t_T_norm, p_N=t_N, mu=mu)
        g_s_norm = np.linalg.norm(g_s)
        
        if f_s < 0 or g_s_norm == 0: # stick
            #                 15x3 @ 3x15
            kc = epsilon_N * B_s @ B_s.T
            status = 'stick'
            contact_variables = [x_bar, B_s]
            return status, kc, contact_variables
        elif f_s >= 0: # slip
            print('\n\n\nAttention!!!!!!!')
            print('It is sliding!!!!!!!\n\n\n')
            # K_N = epsilon_N NN^T + t_N [N_alpha D^{alpha T} + a^{beta alpha}T_alpha (
            #   N_beta^T - D^{gamma T} (n_bar_c dot a_bar_{beta,gamma})
            # )] # (9.103) p246 or (33) Bandeira et al., 2004
            # t_N = epsilon_N * g_N
            # a_bar_{beta,gamma} = 0
            K_N = epsilon_N* N @ N.T + t_N * \
                    (N_alpha[0] @ D_alpha[0].T + N_alpha[1] @ D_alpha[1].T + \
                    # alpha=1,beta=1;    a^11
                    a_alpha_beta[0,0]*T_alpha[0]@N_alpha[0].T + \
                    # alpha=1,beta=2;    a^21
                    a_alpha_beta[1,0]*T_alpha[0]@N_alpha[1].T + \
                    # alpha=2,beta=1;    a^12
                    a_alpha_beta[0,1]*T_alpha[1]@N_alpha[0].T + \
                    # alpha=2,beta=2;    a^22
                    a_alpha_beta[1,1]*T_alpha[1]@N_alpha[1].T)
            
            # Iterations for TRIAL has got stated...
            # a_bar_1_{n+1} = a_bar_1_{n}; same situation for a_bar_{alpha beta}
            # a_bar_{12} = a_bar_{21} = 0, which means
            # if alpha is not equal to beta in equation (9.97) p245,
            # the corresponding term is zero. Therefore, alpha, beta = 1 when alpha is 1.
            # t_T1 = self._compute_tangential_stress(f_tr_s=f_s, t_T_n=t_T_0, a_n1=a_bar_1,
            #         # a_bar_{12} = 0, a_bar_{11} = 1
            #         epsilon_T=epsilon_T, a_alpha_beta=a_alpha_beta[0,0], # beta = 1
            #         delta_beta=delta_xi1)
            t_T1 = np.dot(t_T, a_bar_1)
            # t_T2 = self._compute_tangential_stress(f_tr_s=f_s, t_T_n=t_T_0, a_n1=a_bar_2,
            #         # a_bar_{21} = 0, a_bar_{22} = 1
            #         epsilon_T=epsilon_T, a_alpha_beta=a_alpha_beta[1,1], # beta = 2
            #         delta_beta=delta_xi2)
            t_T2 = np.dot(t_T, a_bar_2)
            
            # K_T1 = term 1 + term 2 + term 3 (9.104) p246 or (34) Bandeira et al., 2004
            # T_tilde is equal to zero because a1,1 a1,2 a2,1 a22 are zero.
            # term 1 = t_{T alpha} h_bar^{alpha eta} 
            # [T_{eta beta} + T_{beta eta} + T_tilde_{eta beta}] D^{beta} T
            #         alpha = 1, beta = 1, eta = 1
            term11 = t_T1 * H_alpha_beta[0,0] * (T_alpha_beta[0,0] + T_alpha_beta[0,0]) @ D_alpha[0].T
            #         alpha = 1, beta = 1, eta = 2
            term12 = t_T1 * H_alpha_beta[0,1] * (T_alpha_beta[1,0] + T_alpha_beta[0,1]) @ D_alpha[0].T
            #         alpha = 1, beta = 2, eta = 1
            term13 = t_T1 * H_alpha_beta[0,0] * (T_alpha_beta[0,1] + T_alpha_beta[1,0]) @ D_alpha[1].T
            #         alpha = 1, beta = 2, eta = 2
            term14 = t_T1 * H_alpha_beta[0,1] * (T_alpha_beta[1,1] + T_alpha_beta[1,1]) @ D_alpha[1].T
            #         alpha = 2, beta = 1, eta = 1
            term15 = t_T2 * H_alpha_beta[1,0] * (T_alpha_beta[0,0] + T_alpha_beta[0,0]) @ D_alpha[0].T
            #         alpha = 2, beta = 1, eta = 2
            term16 = t_T2 * H_alpha_beta[1,1] * (T_alpha_beta[1,0] + T_alpha_beta[0,1]) @ D_alpha[0].T
            #         alpha = 2, beta = 2, eta = 1
            term17 = t_T2 * H_alpha_beta[1,0] * (T_alpha_beta[0,1] + T_alpha_beta[1,0]) @ D_alpha[1].T
            #         alpha = 2, beta = 2, eta = 2
            term18 = t_T2 * H_alpha_beta[1,1] * (T_alpha_beta[1,1] + T_alpha_beta[1,1]) @ D_alpha[1].T
            term1 = term11 + term12 + term13 + term14 + term15 + term16 + term17 + term18
            # term 2 = t_(T alpha) h_bar^{alpha eta} 
            # D^{beta} [T_{neta beta} T + T_{beta eta} T + T_tilde_{eta beta} T]
            #         alpha = 1, beta = 1, eta = 1
            term21 = t_T1 * H_alpha_beta[0,0] *  D_alpha[0] @ (T_alpha_beta[0,0].T + T_alpha_beta[0,0].T)
            #         alpha = 1, beta = 1, eta = 2
            term22 = t_T1 * H_alpha_beta[0,1] *  D_alpha[0] @ (T_alpha_beta[1,0].T + T_alpha_beta[0,1].T)
            #         alpha = 1, beta = 2, eta = 1
            term23 = t_T1 * H_alpha_beta[0,0] *  D_alpha[1] @ (T_alpha_beta[0,1].T + T_alpha_beta[1,0].T)
            #         alpha = 1, beta = 2, eta = 2
            term24 = t_T1 * H_alpha_beta[0,1] *  D_alpha[1] @ (T_alpha_beta[1,1].T + T_alpha_beta[1,1].T)
            #         alpha = 2, beta = 1, eta = 1
            term25 = t_T2 * H_alpha_beta[1,0] *  D_alpha[0] @ (T_alpha_beta[0,0].T + T_alpha_beta[0,0].T)
            #         alpha = 2, beta = 1, eta = 2
            term26 = t_T2 * H_alpha_beta[1,1] *  D_alpha[0] @ (T_alpha_beta[1,0].T + T_alpha_beta[0,1].T)
            #         alpha = 2, beta = 2, eta = 1
            term27 = t_T2 * H_alpha_beta[1,0] *  D_alpha[1] @ (T_alpha_beta[0,1].T + T_alpha_beta[1,0].T)
            #         alpha = 2, beta = 2, eta = 2
            term28 = t_T2 * H_alpha_beta[1,1] *  D_alpha[1] @ (T_alpha_beta[1,1].T + T_alpha_beta[1,1].T)
            term2 = term21 + term22 + term23 + term24 + term25 + term26 + term27 + term28
            # term 3 = t_(T alpha) h_bar^{alpha eta} 
            # [-E E_{eta} T - E_{eta} T E - g_N * N_{eta beta} D^{beta} T - g_N * D^{beta} N_{eta beta} T]
            #         alpha = 1, eta = 1
            #         alpha = 1, eta = 2
            #         alpha = 2, eta = 1
            #         alpha = 3, eta = 2
            #         alpha = 1, eta = 1
            #         alpha = 1, eta = 2
            #         alpha = 2, eta = 1
            #         alpha = 3, eta = 2
            term31 = t_T1*H_alpha_beta[0,0]*(-E.T @ E_alpha[0]) + t_T1*H_alpha_beta[0,1]*(-E.T @ E_alpha[1]) + \
                    t_T2*H_alpha_beta[1,0]*(-E.T @ E_alpha[0]) + t_T2*H_alpha_beta[1,1]*(-E.T @ E_alpha[1])
            #         alpha = 1, eta = 1
            #         alpha = 1, eta = 2
            #         alpha = 2, eta = 1
            #         alpha = 3, eta = 2
            #         alpha = 1, eta = 1
            #         alpha = 1, eta = 2
            #         alpha = 2, eta = 1
            #         alpha = 3, eta = 2
            term32 = t_T1*H_alpha_beta[0,0]*(-E_alpha[0] @ E.T) + t_T1*H_alpha_beta[0,1]*(-E_alpha[1] @ E.T) + \
                    t_T2*H_alpha_beta[1,0]*(-E_alpha[0] @ E.T) + t_T2*H_alpha_beta[1,1]*(-E_alpha[1] @ E.T)
            #         alpha = 1, beta = 1, eta = 1
            term331 = t_T1*H_alpha_beta[0,0]*(-g_N * N_alpha_beta[0,0] @ D_alpha[0].T
                    -g_N * D_alpha[0] @ N_alpha_beta[0,0].T)
            #         alpha = 1, beta = 1, eta = 2
            term332 = t_T1*H_alpha_beta[0,1]*(-g_N * N_alpha_beta[1,0] @ D_alpha[0].T
                    -g_N * D_alpha[0] @ N_alpha_beta[1,0].T)
            #         alpha = 1, beta = 2, eta = 1
            term333 = t_T1*H_alpha_beta[0,0]*(-g_N * N_alpha_beta[0,1] @ D_alpha[1].T
                    -g_N * D_alpha[1] @ N_alpha_beta[0,1].T)
            #         alpha = 1, beta = 2, eta = 2
            term334 = t_T1*H_alpha_beta[0,1]*(-g_N * N_alpha_beta[1,1] @ D_alpha[1].T
                    -g_N * D_alpha[1] @ N_alpha_beta[1,1].T)
            #         alpha = 2, beta = 1, eta = 1
            term335 = t_T2*H_alpha_beta[1,0]*(-g_N * N_alpha_beta[0,0] @ D_alpha[0].T
                    -g_N * D_alpha[0] @ N_alpha_beta[0,0].T)
            #         alpha = 2, beta = 1, eta = 2
            term336 = t_T2*H_alpha_beta[1,1]*(-g_N * N_alpha_beta[1,0] @ D_alpha[0].T
                    -g_N * D_alpha[0] @ N_alpha_beta[1,0].T)
            #         alpha = 2, beta = 2, eta = 1
            term337 = t_T2*H_alpha_beta[1,0]*(-g_N * N_alpha_beta[0,1] @ D_alpha[1].T
                    -g_N * D_alpha[1] @ N_alpha_beta[0,1].T)
            #         alpha = 2, beta = 2, eta = 2
            term338 = t_T2*H_alpha_beta[1,1]*(-g_N * N_alpha_beta[1,1] @ D_alpha[1].T
                    -g_N * D_alpha[1] @ N_alpha_beta[1,1].T)
            term33 = term331 + term332 + term333 + term334 + term335 + term336 + term337 + term338
            term3 = term31 + term32 + term33
            K_T1 = term1 + term2 + term3
            
            # K_T2 (9.105) p246 or (36)
            # assuming a_bar_n+1 = a_bar_n,
            # leading to a_{alpha beta} does not change over time
            
            # delta_xi1, delta_xi2 = delta_xi
            # (44) and \lambda Algorithm (4) in section 4.3 (Wriggers et al. 1990)
            # \lambda is the delta xi in this text
            # t_T1 = t_T dot a_bar_1; t_T2 = t_T dot a_bar_2
            delta_xi1 = (mu*t_N - t_T1)/(epsilon_T*a_bar_alpha_beta[0,0])
            delta_xi2 = (mu*t_N - t_T2)/(epsilon_T*a_bar_alpha_beta[1,1])
            
            t_tr_T1_1 = self.calculate_trial_tangential_stress(t_T_n=t_T, a_n1=a_bar_1,
                    epsilon_T=epsilon_T, a_alpha_beta=a_bar_alpha_beta[0,0], delta_xi_n1=delta_xi1)
            t_tr_T2_1 = self.calculate_trial_tangential_stress(t_T_n=t_T, a_n1=a_bar_2,
                    epsilon_T=epsilon_T, a_alpha_beta=a_bar_alpha_beta[1,1], delta_xi_n1=delta_xi2)
            t_tr_T_1 = t_tr_T1_1*np.array([1, 0, 0]) + t_tr_T2_1*np.array([0, 1, 0])
            # term 1: mu * D^{alpha} * epsilon_N t^tr_T_alpha / ||t^tr_T|| N T
            # alpha = 1, 2
            term1 = mu*epsilon_N*t_tr_T1_1/np.linalg.norm(t_tr_T_1) * D_alpha[0] @ N.T + \
                    mu*epsilon_N*t_tr_T2_1/np.linalg.norm(t_tr_T_1) * D_alpha[1] @ N.T
            # term 2: mu * D^alpha * |t_N| / ||t^tr_T|| * (epsilon_T * a_{beta alpha} D^beta T)
            # alpha = 1, beta = 1
            term21 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)*epsilon_T*a_bar_alpha_beta[0,0] * \
                    D_alpha[0] @ D_alpha[0].T
            # alpha = 1, beta = 2
            term22 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)*epsilon_T*a_bar_alpha_beta[1,0] * \
                    D_alpha[0] @ D_alpha[1].T
            # alpha = 2, beta = 1
            term23 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)*epsilon_T*a_bar_alpha_beta[0,1] * \
                    D_alpha[1] @ D_alpha[0].T
            # alpha = 2, beta = 2
            term24 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)*epsilon_T*a_bar_alpha_beta[1,1] * \
                    D_alpha[1] @ D_alpha[1].T
            term2 = term21 + term22 + term23 + term24
            # term 3: mu * D^alpha * |t_N| / ||t^tr_T|| * 
            # (-epsilon_T * delta xi^beta*(T_{beta alpha} T T_{alpha beta} T))
            # alpha = 1, beta = 1
            term31 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)*(-epsilon_T)*delta_xi1 * \
                    D_alpha[0] @ (T_alpha_beta[0,0].T + T_alpha_beta[0,0].T)
            # alpha = 1, beta = 2
            term32 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)*(-epsilon_T)*delta_xi2 * \
                    D_alpha[0] @ (T_alpha_beta[1,0].T + T_alpha_beta[0,1].T)
            # alpha = 2, beta = 1
            term33 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)*(-epsilon_T)*delta_xi1 * \
                    D_alpha[1] @ (T_alpha_beta[0,1].T + T_alpha_beta[1,0].T)
            # alpha = 2, beta = 2
            term34 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)*(-epsilon_T)*delta_xi2 * \
                    D_alpha[1] @ (T_alpha_beta[1,1].T + T_alpha_beta[1,1].T)
            term3 = term31 + term32 + term33 + term34
            # term 4: mu * D^alpha * |t_N| / ||t^tr_T||^3 * t_tr_T_alpha * t_tr_T_beta * (-P_beta T)
            # alpha = 1, beta = 1
            # TODO
            term41 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)**3 * t_tr_T1_1 * t_tr_T1_1 * \
                    D_alpha[0] @ (-P_alpha[0].T)
            # alpha = 1, beta = 2
            term42 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)**3 * t_tr_T1_1 * t_tr_T2_1 * \
                    D_alpha[0] @ (-P_alpha[1].T)
            # alpha = 2, beta = 1
            term43 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)**3 * t_tr_T2_1 * t_tr_T1_1 * \
                    D_alpha[1] @ (-P_alpha[0].T)
            # alpha = 2, beta = 2
            term44 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)**3 * t_tr_T2_1 * t_tr_T2_1 * \
                    D_alpha[1] @ (-P_alpha[1].T)
            term4 = term41 + term42 + term43 + term44
            # term 5: mu * D^alpha * |t_N| / ||t^tr_T||^3 * * t_tr_T_alpha * t_tr_T_beta *
            # epsilon_T * delta xi^gamma * (T_{beta gamma} T + T_{gamma beta} T)
            # alpha = 1, beta = 1, gamma = 1
            term51 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)**3 * t_tr_T1_1 * t_tr_T1_1 * \
                    epsilon_T*delta_xi1 * D_alpha[0] @ (T_alpha_beta[0,0].T + T_alpha_beta[0,0].T)
            # alpha = 1, beta = 1, gamma = 2
            term52 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)**3 * t_tr_T1_1 * t_tr_T1_1 * \
                    epsilon_T*delta_xi2 * D_alpha[0] @ (T_alpha_beta[0,1].T + T_alpha_beta[1,0].T)
            # alpha = 1, beta = 2, gamma = 1
            term53 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)**3 * t_tr_T1_1 * t_tr_T2_1 * \
                    epsilon_T*delta_xi1 * D_alpha[0] @ (T_alpha_beta[1,0].T + T_alpha_beta[0,1].T)
            # alpha = 1, beta = 2, gamma = 2
            term54 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)**3 * t_tr_T1_1 * t_tr_T2_1 * \
                    epsilon_T*delta_xi2 * D_alpha[0] @ (T_alpha_beta[1,1].T + T_alpha_beta[1,1].T)
            # alpha = 2, beta = 1, gamma = 1
            term55 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)**3 * t_tr_T2_1 * t_tr_T1_1 * \
                    epsilon_T*delta_xi1 * D_alpha[1] @ (T_alpha_beta[0,0].T + T_alpha_beta[0,0].T)
            # alpha = 2, beta = 1, gamma = 2
            term56 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)**3 * t_tr_T2_1 * t_tr_T1_1 * \
                    epsilon_T*delta_xi2 * D_alpha[1] @ (T_alpha_beta[0,1].T + T_alpha_beta[1,0].T)
            # alpha = 2, beta = 2, gamma = 1
            term57 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)**3 * t_tr_T2_1 * t_tr_T2_1 * \
                    epsilon_T*delta_xi1 * D_alpha[1] @ (T_alpha_beta[1,0].T + T_alpha_beta[0,1].T)
            # alpha = 2, beta = 2, gamma = 2
            term58 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)**3 * t_tr_T2_1 * t_tr_T2_1 * \
                    epsilon_T*delta_xi2 * D_alpha[1] @ (T_alpha_beta[1,1].T + T_alpha_beta[1,1].T)
            term5 = term51 + term52 + term53 + term54 + term55 + term56 + term57 + term58
            # term 6: mu * D^alpha * |t_N| / ||t^tr_T||^3 * * t_tr_T_alpha * t_tr_T_beta *
            # D^theta T * (-epsilon_T) * a_theta_beta
            # alpha = 1, beta = 1, theta = 1
            term61 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)**3 * t_tr_T1_1 * t_tr_T1_1 * \
                    (-epsilon_T*a_bar_alpha_beta[0,0]) * D_alpha[0] @ D_alpha[0].T
            # alpha = 1, beta = 1, theta = 2
            term62 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)**3 * t_tr_T1_1 * t_tr_T1_1 * \
                    (-epsilon_T*a_bar_alpha_beta[1,0]) * D_alpha[0] @ D_alpha[1].T
            # alpha = 1, beta = 2, theta = 1
            term63 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)**3 * t_tr_T1_1 * t_tr_T2_1 * \
                    (-epsilon_T*a_bar_alpha_beta[0,1]) * D_alpha[0] @ D_alpha[0].T
            # alpha = 1, beta = 2, theta = 2
            term64 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)**3 * t_tr_T1_1 * t_tr_T2_1 * \
                    (-epsilon_T*a_bar_alpha_beta[1,1]) * D_alpha[0] @ D_alpha[1].T
            # alpha = 2, beta = 1, theta = 1
            term65 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)**3 * t_tr_T2_1 * t_tr_T1_1 * \
                    (-epsilon_T*a_bar_alpha_beta[0,0]) * D_alpha[1] @ D_alpha[0].T
            # alpha = 2, beta = 1, theta = 2
            term66 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)**3 * t_tr_T2_1 * t_tr_T1_1 * \
                    (-epsilon_T*a_bar_alpha_beta[1,0]) * D_alpha[1] @ D_alpha[1].T
            # alpha = 2, beta = 2, theta = 1
            term67 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)**3 * t_tr_T2_1 * t_tr_T2_1 * \
                    (-epsilon_T*a_bar_alpha_beta[0,1]) * D_alpha[1] @ D_alpha[0].T
            # alpha = 2, beta = 2, theta = 2
            term68 = mu*np.abs(t_N)/np.linalg.norm(t_tr_T_1)**3 * t_tr_T2_1 * t_tr_T2_1 * \
                    (-epsilon_T*a_bar_alpha_beta[1,1]) * D_alpha[1] @ D_alpha[1].T
            term6 = term61 + term62 + term63 + term64 + term65 + term66 + term67 + term68
            K_T2 = term1 + term2 + term3 + term4 + term5 + term6
            
            K = K_N + K_T1 + K_T2
            status = 'stick'
            return status, K

    def get_contact_node_tag_pairs(self, contacts: List) -> List[List[np.ndarray]]:
        """Generates all point-to-surface contact combinations from a list of contact surfaces.

        This function processes a list of contact surfaces and generates all possible combinations 
        of slave node tags and master surface node tags, useful for point-to-surface contact formulations.

        Args:
            contacts (List): A list of contact definitions. Each element is a list containing:
                - bottom_nodes (unused here),
                - top_nodes or slave_nodes (2D array),
                - bottom_elements or master_elements (list of elements),
                - top_elements or slave_elements (list of elements).
                It assumes contacts[i] and contacts[i+1] form one contact pair (master/slave).

        Returns:
            List[List[np.ndarray]]: A list where each element contains multiple [5] arrays:
                Each array represents a slave-master surface contact in the form:
                    [slave_node_tag, surface_node_tag_1, ..., surface_node_tag_4]
        """
        contact_pairs = []
        
        for i in range(len(contacts) - 1):
            master_surface = contacts[i]    # master contact surface
            slave_surface = contacts[i + 1] # slave contact surface
            
            # [bottom_nodes, top_nodes, bottom_elements, top_elements]
            _, master_nodes, _, master_elements = master_surface
            slave_nodes, _, slave_elements, _ = slave_surface
            
            for slave_node in slave_nodes:
                # Single node tag (assumed scalar or int)
                slave_tag = slave_node[0] # first element (node tag)

                combinations_for_slave = []
                
                for master_element in master_elements:
                    surface_node_tags = master_element[4:]  # Last 4 nodes (upper 4 node tags) define the contact face
                    contact_pair = np.insert(surface_node_tags, 0, slave_tag)  # [slave_tag, surf_nodes...]
                    combinations_for_slave.append(contact_pair)

                contact_pairs.append(combinations_for_slave)
        
        return contact_pairs

    def search_activated_contact_pairs(self, potential_contact_pairs: List[np.ndarray], node_coords: np.ndarray, margin: float = 0.05) -> List[np.ndarray]:
        """Identifies activated (actual) contact pairs from all potential point-to-surface pairs.

        For each slave node and its associated possible master surfaces, this function checks
        whether the slave lies within the natural coordinate domain of any master surface.
        If no surface strictly contains the point, a margin tolerance is used as fallback.

        Args:
            potential_contact_pairs (List[np.ndarray]): 
                A list of slave-to-surface candidate contact sets.
                Each element is a list of np.ndarrays of shape (5,), where:
                [slave_tag, surf_node_1, surf_node_2, surf_node_3, surf_node_4]
            nodecoords (np.ndarray): Nodal coordinates array of shape (num_nodes, 3).
            margin (float, optional): 
                Margin tolerance beyond the [-1, 1] natural coordinate bounds (default: 0.05).

        Returns:
            List[np.ndarray]: 
                A list of activated contact pairs, where each pair is a 1D NumPy array of 5 node tags.
        """
        activated_pairs = []
        
        for slave_group in potential_contact_pairs:
            slave_tag = slave_group[0][0]
            surface_hits = 0  # Count of surfaces where this slave appears active

            for pair in slave_group:
                slave_coord = node_coords[pair[0]]
                surface_coords = node_coords[pair[1:]]

                xi, _, _, _ = self.calculate_natural_coordinates(slave_coord, surface_coords)

                # Check if slave projects within surface in natural coordinates
                if np.all(np.abs(xi) <= 1):
                    activated_pairs.append(pair)
                    surface_hits += 1

            # If no hit found, retry with margin
            if surface_hits == 0:
                self.logger.debug(
                    "[Contact] No strict contact for slave node %d. Retrying with margin = %.3f.", slave_tag, margin)
                for pair in slave_group:
                    slave_coord = node_coords[pair[0]]
                    surface_coords = node_coords[pair[1:]]

                    xi, _, _, _ = self.calculate_natural_coordinates(slave_coord, surface_coords)

                    if np.all(np.abs(xi) <= 1 + margin):
                        activated_pairs.append(pair)
                        surface_hits += 1

            # Warn if ambiguous (multiple matches) or none at all
            if surface_hits > 1:
                self.logger.debug(
                    "[Contact] Ambiguous activation: slave node %d matched with %d surfaces.", slave_tag, surface_hits)
            elif surface_hits == 0:
                self.logger.debug(
                    "[Contact] No activation found for slave node %d, even after margin extension.", slave_tag)
        
        return activated_pairs
    
    def formulate_contact(self, contacts: List, node_coords: np.ndarray, delta_u: np.ndarray, current_iter: int) -> Tuple[Optional[str], np.ndarray]:
        """Formulates and assembles the global contact stiffness matrix for active contact interfaces.

        This method identifies potential contact pairs, activates the valid contacts based on geometry,
        computes the local contact stiffness for each valid pair, and assembles them into the global
        contact stiffness matrix `global_kc`.

        Args:
            contacts (List): A list of contact definitions. Each element is a list containing:
                - bottom_nodes (unused here),
                - top_nodes or slave_nodes (2D array),
                - bottom_elements or master_elements (list of elements),
                - top_elements or slave_elements (list of elements).
                It assumes contacts[i] and contacts[i+1] form one contact pair (master/slave).
            node_coords (np.ndarray): Global nodal coordinates of shape (n_nodes, 3).
            delta_u (np.ndarray): Global displacement vector (flattened or structured per DOF).
            current_iter (int, optional): Current nonlinear iteration index (default: 0).
                Used to control updating of initial contact points.

        Returns:
            Tuple[Optional[str], np.ndarray]:
                - status (str or None): Contact status ('stick', 'slip') for the last processed pair,
                or None if no contacts are active.
                - global_kc (np.ndarray): Assembled global contact stiffness matrix of shape (ndof_total, ndof_total).
        """
        n_nodes, ndof = node_coords.shape
        total_dof = n_nodes * ndof
        global_kc = np.zeros((total_dof, total_dof))
        
        # Step 1: Get and activate contact pairs
        potential_contact_pairs = self.get_contact_node_tag_pairs(contacts=contacts)
        activated_contact_pairs = self.search_activated_contact_pairs(
                potential_contact_pairs=potential_contact_pairs,
                node_coords=node_coords)
        
        if len(activated_contact_pairs) == 0:
            return None, global_kc
        
        # Initialize storage
        x_0_bars = []
        Bstick = []
        contact_status = None
        
        # Step 2: Loop over activated contact pairs
        for pair in activated_contact_pairs:
            slave_node_coord = node_coords[pair[0]]
            surface_node_coord = node_coords[pair[1:]]

            # Retrieve displacement vector for contact nodes
            global_index = local_to_global_index(pair, dof=ndof)
            delta_u_contact = delta_u[global_index]

            # Compute contact stiffness
            contact_status, kc, contact_vars = self.calculate_contact_stiffness(
                slave_node_coord=slave_node_coord,
                surface_node_coord=surface_node_coord,
                delta_u=delta_u_contact
            )
            
            # Assemble for sticking contact
            if contact_status == 'stick':
                x_bar, B_s = contact_vars # B_s (15x3)
                Bstick.append(B_s)
                if current_iter == 0:
                    x_0_bars.append(x_bar)

            # Assemble into global stiffness matrix
            for (i, j), k_val in np.ndenumerate(kc):
                global_kc[global_index[i], global_index[j]] += k_val
        
        # Save contact data for reuse
        if current_iter == 0:
            self.x_0_bars = x_0_bars
        
        self.Bstick = Bstick
        self.activated_contact_pairs = activated_contact_pairs
        
        return contact_status, global_kc
