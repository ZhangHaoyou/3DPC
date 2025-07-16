# fem/solvers.py

import scipy
import numpy as np

from typing import Optional, Tuple

from pc_fem.fem.geo_mesh import Mesh
from pc_fem.fem.elements import Contact_Element
from pc_fem.fem.boundary_conditions import Boundary_Conditions

from pc_fem.utils.utils import local_to_global_index
from pc_fem.utils.log_config import setup_logger


class Solver:
    """Solver class for linear and nonlinear system solution in finite element analysis (FEA).

    This class provides a base interface for implementing solvers that handle global
    system equations derived from finite element formulations. It supports customizable
    logging and is intended to be extended with specific linear or nonlinear solution strategies.
    """
    
    def __init__(self) -> None:
        """Initializes the Solver instance.

        Sets up a logger specific to the Solver class for tracking solver operations.
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info("Solver initialized and ready for execution.")
        self.coverage = True
    
    def linear_solver(self, k: np.ndarray, f: np.ndarray) -> Optional[np.ndarray]:
        """Solves the linear system Ku = f using a direct solver.

        This method attempts to solve a linear system using SciPy's direct solver.
        If the matrix `k` is singular or ill-conditioned, the method logs the error
        and returns `None` instead of halting execution.

        Args:
            k (np.ndarray): Global stiffness matrix of shape (n, n).
            f (np.ndarray): Global force vector of shape (n,).

        Returns:
            Optional[np.ndarray]: Solution vector `u` of shape (n,). Returns `None`
            if the system cannot be solved due to numerical issues.

        Raises:
            None: The method handles `np.linalg.LinAlgError` internally.
        """
        try:
            u = scipy.linalg.solve(k, f)
            
            return u
        
        except np.linalg.LinAlgError as e:
            msg = f"[Solver Error] Linear system solution failed: {str(e)}"
            
            self.coverage = False
            
            print('❌ ', msg)
            self.logger.error(msg, exc_info=True)
            
            return None
    
    def log_maximum_stiffness(self, k: np.ndarray, nodes: np.ndarray, ndof: int = 3) -> None:
        """Logs the maximum stiffness value and the corresponding node coordinates.

        This method searches the global stiffness matrix `k` for the entry with the
        highest absolute value, maps the corresponding matrix indices back to the 
        relevant nodes and local degrees of freedom (DOFs), and logs this information
        for diagnostic or debugging purposes.

        Args:
            k (np.ndarray): Global stiffness matrix of shape (ndof * n_nodes, ndof * n_nodes).
            nodes (np.ndarray): Nodal coordinates array of shape (n_nodes, 3).
            ndof (int, optional): Number of DOFs per node. Defaults to 3.

        Returns:
            None
        """
        # Find max absolute value index
        i, j = np.unravel_index(np.argmax(np.abs(k)), k.shape)
        max_k_val = k[i, j]

        # Map matrix indices back to node numbers and local DOFs
        node_i, dof_i = divmod(i, ndof)
        node_j, dof_j = divmod(j, ndof)

        # Print detailed result
        msg =  "Maximum stiffness matrix entry located in `log_maximum_stiffness()`:\n"
        msg += f"  Value       : {max_k_val:.6e}\n"
        msg += f"  Matrix Entry: k[{i}, {j}]\n"
        msg += f"  Mapped DOFs : Node {node_i} DOF {dof_i} ↔ Node {node_j} DOF {dof_j}\n"
        msg += f"  Node {node_i} Coord: {np.array2string(nodes[node_i], precision=6)}\n"
        msg += f"  Node {node_j} Coord: {np.array2string(nodes[node_j], precision=6)}"
        

        print('\n📌 ', msg)
        self.logger.debug(msg)
    
    def newton_raphson(self, pressure: float, global_k: np.ndarray, global_f: np.ndarray,
            contact: Contact_Element, mesh: Mesh,bc: Boundary_Conditions,
            tol: float = 1e-7, max_iteration: int = 50, load_direction: int = 2) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Solves a nonlinear contact problem using the Newton-Raphson method.

        This method iteratively solves for nodal displacements in a contact problem using
        a penalty-based approach under geometric nonlinearity. Only 'stick' contact is
        currently handled explicitly.

        Args:
            pressure (float): Surface pressure in Mega-Pascal (e.g., -1 MPa).
            global_k (np.ndarray): Initial global stiffness matrix (nDOF x nDOF).
            global_f (np.ndarray): Initial global force vector (nDOF x 1).
            contact (Contact_Element): Contact formulation object for computing contact stiffness.
            mesh (Mesh): Mesh object containing node and element data.
            bc (Boundary_Conditions): Object handling boundary constraints and loads.
            tol (float): Convergence tolerance based on displacement increment (default: 1e-5).
            max_iteration (int): Maximum number of iterations allowed (default: 50).
            load_direction (int): Direction to apply loading force (0: x, 1: y, 2: z).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - u (np.ndarray): Final converged displacement vector (nDOF x 1).
                - global_k (np.ndarray): Updated global stiffness matrix with contact effects.
        """
        # --- Initialization ---
        initial_coords = mesh.nodes
        n_nodes, ndof = initial_coords.shape
        total_dof = n_nodes * ndof
        
        contact_node_tags = mesh.contact_node_tags
        
        # iteration parameters:
        u = np.zeros((total_dof, 1))         # Total displacement vector
        delta_u = np.zeros((total_dof, 1))   # Incremental displacement
        G_force = np.zeros((total_dof, 1))   # Contact restoring force
        contact_status = None
        current_iteration = 0
        
        while True:
            # --- Update nodal coordinates ---
            node_coords = initial_coords + u.reshape((initial_coords.shape))
            
            # --- Compute internal contact residual force if contact exists ---
            if contact_status == 'stick':
                for pair in contact.activated_contact_pairs:
                    slave_coord = node_coords[pair[0]]
                    surface_coord = node_coords[pair[1:]]
                    
                    # x_bar: projection point on the surface
                    _, x_bar = contact._point_projection_to_plane(slave_coord, surface_coord)
                    g_stick = slave_coord - x_bar # vector
                    
                    vectors, matrices = contact.calculate_vectors_and_matrices(
                            slave_node_coord=slave_coord,
                            surface_node_coord=surface_coord)
                    # basics, B_s, E, N, N_alpha, T_alpha, T_tilde_alpha, D_alpha, E_alpha
                    B_s = vectors[1] # 15x3
                    
                    G_force_stick = B_s @ g_stick.reshape((3,1)) # 15x3 @ 3x1 = 15x1
                    global_index = local_to_global_index(pair, dof=ndof)
                    for (i, j), val in np.ndenumerate(G_force_stick):
                        G_force[global_index[i], j] += val
                
            elif contact_status == 'sliding':
                print('Sliding....')
                print("⚠️ Sliding behavior not implemented.")
            
            # --- Apply external loading and contact stiffness (first iteration only) ---
            if current_iteration == 0:
                global_kbc, global_f = bc.apply_nodal_loading(
                        pressure=pressure,
                        global_k=global_k,
                        global_f=global_f,
                        direction=load_direction) # dir: 0 for x, 1 for y, 2 for z
                
                contact_status, global_kc = contact.formulate_contact(
                    contacts=mesh.contacts,
                    node_coords=node_coords,
                    delta_u=delta_u,
                    current_iter=current_iteration)
                
                # np.savetxt('log/kc.csv', global_kc, delimiter=',')
                global_k = global_kbc + global_kc
                u1 = self.linear_solver(global_k, global_f)
                
                if u1 is None:
                    self.logger.error("[Newton-Raphson] Linear solve failed at initial step. Aborting Newton-Raphson.")
                    return u, global_k
            
                delta_u = u1
                u += u1
                
                if contact_status is None:
                    return u, global_k # No contact activated
            
            # --- Newton-Raphson update (subsequent iterations) ---
            else:
                contact_idx = local_to_global_index(contact_node_tags, dof=ndof)
                # contact_residual_force = global_f - G_force
                # contact_residual_force = contact_residual_force[contact_idx]
                contact_residual_force = G_force[contact_idx]
                
                contact_status, global_kc = contact.formulate_contact(
                    contacts=mesh.contacts,
                    node_coords=node_coords,
                    delta_u=delta_u,
                    current_iter=current_iteration)
                
                global_k = global_kbc + global_kc
                kc = global_k[np.ix_(contact_idx, contact_idx)]
                
                # print(f'\nI am in Newton iteration {current_iteration}')
                # self._print_maximum_stiffness(k=kc, nodes=node_coords)
                
                u1 = self.linear_solver(kc, contact_residual_force)
                
                if u1 is None:
                    self.logger.error("[Newton-Raphson] Linear solve failed at iteration %d. Aborting.", current_iteration)
                    return u1, global_k
                
                delta_u_contact = np.zeros((total_dof, 1))
                for (i, j), val in np.ndenumerate(u1):
                    delta_u_contact[contact_idx[i], j] = val
                
                u += delta_u_contact
            
            current_iteration += 1
            max_delta_u = np.max(np.abs(u1))
            
            self.logger.debug("[Newton-Raphson] Iteration %d | max delta_u = %.3e", current_iteration, max_delta_u)
            
            # --- Check convergence ---
            if max_delta_u <= tol:
                self.logger.debug("[Newton-Raphson] Converged at iteration %d | max delta_u = %.3e", current_iteration, max_delta_u)
                return u, global_k
            
            if current_iteration == max_iteration:
                self.logger.error("[Newton-Raphson] Newton-Raphson did not converge within %d iterations.", max_iteration)
                return u, global_k

