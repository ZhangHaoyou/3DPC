# fem/fem/boundary_conditions.py

import numpy as np

from typing import Tuple


from fem.fem.geo_mesh import Mesh
from fem.utils.utils import local_to_global_index
from fem.utils.log_config import setup_logger


class Boundary_Conditions:
    """Applies boundary conditions to a finite element system.
    
    This class manages displacement constraints and loading conditions,
    specifically:
        - Fixing displacements at base nodes (supports).
        - Applying partial constraints and external loads at top surface nodes.
    """
    def __init__(self, mesh: Mesh, move_vertically: bool = False) -> None:
        """Initializes the boundary condition manager.
        
        Args:
            mesh (Mesh): Mesh object containing node tagging information.
            move_vertically (bool, optional): 
                If True, x and y displacements are fixed at loading nodes 
                (allowing vertical movement only). Defaults to False.
        """
        self.mesh = mesh
        self.base_node_tags = mesh.base_node_tags
        self.loading_node_tags = mesh.loading_node_tags
        self.move_vertically = move_vertically
        
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info(
            "Initialized Boundary_Conditions | base_node_tags=%d | loading_node_tags=%d | move_vertically=%s",
            len(self.base_node_tags),
            len(self.loading_node_tags),
            self.move_vertically
        )
    
    def apply_supports(self, global_k: np.ndarray, global_f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies full displacement constraints (ux, uy, uz) on base nodes.
        
        This enforces zero displacement at base node degrees of freedom by modifying
        the global stiffness matrix and force vector (Dirichlet boundary conditions).
        
        Args:
            global_k (np.ndarray): Global stiffness matrix to be modified in-place.
            global_f (np.ndarray): Global force vector to be modified in-place.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Modified global stiffness matrix and force vector.
        """
        ndof = self.mesh.geo.para.ndof
        global_index = local_to_global_index(self.base_node_tags, dof=ndof)
        
        start_tag = self.base_node_tags[0]
        end_tag = self.base_node_tags[-1]
        total_tags = len(self.base_node_tags)
        
        msg = (f"Applying support constraints (u = 0) on base nodes:\n"
            f"\tStart node tag: {start_tag}\n"
            f"\tEnd node tag: {end_tag}\n"
            f"\tTotal nodes: {total_tags}")
        print('🔒 ', msg)
        self.logger.debug(msg)
        
        # Apply Dirichlet boundary condition: u = 0
        global_k[global_index, :] = 0
        global_k[:, global_index] = 0
        global_k[global_index, global_index] = 1
        global_f[global_index] = 0
        
        return global_k, global_f
    
    def pressure_to_nodal_forces(self, pressure: float) -> np.ndarray:
        """Convert uniform surface pressure to equivalent nodal forces.
        
        This method distributes a given pressure applied on a rectangular surface
        into nodal forces, accounting for each node's position (corner, edge, interior).
        The resulting nodal force values are returned and debug-logged with coordinates.
        
        Args:
            pressure (float): Surface pressure in Mega-Pascal (e.g., -1 MPa).
            
        Returns:
            np.ndarray: Nodal force array of shape (n_nodes_on_surface, ), in Newtons.
    """
        node_tags = self.mesh.loading_node_tags                     # Surface node indices
        coords = self.mesh.nodes[node_tags, :]                      # mm-scale for logging
        dx, dy = self.mesh.para.mesh_size[:2]                       # Element width/height (mm)
        area_per_cell = dx * dy                                     # Area of one rectangle (mm²)
        
        # Compute number of nodes in x and y direction on the surface
        n_nodes_x = self.mesh.n_nodes_x
        n_nodes_y = self.mesh.n_nodes_y
        nodal_forces = np.zeros_like(node_tags, dtype=float)
        
        self.logger.debug("Distributing pressure = %.2f MPa over surface mesh...", pressure)
        
        # Iterate through grid and assign nodal force weights
        for i in range(n_nodes_y):
            for j in range(n_nodes_x):
                idx = i * n_nodes_x + j
                tag = node_tags[idx]
                
                # Determine weighting based on position
                if (i == 0 or i == n_nodes_y - 1) and (j == 0 or j == n_nodes_x - 1):
                    weight = 0.25  # Corner node
                elif i == 0 or i == n_nodes_y - 1 or j == 0 or j == n_nodes_x - 1:
                    weight = 0.5   # Edge node
                else:
                    weight = 1.0   # Interior node
                
                nodal_force = pressure * area_per_cell * weight
                nodal_forces[idx] = nodal_force
                
                # coord = coords[idx]
                # self.logger.debug(
                #     "Node ID: %d | Coord (mm): [%.1f, %.1f, %.1f] | Nodal force (N): %.2f",
                #     tag, coord[0], coord[1], coord[2], nodal_force
                # )
        
        return nodal_forces
    
    def apply_nodal_loading(self, pressure: float, global_k: np.ndarray, global_f: np.ndarray, direction: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Applies external load to top nodes, with optional xy constraints.
        
        Args:
            pressure (float): Surface pressure in Mega-Pascal (e.g., -1 MPa).
            global_k (np.ndarray): Global stiffness matrix (modified in-place).
            global_f (np.ndarray): Global force vector (modified in-place).
            direction (int, optional): Direction to apply the load: 0 = x, 1 = y, 2 = z. Default is 2 (vertical).
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Modified global stiffness matrix and force vector.
        """
        ndof = self.mesh.geo.para.ndof
        global_index = local_to_global_index(self.loading_node_tags, dof=ndof)
        
        area = self.mesh.geo.para.lx * self.mesh.geo.para.ly  # Cross-section area (mm²)
        loading_force = pressure * area               # Total load (N)
        
        nodal_forces = self.pressure_to_nodal_forces(pressure=pressure).reshape(-1, 1)
        # nodal_forces = loading_force / self.mesh.n_nodes_a_slice
        
        # Apply load in the specified direction
        global_f[global_index[direction::ndof]] += nodal_forces
        
        # Log detailed information of loading nodes based on final global_f
        self.logger.debug("Logging final applied nodal forces from global_f...")
        
        node_tags = self.loading_node_tags
        node_coords = self.mesh.nodes[node_tags]  # mm scale
        ndof = self.mesh.geo.para.ndof
        
        for idx, tag in enumerate(node_tags):
            coord = node_coords[idx]
            gidx = tag * ndof + direction  # global index for the specified direction
            applied_force = global_f[gidx]  # Force actually applied to global_f
            
            self.logger.debug(
                "Node ID: %d | Coord (mm): [%.1f, %.1f, %.1f] | Applied force (N): %.2f",
                tag, coord[0], coord[1], coord[2], applied_force
            )
        
        if self.move_vertically:
            start_tag = self.loading_node_tags[0]
            end_tag = self.loading_node_tags[-1]
            total_tags = len(self.loading_node_tags)
        
            msg = (
                "Fixing x and y displacements at loading nodes:\n"
                f"\tStart node tag: {start_tag}\n"
                f"\tEnd node tag: {end_tag}\n"
                f"\tTotal nodes: {total_tags}"
            )
            print('🧷 ', msg)
            self.logger.debug(msg)
            
            # Fix x-direction (ux = 0)
            for i in global_index[0::ndof]:
                global_k[i, :] = 0
                global_k[:, i] = 0
                global_k[i, i] = 1
                global_f[i] = 0
            
            # Fix y-direction (uy = 0)
            for i in global_index[1::ndof]:
                global_k[i, :] = 0
                global_k[:, i] = 0
                global_k[i, i] = 1
                global_f[i] = 0
        
        return global_k, global_f
