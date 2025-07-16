# fem/geo_mesh.py

import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List


from pc_fem.parameters import Geometry_Parameter, Mesh_Parameter
from pc_fem.utils.log_config import setup_logger
from pc_fem.utils.utils import save_to_vtu


class Geometry:
    """Base class for defining 3D geometries in 3D printed concrete simulations.

    This class stores mesh-related geometric parameters, including domain dimensions,
    and is intended to support finite element mesh generation.

    Attributes:
        para (Geometry_Parameter): Geometric parameters, including domain
            dimensions (lx, ly, lz), number of printed layers, and degrees of freedom.
        logger (Logger): Logger for recording model initialization and operations.
    """
    
    def __init__(self, para: Geometry_Parameter) -> None:
        """Initializes the Geometry object with mesh size and geometric dimensions.

        Args:
            para (Geometry_Parameter): Dataclass containing domain dimensions,
                including lx, ly, lz.
        """
        self.para = para
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info(
            "Initialized Geometry model with parameters: %s",
            vars(self.para)
        )
    
    def print_parameters(self) -> None:
        """Prints the geometric parameters in a formatted, readable layout.

        Iterates through all fields in the `para` dataclass and prints
        each parameter name and value, aligned and sorted for readability.
        """
        print(f"\nGeometry Parameters ({self.__class__.__name__}):")
        for key, value in vars(self.para).items():
            print(f"{key:<20}: {value}")
    
    @staticmethod
    def plot_elements_matplotlib(node_coords: np.ndarray, ele_connectivity: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
        """Visualizes the 3D mesh including nodes and elements with numbering.

        This function plots the mesh grid using Matplotlib 3D visualization, 
        displaying nodes as blue points and elements as black wireframes. 
        Node IDs and element IDs are also labeled.

        Args:
            node_coords (np.ndarray): Array of node coordinates with shape (num_nodes, 3),
                                        where each row is [x, y, z].
            ele_connectivity (np.ndarray): Array of element connectivity with shape (num_elements, 8),
                                            where each row contains 8 node indices defining an element.

        Returns:
            tuple: (fig, ax)
                - fig (matplotlib.figure.Figure): The generated Matplotlib figure.
                - ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axis object containing the plot.

        Raises:
            ValueError: If `node_coords` is not a (N, 3) array.
            ValueError: If `ele_connectivity` is not a (M, 8) array.
        """
        # Validate input shapes
        if node_coords.ndim != 2 or node_coords.shape[1] != 3:
            raise ValueError("node_coordinates must be a 2D array with shape (num_nodes, 3).")

        if ele_connectivity.ndim != 2 or ele_connectivity.shape[1] != 8:
            raise ValueError("element_connectivity must be a 2D array with shape (num_elements, 8).")

        # Create figure and 3D axis
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot nodes
        for node_tag, coord in enumerate(node_coords):
            ax.scatter(*coord, color='blue', s=30)  # Plot node as a blue dot
            ax.text(coord[0] + 0.2, coord[1] + 0.2, coord[2], f'{node_tag+1}', color='red', fontsize=8)  

        # Define element edges (connectivity for a hexahedral element)
        element_edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face edges
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top face edges
                [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        
        # Plot elements
        for ele_tag, connectivity in enumerate(ele_connectivity):
            node_positions = node_coords[connectivity]  # Get 8-node positions

            # Draw element edges
            for edge in element_edges:
                start, end = node_positions[edge[0]], node_positions[edge[1]]
                ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='black')

            # Annotate element ID at centroid
            center = np.mean(node_positions, axis=0)  
            ax.text(center[0], center[1], center[2], f'{ele_tag+1}', color='green', fontsize=10)


        # Set axis labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("3D Mesh Visualization")
        
        # ax.invert_xaxis()
        ax.set_aspect('equal')
        
        return fig, ax

class Mesh:
    """Structured mesh generator for 3D-printed concrete layers.

    This class generates a structured finite element mesh for a printed concrete layer,
    using geometric information and mesh parameters.

    Attributes:
        geo (Geometry): Instance of the geometry containing physical dimensions and mesh size.
        para (Mesh_Parameter): Mesh configuration parameters (e.g., layer thickness, refinement).
        logger (Logger): Logger instance for recording meshing details.
    """
    
    def __init__(self, geo: Geometry, para: Mesh_Parameter) -> None:
        """Initializes the mesh with geometry and meshing configuration.

        Args:
            geo (Geometry): Geometry object containing domain dimensions and mesh size.
            para (Mesh_Parameter): Parameters controlling mesh density and meshing behavior.
        """
        self.geo = geo
        self.para = para
        
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info(
            "Initialized Geometry model with parameters: %s",
            vars(self.para)
        )
    
    def print_parameters(self) -> None:
        """Prints the geometry and mesh parameters in a formatted, readable layout.

        Iterates through all fields in the `para` dataclass and prints
        each parameter name and value, aligned and sorted for readability.
        """
        print(f"\nGeometry and Mesh Parameters ({self.__class__.__name__}):")
        for key, value in vars(self.geo.para).items():
            print(f"{key:<20}: {value}")
        for key, value in vars(self.para).items():
            print(f"{key:<20}: {value}")
    
    def mesh_a_cubic_layer(self, node_tag: int, ele_tag: int, origin: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
        """Generates a structured mesh for a cubic layer of 3D-printed concrete.

        This method creates nodal coordinates and hexahedral element connectivity 
        for a single printed layer. Each element contains 8 nodes, and the mesh 
        follows a regular structured grid based on input dimensions and mesh size.

        Args:
            node_tag (int): Starting tag (ID) for nodes.
            ele_tag (int): Starting tag (ID) for elements.
            origin (np.ndarray): Origin of the cubic layer in [x, y, z].

        Returns:
            Tuple:
                nodes (np.ndarray): Array of node coordinates with shape (n_nodes, 3).
                elements (np.ndarray): Array of element connectivity (8 node indices per element).
                bottom_top_nodes (list): List containing bottom/top node and element data:
                    - bottom_nodes (np.ndarray): Node tags and coordinates on bottom surface.
                    - top_nodes (np.ndarray): Coordinates on top surface (without tags).
                    - bottom_elements (np.ndarray): Elements on the bottom layer.
                    - top_elements (np.ndarray): Elements on the top layer.
                bottom_top_node_tags (np.ndarray): Concatenated node tags for bottom and top layers.
        """
        # Extract domain dimensions
        lx, ly, lz = self.geo.para.lx, self.geo.para.ly, self.geo.para.lz
        
        # Mesh element sizes
        mx, my, mz = self.para.mesh_size  # Element sizes in x, y, z directions

        # Compute number of elements along each axis
        nx, ny, nz = int(lx / mx), int(ly / my), int(lz / mz)
        
        # Define grid points along each axis
        x_nodes = np.linspace(0, lx, num=nx+1)  # Ensure correct boundary placement
        y_nodes = np.linspace(0, ly, num=ny+1)
        z_nodes = np.linspace(0, lz, num=nz+1)
        y, z, x = np.meshgrid(y_nodes, z_nodes, x_nodes)
        
        # Generate node coordinates
        xyz = np.vstack((x.ravel(), y.ravel(), z.ravel())).T + origin
        node_tags = np.arange(node_tag, node_tag + xyz.shape[0]) # Create node tags
        # nodes = np.hstack((node_tags.reshape(-1, 1), xyz))  # Combine IDs with coordinates
        nodes = xyz  # Only coordinates are returned; tags are kept separately
        
        # Create element connectivity (8-node hexahedral elements)
        #    8---------7
        #   /|        /|
        #  / |       / |
        # 5---------6  |
        # |  |      |  |
        # |  4------|--3
        # | /       | /
        # |/        |/
        # 1---------2
        elements = []
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    # Compute indices of 8 nodes per element
                    n4 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i
                    n1 = n4 + 1
                    n2 = n1 + (nx + 1)
                    n3 = n4 + (nx + 1)
                    n8 = n4 + (nx + 1) * (ny + 1)
                    n5 = n1 + (nx + 1) * (ny + 1)
                    n6 = n2 + (nx + 1) * (ny + 1)
                    n7 = n3 + (nx + 1) * (ny + 1)
                    
                    # Append element connectivity
                    elements.append([ # element_tag,
                        node_tags[n1], node_tags[n2], node_tags[n3], node_tags[n4],
                        node_tags[n5], node_tags[n6], node_tags[n7], node_tags[n8]
                    ])
                    ele_tag += 1
        
        elements = np.array(elements).astype(int)
        
        # Extract bottom and top node coordinates and tags
        bottom_nodes = nodes[:(nx + 1) * (ny + 1), :]
        bottom_node_tags = node_tags[:(nx + 1) * (ny + 1)]
        
        # Add tags for slave contact formulation
        bottom_nodes = np.hstack((bottom_node_tags.reshape(-1,1), bottom_nodes))
        
        top_nodes = nodes[-(nx + 1) * (ny + 1):, :]
        top_node_tags = node_tags[-(nx + 1) * (ny + 1):]
        
        # Extract elements from the bottom and top layers
        bottom_elements = elements[:nx * ny, :]
        top_elements = elements[-nx * ny:, :]
        
        bottom_top_nodes: List[np.ndarray] = [bottom_nodes, top_nodes, bottom_elements, top_elements]
        bottom_top_node_tags = np.concatenate((bottom_node_tags, top_node_tags))
        
        return nodes, elements, bottom_top_nodes, bottom_top_node_tags
    
    def mesh_layers(self, gap_between_layers: float, plot_geometry: bool = False) -> Tuple[np.ndarray, np.ndarray, List[List[np.ndarray]]]:
        """Generates a structured mesh for multiple 3D-printed concrete layers.

        This method divides the printing area into multiple layers and generates
        node coordinates and hexahedral element connectivity. It also handles
        contact interface nodes between printed layers, optionally visualizes
        the geometry, and stores key structural tags for analysis.

        Args:
            gap_between_layers (float): Vertical gap between layers in mm.
            plot_geometry (bool, optional): If True, plots and saves the geometry figure.
                Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray, List[List[np.ndarray]]]:
                - nodes: (num_nodes, 3) array of node coordinates.
                - elements: (num_elements, 8) array of element node connectivity.
                - concrete_contacts: List containing [bottom_nodes, top_nodes, bottom_elements, top_elements] for each layer.
        """
        # Extract domain dimensions
        lx, ly, lz = self.geo.para.lx, self.geo.para.ly, self.geo.para.lz
        
        # Mesh element sizes
        mx, my, mz = self.para.mesh_size  # Element sizes in x, y, z directions
        
        # Initialize counters and storage
        node_tag = 0 # Node tag, ID or index starts from 0 rather than 1
        element_tag = 0
        nodes = np.empty((0, 3))
        elements = np.empty((0, 8), dtype=int)
        elements_concrete = np.empty((0, 8), dtype=int)
        
        # Contact-related storage
        concrete_contacts: List[List[np.ndarray]] = []
        concrete_contact_node_tags = np.empty((0,), dtype=int)
        
        # Iterate through each printed layer
        for i in range(self.geo.para.n_layers):
            origin = np.array([0, 0, i * lz + gap_between_layers * (i + 1)])
            layer_nodes, layer_elements, bottom_top, contact_node_tags = self.mesh_a_cubic_layer(
                    node_tag=node_tag, ele_tag=element_tag, origin=origin)
            
            # Append to global storage
            nodes = np.vstack((nodes, layer_nodes))
            elements = np.vstack((elements, layer_elements))
            elements_concrete = np.vstack((elements_concrete, layer_elements))
            concrete_contacts.append(bottom_top)
            concrete_contact_node_tags = np.concatenate((concrete_contact_node_tags, contact_node_tags))
            
            # Logging
            print(f'\nConcrete layer {i+1} has been modeled...')
            print(f'\tNode tag from {node_tag} to {node_tag + layer_nodes.shape[0] - 1}')
            print(f'\tElement tag from {element_tag} to {element_tag + layer_elements.shape[0] - 1}')
            print(f'\tlx: {lx}, ly: {ly}, lz: {lz}, mesh size: {self.para.mesh_size} (mm)')
            print(f'\tnx: {lx/mx}, ny: {ly/my}, nz: {lz/mz}')
            
            self.logger.info(f"Concrete layer {i+1} has been modeled...")
            self.logger.info(f"  Node tag: {node_tag} to {node_tag + layer_nodes.shape[0] - 1}")
            self.logger.info(f"  Element tag: {element_tag} to {element_tag + layer_elements.shape[0] - 1}")
            self.logger.info(f"  lx: {lx}, ly: {ly}, lz: {lz}, mesh size: {self.para.mesh_size} (mm)")
            self.logger.info(f"  nx: {lx/mx:.0f}, ny: {ly/my:.0f}, nz: {lz/mz:.0f}")
            
            # Update tags for next layer
            node_tag += layer_nodes.shape[0]
            element_tag += layer_elements.shape[0]
        
        # Save nodes and elements to CSV
        np.savetxt('log/geo_mesh/nodes.csv', nodes, fmt='%.2f', delimiter=',', header='x, y, z')
        np.savetxt('log/geo_mesh/elements.csv', elements, fmt='%d', delimiter=',', header='0, 1, 2, 3, 4, 5, 6, 7 (node tags)')
        
        # save to vtu for visualization of ParaView
        save_to_vtu(filename='log/paraview/geometry.vtu', nodes=nodes, elements=elements, scalar_name='Geometry')

        # Optional: Generate geometry plot
        if plot_geometry:
            print('Generating geometry plot...')
            fig, ax = self.geo.plot_elements_matplotlib(node_coords=nodes, ele_connectivity=elements)
            fig.savefig('log/geo_mesh/geometry.png', dpi=300)
            plt.show()
        
        # Compute node counts in each direction
        n_nodes_x = int(lx / mx + 1)
        n_nodes_y = int(ly / my + 1)
        n_nodes_z = int(lz / mz + 1)
        n_nodes_a_slice = n_nodes_x * n_nodes_y
        
        # Store computed values in the object
        self.n_nodes_x = n_nodes_x
        self.n_nodes_y = n_nodes_y
        self.n_nodes_z = n_nodes_z
        self.n_nodes_a_slice = n_nodes_a_slice
        
        # Compute top and base nodes
        n_nodes = nodes.shape[0]
        top_slice_starting_node_tag = n_nodes - n_nodes_a_slice
        top_center_node = top_slice_starting_node_tag - 1 + n_nodes_a_slice // 2
        top_nodes = np.arange(top_slice_starting_node_tag, n_nodes)
        base_nodes = np.arange(0,n_nodes_a_slice)
        
        # Store node-related data
        self.top_center_node_tag = int(top_center_node) # Used for displacement recording
        self.loading_node_tags = top_nodes
        self.base_node_tags = base_nodes
        
        # Store contacts and mesh data
        self.contacts = concrete_contacts
        self.contact_node_tags = concrete_contact_node_tags
        
        self.nodes = nodes
        self.n_nodes = n_nodes
        self.total_dof = nodes.size
        
        self.elements = elements
        
        return nodes, elements, concrete_contacts

    def update_node_coordinates(self, u: np.ndarray) -> np.ndarray:
        """Updates the node coordinates after displacement is applied.

        This function updates the mesh node coordinates by adding the displacement values.
        It is used in finite element simulations where nodal displacements occur due to external forces.

        Args:
            u (np.ndarray): Displacement vector, shape (num_nodes, 3), where:
                - `u[:, 0]` represents displacement in x-direction.
                - `u[:, 1]` represents displacement in y-direction.
                - `u[:, 2]` represents displacement in z-direction.

        Returns:
            np.ndarray: Updated node coordinates, shape (num_nodes, 3).

        Raises:
            ValueError: If the displacement array `u` does not match the shape of `self.nodes`.
        """
        # Validate input shape
        if u.shape != self.nodes.shape:
            raise ValueError(f"Displacement shape {u.shape} must match node coordinates shape {self.nodes.shape}.")

        # Update node coordinates
        self.nodes += u.reshape(self.nodes.shape)

        return self.nodes  # Shape (num_nodes, 3)
