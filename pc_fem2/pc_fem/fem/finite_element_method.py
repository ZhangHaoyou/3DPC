# /fem/finite_element_method.py

import numpy as np
import matplotlib
import platform

if platform.system() != 'Windows':
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from typing import List, Tuple


from pc_fem.parameters import Contact_Parameter
from pc_fem.fem.materials import Material
from pc_fem.fem.damage import Damage
from pc_fem.fem.geo_mesh import Mesh
from pc_fem.fem.elements import Hexahedral_Element, Contact_Element
from pc_fem.fem.boundary_conditions import Boundary_Conditions
from pc_fem.fem.solvers import Solver

from pc_fem.utils.utils import local_to_global_index, save_to_vtu
from pc_fem.utils.log_config import setup_logger

class Finite_Element_Method:
    """Finite Element Method (FEM) solver for structural analysis.

    This class encapsulates the core logic for performing FEM-based simulations.
    It manages material models, geometry (mesh), and damage mechanics, and provides
    tools for computing stiffness matrices and solving the system.

    Attributes:
        mat (Material): Material model containing constitutive behavior.
        mesh (Mesh): Finite element mesh containing nodes and elements.
        dmg (Damage): Damage model controlling degradation in stiffness.
    """

    def __init__(self, mat: Material, mesh: Mesh, dmg: Damage) -> None:
        """Initializes the FEM solver with material, mesh, and damage model.

        Args:
            mat (Material): Material object defining mechanical properties.
            mesh (Mesh): Mesh object containing node and element data.
            dmg (Damage): Damage model instance controlling stiffness reduction.
        """
        self.mat = mat
        self.mesh = mesh
        self.dmg = dmg
        
        # Initialize logger
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info("Initialized Finite_Element_Method with material: %s, mesh: %s, damage model: %s",
                        type(mat).__name__, type(mesh).__name__, type(dmg).__name__)
        
        # Configure global plotting style (Times New Roman for all figures)
        if platform.system() == 'Windows':
            plt.rcParams.update({
                'font.family': 'Times New Roman',
                'mathtext.fontset': 'custom',
                'mathtext.rm': 'Times New Roman',
                'mathtext.it': 'Times New Roman:italic',
                'mathtext.bf': 'Times New Roman:bold',
            })
    
    def configure_material(self) -> np.ndarray:
        """Configures the material properties and computes the constitutive matrix.

        This method extracts the elastic modulus from the material properties and
        initializes the material stiffness matrix.

        Raises:
            KeyError: If `E` (Young’s modulus) is missing in `self.mat.prop`.
        """

        print("Configuring materials...")

        # Validate material property existence
        self.mat.print_properties()
        
        # Compute constitutive (stiffness) matrix
        D_initial = self.mat.calculate_D(E=self.mat.prop.E)
        
        self.D_initial = D_initial
        
        return D_initial
    
    def configure_geometry(self, gap_between_layers: float, plot_geometry: bool) -> Tuple[np.ndarray, np.ndarray, List[List[np.ndarray]]]:
        """Configures the geometry and mesh of the structure.

        This method calls the mesh object's `mesh_layers()` function to generate the
        nodes, elements, and contact surfaces for all layers. These values are
        stored in the Finite Element Method (FEM) instance and optionally visualized.

        Args:
            gap_between_layers (float, optional): Gap (in mm) between printed layers. 
                Default is 0.0.
            plot_geometry (bool, optional): If True, visualizes the mesh after generation.
                Default is False.

        Returns:
            Tuple[np.ndarray, np.ndarray, list]: A tuple containing:
                - nodes (np.ndarray): Array of nodal coordinates with shape (N, 3).
                - elements (np.ndarray): Array of element connectivity with shape (M, 8).
                - contacts (list): List of contact data between adjacent layers.

        Raises:
            ValueError: If mesh generation fails or returns invalid data.
        """
        # Mesh generation from associated mesh object
        nodes, elements, contacts = self.mesh.mesh_layers(
            gap_between_layers=gap_between_layers,
            plot_geometry=plot_geometry
        )
        
        save_to_vtu('log/paraview/geometry.vtu', nodes, elements, scalar_name='Geometry')

        # Save generated mesh to instance variables
        self.nodes = nodes
        self.elements = elements
        self.contacts = contacts
        
        return nodes, elements, contacts

    def configure_recorder(self):
        pass
    
    def analyze_bricks(self, pressure_per_step: float = -1.0, n_steps: int = 1, move_vertically: bool = False, B_bar: bool = True) -> None:
        """Performs static linear brick element analysis for multiple load steps.

        This function assembles the global stiffness matrix, applies boundary conditions,
        solves the global system using a linear solver, and records the load-displacement
        curve at the top-center node.

        Args:
            pressure_per_step (float, optional): Applied pressure per step (MPa). Negative for compression.
            n_steps (int, optional): Number of load steps. Default is 1.
            move_vertically (bool, optional): If True, constrains x/y displacements at top. Default is False.
            B_bar (bool, optional): Whether to use the B-bar formulation. Default is True.

        Saves:
            Displacement, stress, strain, and plastic strain CSV files and VTU outputs for visualization.
        """
        mat = self.mat
        mesh = self.mesh
        dmg = self.dmg
        
        # Step 1: Load setup
        area = mesh.geo.para.lx * mesh.geo.para.ly  # Cross-section area (mm²)
        loading_force = pressure_per_step * area               # Total load (N)
        dx, dy = self.mesh.para.mesh_size[:2]
        loading_inner_node_force = pressure_per_step * dx * dy  # Load at inner node (N)
        
        # Initialize recorder
        recorder_disp = [0.0]
        recorder_load = [0.0]
        disp_index = local_to_global_index([mesh.top_center_node_tag])  # Index to track displacement

        # Step 2: Initialize solver components
        hexa = Hexahedral_Element(mat=mat, mesh=mesh, dmg=dmg)
        bc = Boundary_Conditions(mesh=mesh, move_vertically=move_vertically)
        solver = Solver()

        total_dof = mesh.total_dof
        current_u = np.zeros((total_dof, 1))  # Global displacement vector (initial)
        
        n_elements = mesh.elements.shape[0]
        current_damage = np.zeros((n_elements, 8)) # 8 Gaussian integration points

        for step in range(n_steps):
            print(f'\n🔧 Loading step {step + 1} / {n_steps}...\n')
            
            # Step 3: Recalculate stiffness matrix
            current_nodes = mesh.nodes + current_u.reshape(mesh.nodes.shape)
            print('\nComputing stiffness matrix...')
            global_ke, current_damage = hexa.calculate_global_stiffness(
                    nodes=mesh.nodes, # current_nodes,
                    elements=mesh.elements,
                    u=current_u, d0=current_damage, B_bar=B_bar)
            
            if step == 0:
                print('📤 Saving initial stiffness matrix to validation/APDL/elastic_brick/ke.csv...')
                np.savetxt('validation/APDL/elastic_brick/ke.csv', global_ke, delimiter=',')

            # Step 4: Initialize solver components
            print('\nSetting boundary conditions...')
            global_f = np.zeros((total_dof, 1))
            global_kbc, global_f = bc.apply_supports(global_ke, global_f)
            global_kbc, global_f = bc.apply_nodal_loading(
                pressure=pressure_per_step,
                global_k=global_kbc,
                global_f=global_f,
            )
            
            if step == 0:
                print('📤 Saving initial stiffness matrix with BCs to validation/APDL/elastic_brick/kbc.csv...')
                np.savetxt('validation/APDL/elastic_brick/kbc.csv', global_kbc, delimiter=',')
            
            # Step 5: Solve linear system Ku = f
            print("\nSolving linear system...")
            u_step = solver.linear_solver(global_kbc, global_f)
            current_u += u_step  # Accumulate displacements

            # Record displacement and force at top-center node
            recorder_disp.append(float(recorder_disp[-1] + u_step[disp_index[2]][0]))  # uz
            recorder_load.append(recorder_load[-1] + loading_force)               # N

        # Step 6: Save final results
        np.savetxt('validation/APDL/elastic_brick/u.csv', current_u, delimiter=',')
        np.savetxt('validation/APDL/elastic_brick/k_final.csv', global_kbc, delimiter=',')

        # Prepare and save load-displacement curve
        recorder_disp = np.array(recorder_disp)
        recorder_load = np.array(recorder_load) / 1000.0  # Convert N to kN
        load_disp = np.column_stack((recorder_disp, recorder_load))
        np.savetxt('validation/APDL/elastic_brick/load_disp.csv', load_disp, delimiter=',', header='disp (mm), load (kN)')
        
        # Store results in object
        self.bc = bc
        self.hexa = hexa
        self.total_u = current_u
        self.total_disp = current_u.reshape(mesh.nodes.shape)
        
        # Step 7: Compute element stress and strain
        ndof = self.mesh.geo.para.ndof
        stress = np.empty((6, 0))
        strain = np.empty((6, 0))
        final_node_coords = mesh.nodes + current_u.reshape(mesh.nodes.shape)
        
        for i, ele in enumerate(mesh.elements):
            coords = final_node_coords[ele]
            global_i = local_to_global_index(ele, dof=ndof)
            sig, eps = hexa.calculate_element_stress_strain(node_coords=coords, u=current_u[global_i], current_d=current_damage[i])
            stress = np.hstack((stress, sig))
            strain = np.hstack((strain, eps))
        
        eps_plastic_eq = hexa.calculate_eps_plastic_eq_ansys(eps=strain)
        
        np.savetxt('validation/APDL/elastic_brick/stress.csv', stress.T, delimiter=',', header='SX, SY, SZ, SXY, SYZ, SXZ')
        np.savetxt('validation/APDL/elastic_brick/strain.csv', strain.T, delimiter=',', header='EX, EY, EZ, EXY, EYZ, EXZ')
        np.savetxt('validation/APDL/elastic_brick/equivalent_plastic_strain.csv', eps_plastic_eq, delimiter=',', header='EPS')
        np.savetxt('validation/APDL/elastic_brick/final_node_coords.csv', final_node_coords, delimiter=',')
        
        print('Saving a hexahedral mesh and associated scalar data to a VTU file for visualization in ParaView.')
        
        disp = current_u.reshape(final_node_coords.shape)
        ux, uy, uz = disp[:, 0], disp[:, 1], disp[:, 2]
        
        elements = mesh.elements
        
        # --- Smooth element-based results to nodal values ---
        elements = mesh.elements[np.newaxis, :] if mesh.elements.ndim == 1 else mesh.elements
        smooth = lambda val: hexa.smooth_result(n_nodes=mesh.n_nodes, elements=elements, values=val.reshape(elements.shape))
        sx, sy, sz = map(smooth, [stress[0], stress[1], stress[2]])
        ex, ey, ez = map(smooth, [strain[0], strain[1], strain[2]])
        eps_plastic_eq = smooth(eps_plastic_eq)
        
        # --- Save deformation and result fields as VTU files for ParaView ---
        save_to_vtu('validation/APDL/elastic_brick/deformed_geometry.vtu', final_node_coords, mesh.elements, scalar_name='Deformed Geometry')

        save_to_vtu('validation/APDL/elastic_brick/disp1.vtu', final_node_coords, elements, scalar=ux, scalar_name='Displacement_X')
        save_to_vtu('validation/APDL/elastic_brick/disp2.vtu', final_node_coords, elements, scalar=uy, scalar_name='Displacement_Y')
        save_to_vtu('validation/APDL/elastic_brick/disp3.vtu', final_node_coords, elements, scalar=uz, scalar_name='Displacement_Z')

        save_to_vtu('validation/APDL/elastic_brick/stress1.vtu', final_node_coords, elements, scalar=sx, scalar_name='Stress_X')
        save_to_vtu('validation/APDL/elastic_brick/stress2.vtu', final_node_coords, elements, scalar=sy, scalar_name='Stress_Y')
        save_to_vtu('validation/APDL/elastic_brick/stress3.vtu', final_node_coords, elements, scalar=sz, scalar_name='Stress_Z')

        save_to_vtu('validation/APDL/elastic_brick/strain1.vtu', final_node_coords, elements, scalar=ex, scalar_name='Strain_X')
        save_to_vtu('validation/APDL/elastic_brick/strain2.vtu', final_node_coords, elements, scalar=ey, scalar_name='Strain_Y')
        save_to_vtu('validation/APDL/elastic_brick/strain3.vtu', final_node_coords, elements, scalar=ez, scalar_name='Strain_Z')
        save_to_vtu('validation/APDL/elastic_brick/equivalent_plastic_strain.vtu', final_node_coords, elements,
                    scalar=eps_plastic_eq, scalar_name='Equivalent_Plastic_Strain')
        print('✅ Brick Analysis done...')
    
    def log_loading_node_displacements(self, current_u: np.ndarray):
        """Log displacement information of loading surface nodes.

        Logs each node's ID, coordinates (in mm), and displacements (in mm) along x, y, z.

        Args:
            current_u (np.ndarray): Global displacement vector, shape = (ndof_total, )
        """
        loading_tags = self.mesh.loading_node_tags
        node_coords = self.mesh.nodes[loading_tags]
        ndof = self.mesh.geo.para.ndof

        self.logger.debug("Logging displacement info for %d loading nodes...", len(loading_tags))
        for idx, tag in enumerate(loading_tags):
            ux = current_u[tag * ndof + 0]
            uy = current_u[tag * ndof + 1]
            uz = current_u[tag * ndof + 2]
            x, y, z = node_coords[idx]

            self.logger.debug(
                "Node ID: %d | Coord (mm): [%.1f, %.1f, %.1f] | Displacement (mm): [%.3e, %.3e, %.3e]",
                tag, x, y, z, ux, uy, uz
            )
    
    def log_maximum_displacement(self, u: np.ndarray, nodes: np.ndarray) -> None:
        """Logs and prints the maximum displacement value and the corresponding node coordinates.

        This function reshapes the global displacement vector `u` to match the nodal layout,
        computes the displacement magnitude at each node, and finds the node with the largest
        displacement. Both console output and logger are used for reporting.

        Args:
            u (np.ndarray): Global displacement vector of shape (3 * n_nodes, 1).
            nodes (np.ndarray): Nodal coordinates of shape (n_nodes, 3).

        Returns:
            None
    """
        # Reshape displacement vector to match node layout
        disp = u.reshape(nodes.shape)  # shape: (n_nodes, 3)

        # Compute L2 norm of displacement at each node
        disp_magnitudes = np.linalg.norm(disp, axis=1)

        # Identify node with max displacement
        max_index = np.argmax(disp_magnitudes)
        max_vector = disp[max_index]
        max_value = np.linalg.norm(max_vector)
        max_coord = nodes[max_index]

        # Create log message
        msg = "Maximum displacement:\n"
        msg += f"      Value: {max_value:.6e}\n"
        msg += f"      Node Index: {max_index}\n"
        msg += f"      Displacement Vector: {np.array2string(max_vector, precision=6)}\n"
        msg += f"      Node Coordinates   : {np.array2string(max_coord, precision=2)}"
        
        # Print and log the message
        print('\n📌 ', msg)
        self.logger.debug(msg)
    
    def log_extreme_damage_strain_info(self, d: np.ndarray, eps: np.ndarray, find_largest: bool = True) -> np.ndarray:
        """
        Logs the strain vector at the Gauss point with the highest or lowest damage 
        according to the specified damage control mode.

        Args:
            d (np.ndarray): Damage tensor, shape (n_elements, 8, 3), where
                            last dim = [combined (tc), tension (t), compression (c)].
            eps (np.ndarray): Strain tensor, shape (n_elements, 8, 6).
            find_largest (bool): If True, find the Gauss point with maximum damage.
                                If False, find the Gauss point with minimum damage.

        Returns:
            np.ndarray: Strain vector (6,) at the selected Gauss point.
        """
        control_mode = self.dmg.para.control.strip().lower()
        control_to_index = {
            'ct': 0, 'tc': 0, 'compression-tension': 0, 'tension-compression': 0,
            't': 1, 'tension': 1,
            'c': 2, 'compression': 2,
        }
        
        if control_mode not in control_to_index:
            raise ValueError(f"Unsupported control mode: '{control_mode}'")

        col_idx = control_to_index[control_mode]
        d_selected = d[:, :, col_idx]  # shape: (n_elements, 8)

        # Find maximum or minimum damage location
        if find_largest:
            extreme_idx = np.argmax(d_selected)
        else:
            extreme_idx = np.argmin(d_selected)
        ele_idx, gp_idx = np.unravel_index(extreme_idx, d_selected.shape)

        # Extract full damage vector and strain
        d_tc, d_t, d_c = d[ele_idx, gp_idx]
        eps_voigt = eps[ele_idx, gp_idx]  # shape: (6,)

        # Prepare message
        extreme_type = "Maximum" if find_largest else "Minimum"
        msg = (
            f"{extreme_type} damage found in element {ele_idx}, Gauss point {gp_idx}\n"
            f"    Damage values:\n"
            f"      d  (combined) = {d_tc:.2f}\n"
            f"      dt (tension)  = {d_t:.2f}\n"
            f"      dc (compression) = {d_c:.2f}\n"
            f"    Strain vector (epsilon) = {np.array2string(eps_voigt, precision=4)}"
        )

        # Output
        print('\n📌 ', msg)
        self.logger.debug(msg)
        
        return eps_voigt
    
    def analyze_contacts(self, contact_para: Contact_Parameter, pressure_per_step: float = -1.0, n_steps: int = 1,
                    move_vertically: bool = False, B_bar: bool = True) -> None:
        """Performs static nonlinear contact element analysis for multiple load steps.

        This function assembles the global stiffness matrix, applies boundary conditions,
        solves the global system using a Newton-Raphson solver, and records the load-displacement
        response at the top-center node, with dynamic plotting and logging.

        Args:
            contact_para (Contact_Parameter): Contact property configuration.
            pressure_per_step (float): Applied pressure per load step in MPa (negative means compression).
            n_steps (int): Total number of loading steps.
            move_vertically (bool): If True, fixes x/y displacement of top nodes.
            B_bar (bool): Flag to use B-bar enhanced strain formulation.

        Saves:
            - Final displacements, stiffness matrices, and load-displacement data to `log/kfu/`
            - Real-time plot with comparison curves and contact parameters
        """
        mat = self.mat
        mesh = self.mesh
        dmg = self.dmg
        
        # Step 1: Load setup
        area = mesh.geo.para.lx * mesh.geo.para.ly  # Cross-section area (mm²)
        loading_force = pressure_per_step * area               # Total load (N)
        
        # Initialize recorder
        recorder_disp = [0.0]
        recorder_load = [0.0]
        recorder_eps = []
        disp_index = local_to_global_index([mesh.top_center_node_tag])  # Index to track displacement

        # Step 2: Initialize FEA components
        hexa = Hexahedral_Element(mat=mat, mesh=mesh, dmg=dmg)
        bc = Boundary_Conditions(mesh=mesh, move_vertically=move_vertically)
        contact = Contact_Element(hexa=hexa, para=contact_para)
        solver = Solver()

        total_dof = mesh.total_dof
        current_u = np.zeros((total_dof, 1))  # Global displacement vector (initial)
        
        n_elements = mesh.elements.shape[0]
        current_damage = np.zeros((n_elements, 8, 3)) # 8 Gaussian integration points

        # Enable interactive mode for real-time plot updates
        if platform.system() == 'Windows':
            plt.ion()
        
        # Load external reference data
        exp_data = np.loadtxt('data/Experimental_data.csv', delimiter=',', skiprows=2)
        telichko_data = np.loadtxt('data/Telichko.csv', delimiter=',', skiprows=2)

        # Initialize figure and axis
        fig, ax = plt.subplots(figsize=(9, 6))
        
        # Plot reference curves
        ax.plot(exp_data[:, 0], exp_data[:, 1], label='Experiment', color='black', linestyle='-', linewidth=2)
        ax.plot(telichko_data[:, 0], telichko_data[:, 1], label="Telichko's Simulation", color=(0, 47/255, 167/255), linestyle='--', linewidth=1.5, alpha=0.6)

        # Initialize "Ours" line as an empty dashed line
        ours_line, = ax.plot([], [], label='Ours', color=(192/255, 0, 0), linestyle='--', marker='o', linewidth=1.5, alpha=0.6)
        
        # Configure plot appearance
        ax.set_xlabel('Displacement (mm)', fontsize=12)
        ax.set_ylabel('Load (kN)', fontsize=12)
        ax.set_xlim(0, 1.5)
        ax.set_ylim(0, 70)
        ax.set_xticks(np.arange(0, 1.5, 0.5))
        ax.set_yticks(np.arange(0, 80, 10))
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=14, loc='upper left')
        
        # Add parameter info in the bottom right corner
        param_text = (f"$A_{{\\mathrm{{c}}}}$ = {dmg.para.Ac:.2f}, $B_{{\\mathrm{{c}}}}$ = {dmg.para.Bc:.0f}\n"
                f"$A_{{\\mathrm{{t}}}}$ = {dmg.para.At:.2f}, $B_{{\\mathrm{{t}}}}$ = {dmg.para.Bt:.0f}\n"
                f"$\epsilon_N$ = {contact.para.epsilon[0]}, $\epsilon_T$ = {contact.para.epsilon[1]}")
        ax.text(0.97, 0.05, param_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right')
        
        # Main loading loop
        for step in range(n_steps):
            msg = f"Loading step {step + 1} / {n_steps}..."
            print('\n🔧 ', msg)
            self.logger.debug(msg)
            
            # Step 3: Recalculate stiffness matrix
            msg = 'Computing stiffness matrix...'
            print('🚀 ', msg)
            self.logger.debug(msg)
            current_nodes = mesh.nodes + current_u.reshape(mesh.nodes.shape)
            global_ke, current_damage = hexa.calculate_global_stiffness(
                    nodes=mesh.nodes, # current_nodes
                    elements=mesh.elements,
                    u=current_u, d0=current_damage, B_bar=B_bar)
            
            # if step == 0:
            #     print('📤 Saving initial stiffness matrix to log/kfu/ke.csv...')
            #     np.savetxt('log/kfu/ke.csv', global_ke, delimiter=',')
            
            # Step 4: Apply boundary conditions
            msg = 'Setting boundary conditions...'
            print('🚀 ', msg)
            self.logger.debug(msg)
            global_f = np.zeros((total_dof, 1))
            global_kbc, global_f = bc.apply_supports(global_ke, global_f)
            
            # if step == 0:
            #     print('📤 Saving initial stiffness matrix with BCs to log/kfu/kbc.csv...')
            #     np.savetxt('log/kfu/kbc.csv', global_kbc, delimiter=',')
            
            # Step 5: Solve nonlinear contact system using Newton-Raphson method
            u_step, global_k_final = solver.newton_raphson(
                    pressure=pressure_per_step,
                    global_k=global_kbc,
                    global_f=global_f,
                    contact=contact,
                    mesh=mesh,
                    bc=bc,
                    )
            
            if u_step is None or not solver.coverage:
                break
            else:
                current_u += u_step  # Accumulate displacements
            
            self.log_loading_node_displacements(current_u=current_u)
            self.log_maximum_displacement(u=np.abs(current_u), nodes=current_nodes)
            eps_vogit = self.log_extreme_damage_strain_info(d=hexa.current_damage, eps=hexa.current_eps, find_largest=True)
            self.log_extreme_damage_strain_info(d=hexa.current_damage, eps=hexa.current_eps, find_largest=False)
            recorder_eps.append(eps_vogit)
            
            # Record displacement and force at top-center node
            recorded_node_disp = float(current_u[disp_index[2]][0])  # uz
            global_loading_node_tags = local_to_global_index(self.mesh.loading_node_tags, dof=self.mesh.geo.para.ndof)
            recorded_node_disp = float(np.mean(current_u[global_loading_node_tags[2::3]]))
            recorded_total_load = recorder_load[-1] + loading_force  # N
            
            max_current_damage = np.max(hexa.current_damage)
            min_current_damage = np.min(hexa.current_damage)
            max_current_eq_eps = np.around(np.max(hexa.current_eps_ep), 5)
            min_current_eq_eps = np.around(np.min(hexa.current_eps_ep), 5)
            
            print(f'\n✅ Current load =', recorded_total_load / 1000.0, 'kN')
            print(f'✅ Current min. damage = {min_current_damage}, min. equivalent strain = {min_current_eq_eps}')
            print(f'✅ Current max. damage = {max_current_damage}, max. equivalent strain = {max_current_eq_eps}')
            print(f'✅ Current disp =', recorded_node_disp, f'(mm) at node {mesh.top_center_node_tag}')
            
            msg = f"Current load = {recorded_total_load / 1000.0:.3f} kN\n"
            msg += f"Current max. damage = {max_current_damage}, max. equivalent strain = {max_current_eq_eps}\n"
            msg += f"Current disp = {recorded_node_disp:.6f} mm at node {mesh.top_center_node_tag}"
            self.logger.debug(msg)

            recorder_disp.append(recorded_node_disp)
            recorder_load.append(recorded_total_load)
            
            # --- Update the red dashed line ("Ours") dynamically ---
            ours_line.set_data(-np.array(recorder_disp), -np.array(recorder_load) / 1000.0)

            # Render update
            if platform.system() == 'Windows':
                plt.pause(0.1)

            if np.abs(recorded_node_disp) > 1.0:
                end_msg = f"Reached target displacement at loading step {step + 1} / {n_steps}"
                print('\n🎯 ', end_msg)
                self.logger.info(end_msg)
                break

        # Step 6: Save final results
        np.savetxt('log/kfu/u.csv', current_u, delimiter=',')
        np.savetxt('log/kfu/k_final.csv', global_k_final, delimiter=',')
        np.savetxt('log/kfu/eps_voigt.csv', np.array(recorder_eps), delimiter=',')

        # Prepare and save load-displacement curve
        recorder_disp = np.array(recorder_disp)
        recorder_load = np.array(recorder_load) / 1000.0  # Convert N to kN
        load_disp = np.column_stack((recorder_disp, recorder_load))
        np.savetxt('log/load_disp.csv', load_disp, delimiter=',', header='disp (mm), load (kN)')
        
        # Store results in object
        self.bc = bc
        self.hexa = hexa
        self.contact = contact
        self.total_u = current_u
        self.total_disp = current_u.reshape(current_nodes.shape)
        
        # --- Load experimental/comparative data ---
        exp_data = np.loadtxt('data/Experimental_data.csv', delimiter=',', skiprows=2)
        telichko_data = np.loadtxt('data/Telichko.csv', delimiter=',', skiprows=2)

        fig, ax = plt.subplots(figsize=(9, 6))
        
        # Plot Experiment - black solid
        ax.plot(exp_data[:, 0], exp_data[:, 1], label='Experiment', color='black', linestyle='-', linewidth=2)
        
        # Plot Telichko - blue dashed with transparency
        ax.plot(telichko_data[:, 0], telichko_data[:, 1], label="Telichko's Simulation", color=(0, 47/255, 167/255),
                linestyle='--', linewidth=1.5, alpha=0.6)

        # Plot Ours - red dashed with transparency
        ax.plot(-load_disp[:, 0], -load_disp[:, 1], label='Ours', color=(192/255, 0, 0),
                linestyle='--', marker='o', linewidth=1.5, alpha=0.6)
        
        # Add parameter info in the bottom right corner
        param_text = (f"$A_{{\\mathrm{{c}}}}$ = {dmg.para.Ac:.2f}, $B_{{\\mathrm{{c}}}}$ = {dmg.para.Bc:.0f}\n"
                f"$A_{{\\mathrm{{t}}}}$ = {dmg.para.At:.2f}, $B_{{\\mathrm{{t}}}}$ = {dmg.para.Bt:.0f}\n"
                rf"$\epsilon_N$ = {contact.para.epsilon[0]}, $\epsilon_T$ = {contact.para.epsilon[1]}")
        ax.text(0.97, 0.05, param_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right')

        # Axis settings
        ax.set_xlabel('Displacement (mm)', fontsize=16)
        ax.set_ylabel('Load (kN)', fontsize=16)
        ax.set_xlim(0, 1.5)
        ax.set_ylim(0, 70)
        ax.set_xticks(np.arange(0, 1.5, 0.5))
        ax.set_yticks(np.arange(0, 80, 10))
        ax.tick_params(labelsize=12)

        # Legend
        ax.legend(fontsize=14, loc='upper left')

        # Save figure
        fig.tight_layout()
        fig.savefig('log/Result.png', dpi=300)
        
        # calculate stress and strain
        dof = 3
        stress = np.empty((6, 0))
        strain = np.empty((6, 0))
        final_node_coords = mesh.nodes + current_u.reshape(mesh.nodes.shape)
        
        for i, ele in enumerate(mesh.elements):
            coords = final_node_coords[ele]
            global_i = local_to_global_index(ele, dof=dof)
            sigma, eps = hexa.calculate_element_stress_strain(node_coords=coords, u=current_u[global_i], current_d=current_damage[i])
            stress = np.hstack((stress, sigma))
            strain = np.hstack((strain, eps))
        
        equivalent_plastic_strain = hexa.calculate_eps_plastic_eq_ansys(eps=strain)
        
        np.savetxt('log/kfu/stress.csv', stress.T, delimiter=',', header='SX, SY, SZ, SXY, SYZ, SXZ')
        np.savetxt('log/kfu/strain.csv', strain.T, delimiter=',', header='EX, EY, EZ, EXY, EYZ, EXZ')
        np.savetxt('log/kfu/equivalent_plastic_strain.csv', equivalent_plastic_strain, delimiter=',', header='EPS')
        np.savetxt('log/kfu/final_node_coords.csv', final_node_coords, delimiter=',')

    def post_process(self, show: bool = True) -> None:
        """Performs post-processing tasks after finite element analysis.

        This includes:
        - Loading computed displacement, stress, strain, and plastic strain.
        - Applying smoothing to element-level data.
        - Saving results to `.vtu` files for visualization in ParaView.
        - Plotting and saving load-displacement curves with comparisons to experiments.

        Args:
            show (bool, optional): Whether to display the load-displacement plot using matplotlib. 
                Defaults to True.

        Saves:
            - Various `.vtu` files for stress, strain, and displacement.
            - A load-displacement comparison plot at 'log/Result.png'.
        """
        mesh = self.mesh
        hexa = self.hexa
        dmg = self.dmg
        contact = self.contact
        
        # --- Load numerical results from disk ---
        elements = np.loadtxt('log/geo_mesh/elements.csv', delimiter=',').astype(int)
        stress = np.loadtxt('log/kfu/stress.csv', delimiter=',').T          # shape: (6, num_gauss_pts)
        strain = np.loadtxt('log/kfu/strain.csv', delimiter=',').T
        eps_p_eq = np.loadtxt('log/kfu/equivalent_plastic_strain.csv', delimiter=',')
        
        final_coords = np.loadtxt('log/kfu/final_node_coords.csv', delimiter=',')
        total_u = np.loadtxt('log/kfu/u.csv', delimiter=',')
        disp = total_u.reshape(final_coords.shape)
        ux, uy, uz = disp[:, 0], disp[:, 1], disp[:, 2]
        
        # --- Smooth element-based results to nodal values ---
        smooth = lambda val: hexa.smooth_result(n_nodes=mesh.n_nodes, elements=elements, values=val.reshape(elements.shape))
        sx, sy, sz = map(smooth, [stress[0], stress[1], stress[2]])
        ex, ey, ez = map(smooth, [strain[0], strain[1], strain[2]])
        eps_p = smooth(eps_p_eq)
        
        # --- Save deformation and result fields as VTU files for ParaView ---
        if elements.ndim == 1:
            elements = elements[np.newaxis, :]

        print('Saving a hexahedral mesh and associated scalar data to a VTU file for visualization in ParaView.')
        
        save_to_vtu('log/paraview/deformed_geometry.vtu', final_coords, elements, scalar_name='Deformed Geometry')

        save_to_vtu('log/paraview/disp1.vtu', final_coords, elements, scalar=ux, scalar_name='Displacement_X')
        save_to_vtu('log/paraview/disp2.vtu', final_coords, elements, scalar=uy, scalar_name='Displacement_Y')
        save_to_vtu('log/paraview/disp3.vtu', final_coords, elements, scalar=uz, scalar_name='Displacement_Z')

        save_to_vtu('log/paraview/stress1.vtu', final_coords, elements, scalar=sx, scalar_name='Stress_X')
        save_to_vtu('log/paraview/stress2.vtu', final_coords, elements, scalar=sy, scalar_name='Stress_Y')
        save_to_vtu('log/paraview/stress3.vtu', final_coords, elements, scalar=sz, scalar_name='Stress_Z')

        save_to_vtu('log/paraview/strain1.vtu', final_coords, elements, scalar=ex, scalar_name='Strain_X')
        save_to_vtu('log/paraview/strain2.vtu', final_coords, elements, scalar=ey, scalar_name='Strain_Y')
        save_to_vtu('log/paraview/strain3.vtu', final_coords, elements, scalar=ez, scalar_name='Strain_Z')
        save_to_vtu('log/paraview/equivalent_strain.vtu', final_coords, elements,
                    scalar=eps_p, scalar_name='Equivalent_Strain')
        
        # --- Load experimental/comparative data ---
        load_disp = np.loadtxt('log/load_disp.csv', delimiter=',')
        exp_data = np.loadtxt('data/Experimental_data.csv', delimiter=',', skiprows=2)
        telichko_data = np.loadtxt('data/Telichko.csv', delimiter=',', skiprows=2)

        fig, ax = plt.subplots(figsize=(9, 6))
        
        # Plot Experiment - black solid
        ax.plot(exp_data[:, 0], exp_data[:, 1], label='Experiment', color='black', linestyle='-', linewidth=2)
        
        # Plot Telichko - blue dashed with transparency
        ax.plot(telichko_data[:, 0], telichko_data[:, 1], label="Telichko's Simulation", color=(0, 47/255, 167/255),
                linestyle='--', linewidth=1.5, alpha=0.6)

        # Plot Ours - red dashed with transparency
        ax.plot(-load_disp[:, 0], -load_disp[:, 1], label='Ours', color=(192/255, 0, 0),
                linestyle='--', marker='o', linewidth=1.5, alpha=0.6)
        
        # Add parameter info in the bottom right corner
        param_text = (f"$A_{{\\mathrm{{c}}}}$ = {dmg.para.Ac:.2f}, $B_{{\\mathrm{{c}}}}$ = {dmg.para.Bc:.0f}\n"
                f"$A_{{\\mathrm{{t}}}}$ = {dmg.para.At:.2f}, $B_{{\\mathrm{{t}}}}$ = {dmg.para.Bt:.0f}\n"
                rf"$\epsilon_N$ = {contact.para.epsilon[0]}, $\epsilon_T$ = {contact.para.epsilon[1]}")
        ax.text(0.97, 0.05, param_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right')

        # Axis settings
        ax.set_xlabel('Displacement (mm)', fontsize=16)
        ax.set_ylabel('Load (kN)', fontsize=16)
        ax.set_xlim(0, 1.5)
        ax.set_ylim(0, 70)
        ax.set_xticks(np.arange(0, 1.5, 0.5))
        ax.set_yticks(np.arange(0, 80, 10))
        ax.tick_params(labelsize=12)

        # Legend
        ax.legend(fontsize=14, loc='upper left')

        # Save figure
        fig.tight_layout()
        fig.savefig('log/Result.png', dpi=300)

        if show:
            plt.show()


