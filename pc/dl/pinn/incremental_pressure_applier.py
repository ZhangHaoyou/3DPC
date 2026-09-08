# dl/pinn/incremental_pressure_applier.py


import torch
import platform
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple


from fem.fem.damage import Damage

from dl.pinn.pinn import PINN
from dl.pinn.scaler import Scaler


# Configure global plotting style (Times New Roman for all figures)
if platform.system() == 'Windows':
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic',
        'mathtext.bf': 'Times New Roman:bold',
    })


def voigt_to_tensor(strain: torch.Tensor) -> torch.Tensor:
    """Convert a batch of 6-component Voigt strain vectors into 3x3 tensors.
    
    This assumes the incoming `strain` has shape `(N, 6)` and is ordered as
    `[ε_xx, ε_yy, ε_zz, ε_xy, ε_yz, ε_zx]`.  The returned tensor has shape
    `(N, 3, 3)` with
    
        [[ε_xx, ε_xy, ε_zx],
         [ε_xy, ε_yy, ε_yz],
         [ε_zx, ε_yz, ε_zz]].
        
    Args:
        strain: FloatTensor of shape `(N, 6)` in “Voigt” order.
        
    Returns:
        FloatTensor of shape `(N, 3, 3)`, the symmetric strain tensor.
        
    Raises:
        ValueError: If `strain` is not a 2-D tensor with size 6 in the second dim.
    """
    if strain.dim() != 2 or strain.size(1) != 6:
        raise ValueError(f"Expected strain shape (N,6), got {tuple(strain.shape)}")
    # unpack components
    eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_zx = strain.unbind(dim=1)
    
    # assemble each row
    row0 = torch.stack([eps_xx, eps_xy, eps_zx], dim=1)  # (N,3)
    row1 = torch.stack([eps_xy, eps_yy, eps_yz], dim=1)  # (N,3)
    row2 = torch.stack([eps_zx, eps_yz, eps_zz], dim=1)  # (N,3)
    
    # stack into (N,3,3)
    return torch.stack([row0, row1, row2], dim=1)


class Incremental_Pressure_Applier:
    """Apply incremental pressure loading to a printed concrete column using PINN + damage.
    
    This class drives a loop of:
        1. training the PINN under an applied pressure increment,
        2. predicting top-face response (disp., stress, strain),
        3. updating the damage state and degraded modulus,
        4. updating the geometry for the next increment.
        
    Attributes:
        pinn: A configured PINN solver with geometry, PDE, and BCs.
        dmg:  A damage model that computes stiffness degradation.
    """
    def __init__(self, pinn: PINN, dmg: Damage) -> None:
        """Initialize with a PINN instance and a damage model.
        
        Args:
            pinn: The PINN solver to train and predict displacements/stresses.
            dmg:  The damage model used to degrade material stiffness.
        """
        self.pinn = pinn
        self.dmg  = dmg
        
    def apply_load(self, pressure: float, n_step: int, num_epochs: int) -> None:
        """Apply pressure in n_step increments, retrain PINN, and record response.
        
        Args:
            pressure (float):  Pressure increment to apply at each step.
            n_step (int):      Number of loading increments.
            num_epochs (int):  Number of Adam+L-BFGS epochs per increment.
        
        Side effects:
            - Retrains and re-saves PINN weights for each layer
            - Updates PINN geometry in place
            - Writes `log/pinn/load_disp.csv` containing [disp, load] history
        """
        # 1) Prepare the set of top-face sample points (constant throughout)
        top_points = self.pinn.geom.generate_random_points_on_top(N=self.pinn.num_bc, method='linspace', seed=0)
        N = top_points.shape[0]
        # 2) Initialize accumulators
        disp_accum = torch.zeros([N, 3], device=self.pinn.device)
        stress = torch.zeros([N, 6], device=self.pinn.device)
        strain = torch.zeros([N, 6], device=self.pinn.device)
        d0     = torch.zeros([N, 3], device=self.pinn.device)
        # 3) Histories for output
        load_history = [0.0]
        disp_history = [0.0]
        # 4) Incremental loading loop
        for step in range(n_step):
            if step == 0:
                self.pinn.train(pressure=pressure, num_epochs=num_epochs, step=step)
            # 5) compute current damage -> degraded moduli
            eps_tensor = voigt_to_tensor(strain=strain)
            dmg_vec = self.dmg.calculate_damage(eps=eps_tensor, d0=d0)
            # Compute damaged elastic modulus
            control = self.dmg.para.control.strip().lower()
            valid_modes = {
                "ct": 0, "tc": 0, "compression-tension": 0, "tension-compression": 0,
                "t": 1, "tension": 1,
                "c": 2, "compression": 2,
            }
            if control in valid_modes:
                d = dmg_vec[:, valid_modes[control]]
            else:
                raise ValueError(
                    f"[apply_load] Invalid control mode '{self.dmg.para.control}'.\n"
                    f"-> Expected one of: {', '.join(sorted(valid_modes.keys()))}"
                )
            E_d = (1.0 - d) * self.dmg.mat.prop.E
            # 6) predict under current degraded modulus
            pred_top = self.pinn.predict(x=top_points, E=E_d, step=0)
            disp_accum += pred_top[:, :3]
            stress += pred_top[:, 3:9]
            strain += pred_top[:, 9:15]
            # 7) update damage “history” d0 for next iteration
            eps_tensor = voigt_to_tensor(strain=strain)
            d0 = self.dmg.calculate_damage(eps=eps_tensor, d0=d0, clip=True)
            # 8) record mean vertical displacement and applied load
            disp = disp_accum[:, 2].mean()
            disp_history.append(disp.item())
            load_history.append((step + 1) * pressure)
            # 9) update geometry for next increment
            self.pinn.geom.update_geometry(step=0, pred_func=self.predict)
            # if torch.max(d) >= 0.95:
            #     break
        # 10) dump the load-displacement history
        load_disp = np.column_stack((disp_history, load_history))
        np.savetxt('log/pinn/load_disp.csv', load_disp, delimiter=',')
        print("Saved load-disp history to log/pinn/load_disp.csv")
    
    def predict(self, x: torch.Tensor, n_step: int) -> torch.Tensor:
        """Compute the full (u,v,w,σ,ε) response after `n_step` load increments.
        
        This will:
            1. Determine layer for each point,
            2. Stitch out PINN predictions layer-by-layer (see `_stitch_layer_response`),
            3. Update damage, compute degraded Eₙ,
            4. Rescale to physical units, and
            5. Accumulate over all n_step increments.
        
        Args:
            x (torch.Tensor): FloatTensor of shape (N,3) physical coords.
            n_step (int):     Number of incremental load steps applied.
            
        Returns:
            FloatTensor (N,15) with columns
            [u, v, w_total,
             sxx, syy, szz,
             sxy, syz, szx,
             εₓₓ, εᵧᵧ, ε𝓏𝓏,
             εₓᵧ, εᵧ𝓏, εₓ𝓏].
        """
        # assign each point to a layer index in [0, n_layers)
        layer_ids = self.pinn.geom._calculate_layer_indices(x=x)
        N = x.shape[0]
        disp_acc = torch.zeros([N, 3], device=self.pinn.device)
        stress = torch.zeros([N, 6], device=self.pinn.device)
        strain = torch.zeros([N, 6], device=self.pinn.device)
        d0     = torch.zeros([N, 3], device=self.pinn.device)
        for step in range(n_step):
            # prepare output placeholder
            pred = torch.zeros([N, 15], device=self.pinn.device)
            # precompute vertical offsets at each layer interface
            vertical_disp = self.pinn.predict_vertical_displacements(step=0)
            # evaluate each unique layer
            for layer in torch.unique(layer_ids):
                mask = (layer_ids == layer)
                if not mask.any():
                    continue
                # points in this layer
                x_layer = x[mask]
                # load & eval
                weight_path = f"log/pinn/weights/Step0_layer{layer}_weight.pth"
                state = torch.load(weight_path, map_location=self.pinn.device)
                self.pinn.net.load_state_dict(state)
                # build a scaler for this layer’s physical bounds
                scaler = Scaler(net=self.pinn.net, mins=self.pinn.geom.layer_mins[layer],
                        maxs=self.pinn.geom.layer_maxs[layer])
                # compute original displacement & stress
                u = scaler.calculate_original_displacement(x=x_layer)
                sig = scaler.calculate_original_stress(x=x_layer)
                eps = scaler.calculate_original_strain(x=x_layer)
                out_layer = torch.cat([u, sig, eps], dim=1)
                # add cumulative vertical shift from all underlying layers
                out_layer[:, 2] += sum(vertical_disp[:layer+1])
                # write them back into pred
                pred[mask] = out_layer
            eps_tensor = voigt_to_tensor(strain=strain)
            dmg_vec = self.dmg.calculate_damage(eps=eps_tensor, d0=d0, clip=True)
            control = self.dmg.para.control.strip().lower()
            valid_modes = {
                "ct": 0, "tc": 0, "compression-tension": 0, "tension-compression": 0,
                "t": 1, "tension": 1,
                "c": 2, "compression": 2,
            }
            if control in valid_modes:
                d = dmg_vec[:, valid_modes[control]]
            else:
                raise ValueError(
                    f"[apply_load] Invalid control mode '{self.dmg.para.control}'.\n"
                    f"-> Expected one of: {', '.join(sorted(valid_modes.keys()))}"
                )
            E_d = (1 - d) * self.dmg.mat.prop.E
            pred = self.pinn.calculate_physical_prediction(prediction=pred, E=E_d)
            disp_acc += pred[:, :3]
            stress += pred[:, 3:9]
            strain += pred[:, 9:15]
            eps_tensor = voigt_to_tensor(strain=strain)
            d0 = self.dmg.calculate_damage(eps=eps_tensor, d0=d0, clip=True)
            # if torch.max(d0[:, valid_modes[control]]) >= 0.75:
            #     break
        final_pred = torch.cat([disp_acc, stress, strain], dim=1)
        return final_pred
    
    def save_to_vtu(self, n_step: int) -> None:
        """Generate a 3D mesh of the layered domain and export displacement/stress fields to VTU.
        
        This method:
            1. Uses gmsh to build and mesh stacked boxes for each layer.
            2. Reads the mesh with meshio to get node coordinates and cell connectivity.
            3. Predicts u, v, w, and stresses sxx…szx at each node.
            4. Writes separate VTU files for each field component under `log/pinn/paraview/`.
            
        Side effects:
            - Writes `mesh.msh` for gmsh and multiple `.vtu` files for ParaView.
            - Logs each file write.
        """
        import gmsh
        import meshio
        # 1) Initialize gmsh and build the layered boxes
        gmsh.initialize()
        gmsh.model.add("layered_boxes")
        for i in range(self.pinn.geom.n_layers):
            bbox = self.pinn.geom.bboxes[i].detach().cpu().numpy()
            x_min, y_min, z_min = bbox[0]
            x_max, y_max, z_max = bbox[1]
            gmsh.model.occ.addBox(x_min, y_min, z_min, x_max - x_min, y_max - y_min, z_max - z_min)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        # write to disk so meshio can read
        gmsh.write(f'log/pinn/paraview/mesh.msh')
        gmsh.finalize()
        # 2) Read mesh with meshio
        m = meshio.read(f'log/pinn/paraview/mesh.msh')
        points = m.points               # shape (Nnodes, 3)
        cells  = m.cells                # list of (cell_type, indices) tuples
        # 3) Predict fields at each node
        pts_torch = torch.from_numpy(points.astype(np.float32)).to(self.pinn.device)
        pred = self.predict(x=pts_torch, n_step=n_step).detach().cpu().numpy()    # (Nnodes,15)
        
        disps  = {"dispx": pred[:, 0], "dispy": pred[:, 1], "dispz": pred[:, 2]}
        sigmas = {
            "sig_x":  pred[:, 3],
            "sig_y":  pred[:, 4],
            "sig_z":  pred[:, 5],
            # "sig_xy": pred[:, 6],
            # "sig_yz": pred[:, 7],
            # "sig_zx": pred[:, 8],
        }
        epsilons = {
            "eps_x":  pred[:, 9],
            "eps_y":  pred[:, 10],
            "eps_z":  pred[:, 11],
            # "eps_xy": pred[:, 12],
            # "eps_yz": pred[:, 13],
            # "eps_x": pred[:, 14],
        }
        # 4) Write each field to its own VTU
        for name, data in {**disps, **sigmas, **epsilons}.items():
            vtk = meshio.Mesh(points=points, cells=cells, point_data={name: data})
            vtu_path = f"log/pinn/paraview/{name}.vtu"
            meshio.write(vtu_path, vtk)
            print(f"Wrote {vtu_path}")
    
    def plot_load_disp(self) -> Tuple[plt.Figure, plt.Axes]:
        """Plot and save the load-displacement curves from multiple data sources.
        
        This method loads experimental, FEM, Telichko, and PINN
        load-displacement data from CSV files, plots them on a single figure
        with consistent styling, and saves the result to the project's log
        directory.
        
        Returns:
            Tuple[plt.Figure, plt.Axes]: Matplotlib figure and axes containing
                the combined load-displacement plot.
        """
        # --- Load experimental/comparative data ---
        fem_data = np.loadtxt('log/fem/load_disp.csv', delimiter=',')
        exp_data = np.loadtxt('data/Experimental_data.csv', delimiter=',', skiprows=2)
        telichko_data = np.loadtxt('data/Telichko.csv', delimiter=',', skiprows=2)
        pinn_data = np.loadtxt('log/pinn/load_disp.csv', delimiter=',')
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Plot Experiment - black solid
        # ax.plot(exp_data[:, 0], exp_data[:, 1], label='Experiment', color='black', linestyle='-', linewidth=2)
        
        # Plot Telichko - blue dashed with transparency
        # ax.plot(telichko_data[:, 0], telichko_data[:, 1], label="Telichko's Simulation", color=(0, 47/255, 167/255),
        #         linestyle='--', linewidth=1.5, alpha=0.6)
        
        # Plot FEM - red dashed with transparency
        ax.plot(-fem_data[:, 0], -fem_data[:, 1], label='FEM', color=(192/255, 0, 0),
                linestyle='--', marker='o', linewidth=1.5, alpha=0.6)
        
        # Plot PINN - gold dashed with transparency
        load = (-pinn_data[:, 1] * self.pinn.geom.area) / 1000.0
        ax.plot(-pinn_data[:, 0], load, label='PINN', color=(255/255,204/255,0),
                linestyle='--', marker='o', linewidth=1.5, alpha=0.6)
        
        # Axis settings
        ax.set_xlabel('Displacement (mm)', fontsize=12)
        ax.set_ylabel('Load (kN)', fontsize=12)
        ax.set_xlim(0, 0.3)
        ax.set_ylim(0, 80)
        ax.set_xticks(np.arange(0, 0.4, 0.1))
        ax.set_yticks(np.arange(0, 90, 20))
        ax.tick_params(labelsize=10)
        
        # Legend
        ax.legend(fontsize=10, loc='lower right')
        
        # Save figure
        fig.tight_layout()
        fig.savefig('log/pinn/Result.png', dpi=300)
        
        return fig, ax
    
