# dl/graph/pinn.py


import os
import torch
import numpy as np
import torch.nn as nn

from typing import List
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph


from dl.graph.pde import PDE_Graph
from dl.graph.scaler import Scaler_Graph
from dl.pinn.geometry import Layered_Contact_Geometry
from dl.pinn.contact_boundary_conditions import Contact_Boundary_Condition

from fem.utils.log_config import setup_logger


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class PINN_Graph:
    """Physics-informed neural network solver for layered contact elasticity.
    
    This class orchestrates:
        1. A neural network (`net`) predicting displacements and stresses.
        2. A PDE wrapper (`pde`) that computes residuals from those predictions.
        3. A geometry helper (`geom`) defining collocation and boundary samples.
        4. Logging, training loops, and boundary condition enforcement.
        
    Attributes:
        device (str): Torch device tag (e.g., "cpu" or "cuda:0").
        net (nn.Module): Neural network for mapping (x,y,z)->[u,v,w,σ_xx,…].
        pde (PDE_Graph): PDE helper with methods for strain, stress, and residuals.
        geom (Layered_Contact_Geometry): Geometry helper for sampling points.
        num_domain (int): Number of interior collocation points per layer.
        num_bc (int): Number of boundary points per face per layer.
        num_points (int): Total points per forward pass (domain + boundaries).
        proj (str): Project name used for logging directories.
        logger (logging.Logger): Configured logger for training output.
    """
    def __init__(self, net: nn.Module, pde: PDE_Graph, geom: Layered_Contact_Geometry,
                num_domain: int, num_bc: int, device: str, proj: str = 'graph') -> None:
        """Initialize the PINN with model, physics, geometry, and sampling counts.
        
        Args:
            net (nn.Module): Network that takes (x, y, z) and returns
                [u, v, w, σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_zx].
            pde (PDE_Graph): PDE instance providing methods to compute
                residuals (strain→stress, equilibrium, constitutive coupling).
            geom (Layered_Contact_Geometry): Geometry instance for generating
                collocation and boundary point sets per layer.
            num_domain (int): Number of interior (collocation) points per layer.
            num_bc (int): Number of boundary points per face per layer.
            device (str): Device string ("cpu" or "cuda:0") on which to place
                the model and all tensors.
            proj (str): Project name for log and output directories.
        """
        self.device = device
        self.net  = net.to(self.device)
        self.pde  = pde
        self.geom = geom
        self.num_domain = num_domain
        self.num_bc = num_bc
        self.num_points = (num_domain**3 + num_bc**2 * 8) * geom.n_layers
        
        self.proj = proj
        
        os.makedirs(f'log/{self.proj}/log', exist_ok=True)
        os.makedirs(f'log/{self.proj}/weights', exist_ok=True)
        os.makedirs(f'log/{self.proj}/paraview', exist_ok=True)
        os.makedirs(f'log/{self.proj}/loss', exist_ok=True)
        self.logger = setup_logger(name=self.__class__.__name__, log_dir=f'log/{self.proj}/log')
        
        self.logger.debug(f"Using device: {device}")
        self.logger.info("Geometry parameters: lx=%.3f, ly=%.3f, lz=%.3f", self.geom.lx, self.geom.ly, self.geom.lz)
        self.logger.info("NN architecture layers: %s", net)  # or print net layers list
        self.logger.info("PDE material: E=%.3f, nu=%.3f", pde.E, pde.nu)
        
        self.count_parameters()
    
    def generate_points(self, layer: int, method: str, seed: int, k: int) -> None:
        """Generate and store collocation and boundary graphs for a given layer.
        
        This method samples interior and boundary points on the specified layer,
        constructs k‑NN graphs over each point set, and stores them (with
        `requires_grad=True` on `.x`) as attributes on the PINN instance:
        
            - inside_points : Data with (N_domain**3, 3) interior points  
            - top_points    : Data with (N_bc**2, 3) points on the top face  
            - bottom_points : Data with (N_bc**2, 3) points on the bottom face  
            - front_points  : Data with (N_bc**2, 3) points on the front face  
            - back_points   : Data with (N_bc**2, 3) points on the back face  
            - right_points  : Data with (N_bc**2, 3) points on the right face  
            - left_points   : Data with (N_bc**2, 3) points on the left face  
            - domain_points : Data combining all of the above  

        Args:
            layer (int): Index of the layer to sample (0 ≤ layer < n_layers).
            method (str): Sampling strategy ('linspace', 'uniform', or 'lhs').
            seed (int): RNG seed for reproducibility.
            k (int): Number of nearest neighbors per node for the k‑NN graph.
        """
        # generate collocation / BC points
        geo = self.geom.standard_layers[layer]
        self.inside_points = geo.generate_inside_graph(
                N=self.num_domain, method=method, seed=seed, k=k).to(self.device)
        self.inside_points.x = self.inside_points.x.requires_grad_(True)
        self.top_points = geo.generate_top_graph(
                N=self.num_bc, method=method, seed=seed, k=k).to(self.device)
        self.top_points.x = self.top_points.x.requires_grad_(True)
        self.bottom_points = geo.generate_bottom_graph(
                N=self.num_bc, method=method, seed=seed, k=k).to(self.device)
        self.bottom_points.x = self.bottom_points.x.requires_grad_(True)
        self.front_points = geo.generate_front_graph(
                N=self.num_bc, method=method, seed=seed, k=k).to(self.device)
        self.front_points.x = self.front_points.x.requires_grad_(True)
        self.back_points = geo.generate_back_graph(
                N=self.num_bc, method=method, seed=seed, k=k).to(self.device)
        self.back_points.x = self.back_points.x.requires_grad_(True)
        self.right_points = geo.generate_right_graph(
                N=self.num_bc, method=method, seed=seed, k=k).to(self.device)
        self.right_points.x = self.right_points.x.requires_grad_(True)
        self.left_points = geo.generate_left_graph(
                N=self.num_bc, method=method, seed=seed, k=k).to(self.device)
        self.left_points.x = self.left_points.x.requires_grad_(True)
        
        
        self.domain_points = self.geom.combine_graphs(graph_list=[self.inside_points, self.bottom_points, self.top_points,
                self.front_points, self.back_points, self.right_points, self.left_points])
        # --- logging ---
        if seed == 1:
            self.logger.debug("Point counts: inside=%d, top=%d, bottom=%d, front=%d, back=%d, right=%d, left=%d",
                    self.inside_points.x.shape[0],
                    self.top_points.x.shape[0],
                    self.bottom_points.x.shape[0],
                    self.front_points.x.shape[0],
                    self.back_points.x.shape[0],
                    self.right_points.x.shape[0],
                    self.left_points.x.shape[0],
                    )
            self.logger.debug("Inside points shape: %s", tuple(self.inside_points.x.shape))
            self.logger.debug("Top points shape:    %s", tuple(self.top_points.x.shape))
            self.logger.debug("Bottom points shape: %s", tuple(self.bottom_points.x.shape))
            self.logger.debug("Front points shape:  %s", tuple(self.front_points.x.shape))
            self.logger.debug("Back points shape:   %s", tuple(self.back_points.x.shape))
            self.logger.debug("Right points shape:  %s", tuple(self.right_points.x.shape))
            self.logger.debug("Left points shape:   %s", tuple(self.left_points.x.shape))
            
            self.logger.debug("Inside edge shape: %s", tuple(self.inside_points.edge_index.shape))
            self.logger.debug("Top edge shape:    %s", tuple(self.top_points.edge_index.shape))
            self.logger.debug("Bottom edge shape: %s", tuple(self.bottom_points.edge_index.shape))
            self.logger.debug("Front edge shape:  %s", tuple(self.front_points.edge_index.shape))
            self.logger.debug("Back edge shape:   %s", tuple(self.back_points.edge_index.shape))
            self.logger.debug("Right edge shape:  %s", tuple(self.right_points.edge_index.shape))
            self.logger.debug("Left edge shape:   %s", tuple(self.left_points.edge_index.shape))
    
    def add_pressure_to_input_nodes(self, points: Data, pressure: float) -> Data:
        bc = torch.tensor([0.0, 0.0, pressure])
        bcs = bc.repeat((points.x.shape[0], 1))
        points.x = torch.cat((points.x, bcs), dim=1)
        points.x.require_grad_(True)
        return points
        
    def calculate_loss(self, epoch: int, bc: Contact_Boundary_Condition) -> torch.Tensor:
        """Compute the total loss combining PDE residuals and boundary conditions.
        
        This method computes:
            1. PDE loss: mean-square error of each mixed-formulation residual over
                the interior collocation points.
            2. Boundary loss: mean-square error enforcing traction or displacement
                conditions on each face of the current layer.
            
        Args:
            epoch: Current training epoch (used for logging).
            bc:    Contact_Boundary_Condition providing pressure and normal-gap logic.
            
        Returns:
            A scalar tensor representing the sum of the PDE loss and all boundary losses.
        """
        loss_fn = nn.MSELoss()
        # PDE loss
        # domain_points = self.add_pressure_to_input_nodes(points=self.domain_points, pressure=bc.pressure)
        outputs_dom = self.net(x=self.domain_points.x, edge_index=self.domain_points.edge_index)  
        pde_terms = self.pde.pde_mixed_central_diff(x=self.domain_points.x,
                edge_index=self.domain_points.edge_index, outputs=outputs_dom)
        # Zero target (we want each PDE residual to be zero)
        targets = [torch.zeros_like(R) for R in pde_terms]
        # Compute MSE loss for each residual
        loss_pde = sum(loss_fn(R, target) for R, target in zip(pde_terms, targets))
        
        if epoch % 50 == 0:
            # self.logger (MSE + original mean for monitoring)
            pde_mse_vals = [f"{loss_fn(R, target).item():.3e}" for R, target in zip(pde_terms, targets)]
            pde_str = ', '.join([f"{i+1}: {mse}" for i, mse in enumerate(pde_mse_vals)])
            self.logger.debug(f"Epoch {epoch} PDE terms: [{pde_str}]")
        
        # boundary losses
        boundary_sets = {
            'top':    self.top_points,
            'front':  self.front_points,
            'back':   self.back_points,
            'right':  self.right_points,
            'left':   self.left_points,
            'bottom': self.bottom_points,
        }
        loss_boundaries = {}
        for name, pts in boundary_sets.items():
            # pts = self.add_pressure_to_input_nodes(points=pts, pressure=bc.pressure)
            out = self.net(x=pts.x, edge_index=pts.edge_index)  # [N, 5]
            u, v, w, *_ = out.unbind(dim=1)
            sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_zx = self.pde.calculate_stress_tensor(x=pts.x, edge_index=pts.edge_index)
            
            terms = []
            targets = []
            labels = []
            if name == 'top':
                # sigma_zz = pressure, sigma_yz = 0, sigma_zx = 0
                terms   = [sigma_zz, sigma_yz, sigma_zx]
                targets = [bc.pressure.expand_as(sigma_zz), torch.zeros_like(sigma_yz), torch.zeros_like(sigma_zx)]
                labels  = ['szz=pressure', 'syz=0', 'szx=0']
            elif name == 'front':
                # sigma_xx = 0, sigma_xy = 0, sigma_zx = 0
                terms   = [sigma_xx, sigma_xy, sigma_zx]
                targets = [torch.zeros_like(sigma_xx), torch.zeros_like(sigma_xy), torch.zeros_like(sigma_zx)]
                labels  = ['sxx=0', 'sxy=0', 'szx=0']
            elif name == 'back':
                # u = 0, sigma_xy = 0, sigma_zx = 0
                terms   = [u, sigma_xy, sigma_zx]
                targets = [torch.zeros_like(u), torch.zeros_like(sigma_xy), torch.zeros_like(sigma_zx)]
                labels  = ['u=0', 'sxy=0', 'szx=0']
            elif name == 'right':
                # sigma_yy = 0, sigma_xy = 0, sigma_yz = 0
                terms   = [sigma_yy, sigma_xy, sigma_yz]
                targets = [torch.zeros_like(sigma_yy), torch.zeros_like(sigma_xy), torch.zeros_like(sigma_yz)]
                labels  = ['syy=0', 'sxy=0', 'syz=0']
            elif name == 'left':
                # v = 0, sigma_xy = 0, sigma_yz = 0
                terms   = [v, sigma_xy, sigma_yz]
                targets = [torch.zeros_like(v), torch.zeros_like(sigma_xy), torch.zeros_like(sigma_yz)]
                labels  = ['v=0', 'sxy=0', 'syz=0']
            elif name == 'bottom': # contact
                out_c = self.net(x=pts.x, edge_index=pts.edge_index)
                t1 = bc.calculate_tangential_traction_component1(inputs=pts.x, outputs=out_c)
                t2 = bc.calculate_tangential_traction_component2(inputs=pts.x, outputs=out_c)
                gn = bc.calculate_complementarity_based_fisher_burmeister(inputs=pts.x, outputs=out_c)
                # gn = 0, t1 = 0, t2 = 0
                terms   = [gn, t1, t2]
                targets = [torch.zeros_like(gn), torch.zeros_like(t1), torch.zeros_like(t2)]
                labels  = ['gn=0', 't1=0', 't2=0']
            # MSE loss on each term
            face_loss = sum(loss_fn(term, target) for term, target in zip(terms, targets))
            loss_boundaries[name] = face_loss
            if epoch % 50 == 0:
                # Logging
                loss_details = [f"{label}: {loss_fn(term, target).item():.3e}" for term, target, label in zip(terms, targets, labels)]
                loss_str = ', '.join(loss_details)
                self.logger.debug(f"Epoch {epoch:3d} boundary {name:6s} loss: {face_loss.item():.3e} | {loss_str}")
        # Total boundary loss
        loss_bcs = sum(loss_boundaries.values())
        
        # — total loss —
        loss = loss_pde + loss_bcs
        if epoch % 50 == 0:
            self.logger.debug(
                "Epoch %d TOTALS -> PDE: %.3e, Bcs: %.3e, LOSS: %.3e",
                epoch, loss_pde.item(), loss_bcs.item(), loss.item()
            )
        self.loss_recorder.append([epoch, loss_pde.item(), loss_bcs.item(), loss.item()])
        return loss
    
    def train_using_Adam(self, layer: int, num_epochs: int, bc: Contact_Boundary_Condition,
                step: int, lr: float = 1e-3):
        """Train the network for a single layer using the Adam optimizer.
        
        During each epoch:
            1. Resample interior and boundary points via Latin-hypercube sampling.
            2. Compute the combined PDE + BC loss.
            3. Step the Adam optimizer.
            4. Optionally checkpoint every 1 000 000 epochs.
            
        Args:
            layer:      Index of the current layer being trained (0 ≤ layer < n_layers).
            num_epochs: Total number of epochs to run.
            bc:         Contact_Boundary_Condition providing boundary-pressure data.
            lr:         Learning rate for the Adam optimizer (default 1e-3).
            
        Side effects:
            - Updates `self.net` weights in place.
            - Saves model checkpoints under `log/weights/`.
            - Logs training progress and final checkpoint.
        """
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.logger.debug(
            f"Starting Adam training for layer {layer} "
            f"({num_epochs} epochs, lr={lr:.1e})"
        )
        for epoch in range(1, num_epochs + 1):
            self.generate_points(layer=layer, method='lhs', seed=epoch, k=8)
            self.net.train()
            optimizer.zero_grad()
            loss = self.calculate_loss(epoch=epoch, bc=bc)
            loss.backward()
            optimizer.step()
            
            if epoch % 1_000_000 == 0:
                torch.save(self.net.state_dict(),
                        f"log/{self.proj}/weights/Step{step}_layer{layer}_weight_{epoch}.pth")
                self.logger.debug(f"Saved weights to log/{self.proj}/weights/Step{step}_layer{layer}_weight_{epoch}.pth")
        # final save
        torch.save(self.net.state_dict(),
                f"log/{self.proj}/weights/Step{step}_layer{layer}_weight.pth")
        self.logger.debug(f"Saved weights to log/{self.proj}/weights/Step{step}_layer{layer}_weight.pth")
    
    def train_using_LBFGS(self, layer: int, num_epochs: int, bc: Contact_Boundary_Condition,
                step: int):
        """Fine-tune the network for one layer using the L-BFGS optimizer.
        
        This routine alternates:
            1. Resampling collocation and boundary points (single draw at seed=0).
            2. Evaluating the PDE+BC loss via a closure.
            3. A full L-BFGS update.
            
        Args:
            layer:      Index of the current layer being trained.
            num_epochs: Number of outer iterations (calls to .step).
            bc:         Contact_Boundary_Condition with pressure data.
            
        Side effects:
            - Updates `self.net` in place.
            - Saves a final checkpoint under `log/weights/`.
            - Logs progress at the start and end.
        """
        optimizer = torch.optim.LBFGS(self.net.parameters(),
                lr=1, max_iter=50000, max_eval=50000, history_size=50,
                tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn='strong_wolfe')
        self.logger.debug(f"Switching to L-BFGS optimization...")
        for epoch in range(1, num_epochs + 1):
            self.generate_points(layer=layer, method='lhs', seed=0, k=8)
            def closure():
                optimizer.zero_grad()
                loss = self.calculate_loss(epoch=epoch, bc=bc)
                loss.backward()
                return loss
            optimizer.step(closure)
        torch.save(self.net.state_dict(), f"log/{self.proj}/weights/Step{step}_layer{layer}_weight.pth")
        self.logger.debug(f"Saved weights to log/{self.proj}/weights/Step{step}_layer{layer}_weight.pth")
    
    def train(self, pressure: float, num_epochs: int, step: int):
        """Sequentially train each layer of the PINN with contact pressure updates.
        
        For each layer (from top to bottom):
            1. Reinitialize network weights.
            2. Create a Contact_Boundary_Condition with the current `pressure`.
            3. Log the applied pressure.
            4. Train with Adam for `num_epochs` epochs.
            5. Refine with one iteration of L-BFGS.
            6. Evaluate the bottom-face σzz to update the pressure for the next layer.
            7. Log the computed top-corner displacement for monitoring.
            
        Args:
            pressure:     Initial contact pressure (positive in compression).
            num_epochs:   Number of Adam epochs to run per layer.
            
        Side effects:
            - Updates `self.net` weights layer by layer.
            - Logs training progress, pressures, and displacements.
            - Overwrites `pressure` each iteration with the mean σzz at the bottom.
        """
        N = 16
        xy = torch.linspace(0.0, 1.0, N, device=self.device)
        z = torch.ones(1, device=self.device)
        top_points = torch.cartesian_prod(xy, xy, z)
        edge_index = knn_graph(x=top_points, k=8, loop=False)
        data_top = Data(x=top_points, edge_index=edge_index).to(self.device)
        
        for i in range(self.geom.n_layers):
            # self.net.initiate_weights()
            self.loss_recorder = []
            layer = self.geom.n_layers - (i + 1)
            bc = Contact_Boundary_Condition(geom=self.geom.standard_layers[layer], pressure=pressure)
            self.logger.info("Contact BC pressure: %.3f from layer %d", bc.pressure, layer+1)
            
            self.train_using_Adam(layer=layer, num_epochs=num_epochs, bc=bc, step=step)
            # self.train_using_LBFGS(layer=layer, num_epochs=1, bc=bc, step=step)
            self.net.eval()
            with torch.no_grad():
                sig_z = self.net(x=self.bottom_points.x, edge_index=self.bottom_points.edge_index)[:, 5]
            pressure = sig_z.mean().item()
            
            with torch.no_grad():
                disp1 = self.net(x=data_top.x, edge_index=data_top.edge_index)[-1, :3]
                self.logger.debug("Layer %d top-corner displacement: %s", layer + 1, disp1.tolist())
            
            np.savetxt(f'log/graph/loss/layer{i}.csv', np.array(self.loss_recorder), delimiter=',',
                    header='epoch, loss_pde, loss_bcs, total_loss')
    
    def predict_vertical_displacements(self, step: int) -> List[float]:
        """Compute the mean vertical displacement at the top of each layer.
        
        For each layer (0-indexed from bottom to top), this method:
            1. Generates N_bc² normalized points on the layer's top face using linspace.
            2. Loads the corresponding trained weights for that layer.
            3. Scales those normalized points back to physical coordinates.
            4. Computes the original displacement field at those points.
            5. Takes the mean of the w-component (vertical) displacement.
            
        Returns:
            A list of length `n_layers + 1` where:
                - disp[0] == 0.0 (base reference).
                - disp[i+1] is the mean vertical displacement at the top of layer i.
        """
        disp = [0.0]
        for i in range(self.geom.n_layers):
            single_geom = self.geom.standard_layers[i]
            top_points_norm = single_geom.generate_top_graph(N=self.num_bc, method='linspace', seed=0, k=8)
            weight_path = f"log/{self.proj}/weights/Step{step}_layer{i}_weight.pth"
            state = torch.load(weight_path, map_location=self.device)
            self.net.load_state_dict(state)
            
            scaler = Scaler_Graph(net=self.net, mins=self.geom.layer_mins[i], maxs=self.geom.layer_maxs[i])
            top_points = scaler.scale_back_from_standard_coordinates(x_norm=top_points_norm.x)
            d_scaled = scaler.calculate_original_displacement(x=top_points)
            d = d_scaled[:, 2].mean()
            disp.append(d)
        return disp
    
    def predict(self, x: torch.Tensor, edge_index: torch.Tensor,
                E: torch.Tensor, step: int) -> torch.Tensor:
        """Predict displacements, stresses, and strains at arbitrary points.
        
        For each input point in `x` this method:
            1. Determines which printed layer the point falls into.
            2. Loads that layer's trained network weights.
            3. Computes the original displacement `[u, v, w]`, stress components
                `[σₓₓ, σᵧᵧ, σ𝓏𝓏, σₓᵧ, σᵧ𝓏, σ𝓏ₓ]`, and small-strain components
                `[εₓₓ, εᵧᵧ, ε𝓏𝓏, εₓᵧ, εᵧ𝓏, εₓ𝓏]`.
            4. Adds the cumulative vertical offset from all layers below to `w`.
            5. Assembles a single output tensor of shape `(N, 15)`.
            
        Args:
            x: FloatTensor of shape `(N, 3)`, the physical coordinates to evaluate.
            
        Returns:
            FloatTensor of shape `(N, 15)`, with columns in the order:
            ```
            [u, v, w_total,
            σₓₓ, σᵧᵧ, σ𝓏𝓏,
            σₓᵧ, σᵧ𝓏, σ𝓏ₓ,
            εₓₓ, εᵧᵧ, ε𝓏𝓏,
            εₓᵧ, εᵧ𝓏, εₓ𝓏]
            ```
        """
        # assign each point to a layer index in [0, n_layers)
        layer_ids = self.geom._calculate_layer_indices(x=x)
        N = x.shape[0]
        # prepare output placeholder
        preds = torch.zeros([N, 15], device=self.device)
        # precompute vertical offsets at each layer interface
        vertical_disp = self.predict_vertical_displacements(step=0)
        # evaluate each unique layer
        for layer in torch.unique(layer_ids):
            mask = (layer_ids == layer)
            if not mask.any():
                continue
            # points in this layer
            x_layer = x[mask]
            edge_index_layer = knn_graph(x=x_layer, k=8, loop=False)
            # load & eval
            weight_path = f"log/{self.proj}/weights/Step0_layer{layer}_weight.pth"
            state = torch.load(weight_path, map_location=self.device)
            self.net.load_state_dict(state)
            
            # build a scaler for this layer's physical bounds
            scaler = Scaler_Graph(net=self.net, mins=self.geom.layer_mins[layer], maxs=self.geom.layer_maxs[layer])
            # compute original displacement & stress
            disp = scaler.calculate_original_displacement(x=x_layer)
            stress = scaler.calculate_original_stress(x=x_layer, edge_index=edge_index_layer)
            strain = scaler.calculate_original_strain(x=x_layer, edge_index=edge_index_layer)
            out_layer = torch.cat([disp, stress, strain], dim=1)
            # add cumulative vertical shift from all underlying layers
            out_layer[:, 2] += sum(vertical_disp[:layer+1])
            # write them back into preds
            preds[mask] = out_layer
        preds = self.calculate_physical_prediction(prediction=preds, E=E)
        return preds
    
    def calculate_physical_prediction(self, prediction: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """Convert PINN outputs from nondimensional to physical units.
        
        The PINN is trained with a normalized Young's modulus E=1.0 to improve
        numerical stability. To recover physical displacements and strains, we
        divide those quantities by the true Young's modulus.
        
        Args:
            prediction (torch.Tensor):
                FloatTensor of shape (N, 15) returned by `predict()`. Columns are
                ordered as
                ```
                [u, v, w_total,
                sxx, syy, szz,
                sxy, syz, szx,
                εₓₓ, εᵧᵧ, ε𝓏𝓏,
                εₓᵧ, εᵧ𝓏, εₓ𝓏]
                ```
                and are all expressed in the same nondimensional units used
                during PINN training.
            E (float):
                The physical Young's modulus. Used to scale displacements (first
                3 columns) and strains (last 6 columns) back to physical units.
                
        Returns:
            torch.Tensor:
                The same `prediction` tensor (shape `(N, 15)`), but with
                - `prediction[:, :3]` (u, v, w) divided by `E`, and
                - `prediction[:, 9:]` (εₓₓ…εₓ𝓏) divided by `E`.
                Stress components (`σₓₓ…σₓ𝓏`) in columns 3–8 remain unchanged.
        """
        if E.dim() == 1:
            E = E.unsqueeze(1)
        # Scale displacements back to physical units
        prediction[:, :3] /= E
        # Scale strains back to physical units
        prediction[:, 9:15] /= E
        return prediction
    
    def save_to_vtu(self) -> None:
        """Generate a 3D mesh of the layered domain and export displacement/stress fields to VTU.
        
        This method:
            1. Uses gmsh to build and mesh stacked boxes for each layer.
            2. Reads the mesh with meshio to get node coordinates and cell connectivity.
            3. Predicts u, v, w, and stresses sxx…szx at each node.
            4. Writes separate VTU files for each field component under `log/graph/paraview/`.
            
        Side effects:
            - Writes `mesh.msh` for gmsh and multiple `.vtu` files for ParaView.
            - Logs each file write.
        """
        import gmsh
        import meshio
        # 1) Initialize gmsh and build the layered boxes
        gmsh.initialize()
        gmsh.model.add("layered_boxes")
        for i in range(self.geom.n_layers):
            bbox = self.geom.bboxes[i].cpu().numpy()
            x_min, y_min, z_min = bbox[0]
            x_max, y_max, z_max = bbox[1]
            gmsh.model.occ.addBox(x_min, y_min, z_min, x_max - x_min, y_max - y_min, z_max - z_min)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        # write to disk so meshio can read
        gmsh.write(f'log/{self.proj}/paraview/mesh.msh')
        gmsh.finalize()
        # 2) Read mesh with meshio
        m = meshio.read(f'log/{self.proj}/paraview/mesh.msh')
        points = m.points               # shape (Nnodes, 3)
        cells  = m.cells                # list of (cell_type, indices) tuples
        # 3) Predict fields at each node
        pts_torch = torch.from_numpy(points.astype(np.float32)).to(self.device)
        edge_index = knn_graph(x=pts_torch, k=8, loop=False)
        data = Data(x=pts_torch, edge_index=edge_index)
        pred = self.predict(data.x, data.edge_index).cpu().numpy()    # (Nnodes,15)
        # pred = self.calculate_physical_prediction(prediction=pred, E=E)
        
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
            vtu_path = f"log/{self.proj}/paraview/{name}.vtu"
            meshio.write(vtu_path, vtk)
            self.logger.debug(f"Wrote {vtu_path}")
    
    def count_parameters(self) -> int:
        num_parameters = count_parameters(model=self.net)
        self.logger.info(f'Number of parameters: {num_parameters}')
        return num_parameters

