# dl/graph/explainer.py


import torch
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.explain import Explanation, Explainer, GNNExplainer


from fem.config_loader import load_config
from fem.fem.materials import Quadrilinear_Concrete

from dl.graph.pde import PDE_Graph
from dl.graph.pinn import PINN_Graph
from dl.pinn.geometry import Layered_Contact_Geometry
from dl.pinn.networks import GraphSAGE

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    })


class Graph_Explainer:
    def __init__(self):
        # units: mm, N, MPa, ton, second
        config = 'configs/config.yaml'
        conc_prop, dmg_para, geo_para, mesh_para, contact_para = load_config(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Geometry
        geom = Layered_Contact_Geometry(para=geo_para, device=device)
        # Network
        net = GraphSAGE(input_dim=3, hidden_dim=3, output_dim=9)
        # PDE
        mat = Quadrilinear_Concrete(prop=conc_prop)
        pde = PDE_Graph(net=net, E=mat.prop.E_pinn, nu=mat.prop.nu)
        # PINN
        pinn = PINN_Graph(net=net, pde=pde, geom=geom, num_domain=8, num_bc=16, device=device)
        pinn.generate_points(layer=10, method='lhs', seed=1, k=8)
        self.pinn = pinn
        self.pinn_explanation = PINN_Graph(net=net, pde=pde, geom=geom, num_domain=8, num_bc=3, device=device)

        model = pinn.net
        layer = 10
        weight_path = f"log/graph/3x2/weights/Step0_layer{layer}_weight.pth"
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
        self.explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='regression',
                task_level='node',
                return_type='raw',
            ),
        )
    
    def calculate_feature_importance(self, node_index: int, data: Data, path: str = '') -> torch.Tensor:
        explanation = self.explainer(data.x, data.edge_index, index=node_index)
        print(f'Generated explanations in {explanation.available_explanations}')
        
        # ---- get feature importance values directly
        feat_imp = explanation.node_mask.mean(dim=0)
        
        # Debug: Check shapes
        num_features = data.x.shape[1]  # Number of features per node
        print(f"Number of features in data: {num_features}")
        print(f"Feature importance shape: {feat_imp.shape}")
        print(f"Feature importance values: {feat_imp}")
        
        # (Optional) normalize for readability
        feat_imp_norm = feat_imp / (feat_imp.sum() + 1e-12)
        print("Normalized feature importance:", feat_imp_norm)
        
        if path:
            print('Node features:', data.x[node_index, :])
            explanation.visualize_feature_importance(path, top_k=10)
            print(f"Feature importance plot has been saved to '{path}'")
            
        return feat_imp_norm
    
    def analyze_nodes_feature_importance(self, data: Data, pos: str, normalize=True) -> tuple[torch.Tensor, torch.Tensor]:
        fi_list = []
        for node_idx in range(data.num_nodes):
            # no per-node plot to keep things fast; pass "" to skip plotting
            fi = self.calculate_feature_importance(
                node_index=node_idx, data=data, path=""
            )
            # keep as torch tensor
            fi_list.append(fi.detach().cpu())

        # Stack into [N_nodes, F] tensor
        M = torch.stack(fi_list, dim=0)  # shape [N, F]
            
        # Save in log/graph/
        np.savetxt(f"log/graph/feature_importance/feature_importance_nodes_{pos}.csv", M.numpy(), delimiter=",")
        
        mean_imp= M.mean(dim=0)
        std_imp = M.std(dim=0, correction=0)  # ddof=0 equivalent

        # 4) Report global top-k features
        F = mean_imp.shape[0]
        k_print = min(10, F)
        vals, idx = torch.topk(mean_imp, k=k_print)
        print(f"\n=== Global Top-{k_print} Features on {pos} nodes ===")
        for r, (i, v) in enumerate(zip(idx.tolist(), vals.tolist()), 1):
            print(f"{r:>2}. feature {i:<3d} mean={v:.4f}  std={std_imp[i].item():.4f}")

        # 5) Save summary CSV (NumPy only)
        F = mean_imp.numel()
        feature_idx = np.arange(F, dtype=np.int64)[:, None]
        mean_np = mean_imp[:, None].numpy()
        std_np  = std_imp[:, None].numpy()
        table = np.hstack([feature_idx, mean_np, std_np])
        np.savetxt(
            f'log/graph/feature_importance/feature_importance_nodes_summary_{pos}.csv',
            table,
            delimiter=",",
            header="feature_index,mean_importance,std_importance",
        )
        
        # 6) Box-and-whisker plot (each box = a feature’s distribution across nodes)
        M_np = M.numpy()  # [N, F]
        plt.figure(figsize=(10, 4))
        plt.boxplot([M_np[:, j] for j in range(M_np.shape[1])], showmeans=True)
        plt.xlabel("Feature index")
        plt.ylabel("Importance" + (" (normalized)" if normalize else ""))
        plt.title(f"Feature importance across {pos} nodes")
        plt.tight_layout()
        plt.savefig(f'log/graph/feature_importance/fi_box_plot_{pos}.pdf', dpi=300, bbox_inches="tight")
        plt.close()
        
        return mean_imp, std_imp
    
    
    def plot_subgraph(self, node_index: int, data: Data, path: str) -> None:
        explanation = self.explainer(data.x, data.edge_index, index=node_index)
        print(f'Generated explanations in {explanation.available_explanations}')
        explanation.visualize_graph(path)
        print(f"Subgraph visualization plot has been saved to '{path}'")
        
    def plot_pruned_subgraph(self, node_index: int, data: Data, path: str, K: int = 15) -> None:
        import matplotlib as mpl
        mpl.rcParams["font.family"] = "Times New Roman"
        mpl.rcParams["font.size"] = 12

        explanation = self.explainer(data.x, data.edge_index, index=node_index)
        edge_mask  = explanation.edge_mask          # [E_sub]
        edge_index = explanation.edge_index         # [2, E_sub]
        
        # --- pick top-K edges
        vals, order = torch.sort(edge_mask, descending=True)
        order = order[:min(K, edge_mask.numel())]
        
        # pruned edges
        ei_pruned = edge_index[:, order]            # [2, kE]
        em_pruned = edge_mask[order]                # [kE]
        
        # --- find the node subset that remains after pruning
        subset = torch.unique(ei_pruned).sort()[0]  # [kN] (original ids inside the explanation)
        old2new = {int(n): i for i, n in enumerate(subset.tolist())}
        
        # relabel edges to 0..kN-1
        ei_pruned = ei_pruned.clone().cpu()
        ei_pruned[0] = torch.tensor([old2new[int(u)] for u in ei_pruned[0].tolist()], dtype=torch.long)
        ei_pruned[1] = torch.tensor([old2new[int(v)] for v in ei_pruned[1].tolist()], dtype=torch.long)
        
        # slice x / (optional) node_mask if present
        x_pruned = explanation.x[subset].cpu() if explanation.x is not None else None
        nm_pruned = (explanation.node_mask[subset].cpu()
                    if getattr(explanation, "node_mask", None) is not None else None)
        
        # reassemble a minimal Explanation on CPU
        explanation_pruned = Explanation(
            x=x_pruned,
            edge_index=ei_pruned,
            edge_mask=em_pruned.cpu(),
            node_mask=nm_pruned,
        )
        
        explanation_pruned.visualize_graph(path)
        print(f"Pruned subgraph saved to: {path}")
        
        # Get current figure and axis
        fig = plt.gcf()
        ax = plt.gca()
        
        # Method 1: Remove all spines (frame)
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Method 2: Alternative way to remove frame
        ax.set_frame_on(False)
        
        # Remove ticks and tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Remove axis labels if any
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Customize node labels (text elements)
        for text in ax.texts:
            text.set_fontsize(14)  # Set font size for node labels
            text.set_fontfamily("Times New Roman")
            text.set_fontweight('bold')  # Make text bold if desired
            text.set_color('black')  # Set text color
        
        # Customize arrows/edges if using matplotlib patches
        # This depends on how visualize_graph() creates the arrows
        for patch in ax.patches:
            if hasattr(patch, 'set_arrowstyle'):
                # For FancyArrowPatch objects
                patch.set_arrowstyle('->')  # or '-|>', '<->', etc.
                patch.set_mutation_scale(20)  # Arrow head size
                patch.set_linewidth(2)  # Arrow line width
                patch.set_color('gray')  # Arrow color
            elif hasattr(patch, 'set_linewidth'):
                # For other patch objects
                patch.set_linewidth(2)
        
        # If using Line2D objects for edges (common in networkx)
        for line in ax.lines:
            line.set_linewidth(2.5)  # Set edge width
            line.set_color('darkgray')  # Set edge color
            line.set_alpha(0.7)  # Set transparency
        
        # Alternative: If you want to completely customize the arrow style
        # and the library allows it, you might need to recreate arrows manually
        
        # Adjust layout to remove extra whitespace
        plt.tight_layout()
        
        # Save with custom settings
        plt.savefig("log/graph/subgraph_custom.pdf", 
                    bbox_inches="tight",  # Remove extra whitespace
                    pad_inches=0.1,       # Minimal padding
                    dpi=300,              # High resolution
                    facecolor='white',    # Background color
                    edgecolor='none')     # No edge color for figure
        
        plt.close()
    
    def plot_subgraph_test2(self) -> None:
        import networkx as nx
        from torch_geometric.utils import to_networkx
        
        self.pinn_explanation.generate_points(layer=0, method='lhs', seed=1, k=8)
        data = self.pinn_explanation.top_points
        coordinates = data.x.detach().cpu().numpy()
        np.savetxt('log/graph/top_surface_coordinates.csv', coordinates, delimiter=',', header='x, y, z')
        
        # Choose a target node index
        node_index = 8

        # Generate explanation
        explanation = self.explainer(
            x=data.x,
            edge_index=data.edge_index,
            index=node_index
        )

        # Threshold for important edges (adjust as needed)
        edge_mask = explanation.edge_mask > 0.5
        sub_edge = data.edge_index[:, edge_mask]

        # explanation = self.explainer(
        #     x=data.x,
        #     edge_index=sub_edges,
        #     index=node_index
        # )
        
        # Create subgraph
        G = to_networkx(
            explanation,
            # node_attrs=['x', 'y', 'z'],  # Adjust attributes as needed
            edge_attrs=['edge_mask'],
            to_undirected=False  # Set True if graph is undirected
        )

        # Plotting
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12
        fig, ax = plt.subplots(figsize=(10, 9))  # Adjust figure size

        pos = nx.spring_layout(G)  # or use another layout algorithm

        # Draw nodes and edges
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=list(G.edges),
            edge_color='gray',
            arrows=True,  # Set to False for no arrows
            arrowstyle='-|>',  # Customize arrow style: '->', '-|>', '<->', etc.
            arrowsize=10,  # Adjust arrow size
            node_size=500
        )

        # Labels
        nx.draw_networkx_labels(G, pos, font_family='Times New Roman', font_size=12)
        edge_labels = {(u, v): f'{explanation.edge_mask[i]:.2f}'
                    for i, (u, v) in enumerate(G.edges)}
        nx.draw_networkx_edge_labels(G, pos,
                font_family='Times New Roman', edge_labels=edge_labels, font_size=10, label_pos=0.8)

        # plt.title(f'Explanation for Node {node_index}')
        plt.axis('off')
        plt.tight_layout()
        # plt.show()
        
        fig.savefig('log/graph/subgraph.pdf', dpi=300)

    def plot_subgraph_test(self) -> None:
        import numpy as np
        import networkx as nx
        import matplotlib.pyplot as plt
        from torch_geometric.utils import to_networkx

        # -----------------------------
        # 1) Prepare data (same as before)
        # -----------------------------
        self.pinn_explanation.generate_points(layer=0, method='lhs', seed=1, k=8)
        data = self.pinn_explanation.top_points
        np.savetxt(
            'log/graph/top_surface_coordinates.csv',
            data.x.detach().cpu().numpy(),
            delimiter=',', header='x, y, z'
        )

        node_index = 8      # focal node
        THRESH = 0.50       # edge-importance threshold

        # -----------------------------
        # 2) Run explainer on the full graph
        # -----------------------------
        explanation = self.explainer(
            x=data.x,
            edge_index=data.edge_index,
            index=node_index
        )

        # -----------------------------
        # 3) Convert to NetworkX and keep edge scores
        # -----------------------------
        G_full = to_networkx(
            explanation,
            edge_attrs=['edge_mask'],
            to_undirected=False
        )

        # -----------------------------
        # 4) Filter: only edges incident to node_index AND >= THRESH
        # -----------------------------
        incident_edges = []
        touched = {node_index}
        for u, v, d in G_full.edges(data=True):
            score = float(d.get('edge_mask', 0.0))
            if score >= THRESH and (u == node_index or v == node_index):
                incident_edges.append((u, v, score))
                touched.add(u); touched.add(v)

        # Build a compact graph H with just touched nodes + incident edges
        H = nx.DiGraph()
        H.add_nodes_from(sorted(touched))
        for u, v, score in incident_edges:
            H.add_edge(u, v, edge_mask=score)

        # -----------------------------
        # 5) Deterministic, readable positions:
        #    node 8 fixed at center, neighbors on a circle
        # -----------------------------
        pos = {}
        pos[node_index] = (0.0, 0.0)

        # stable neighbor order (sorted) and even angular spacing
        neighbors = sorted([n for n in H.nodes if n != node_index])
        angles = np.linspace(0, 2*np.pi, max(len(neighbors), 1), endpoint=False)
        radius = 1.0
        for a, n in zip(angles, neighbors):
            pos[n] = (radius*np.cos(a), radius*np.sin(a))

        # -----------------------------
        # 6) Plot with importance-aware styling
        # -----------------------------
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12
        fig, ax = plt.subplots(figsize=(6, 5))

        # focal node highlighted
        nx.draw_networkx_nodes(
            H, pos, nodelist=[node_index], node_color='#87CEFA', node_size=720, ax=ax,
        )
        nx.draw_networkx_nodes(
            H, pos, nodelist=[n for n in H.nodes if n != node_index],
            node_color='lightblue', node_size=520, ax=ax,
        )
        nx.draw_networkx_labels(H, pos, font_family='Times New Roman', font_size=12, ax=ax)

        # draw edges one-by-one to vary width/alpha by importance
        for (u, v, d) in H.edges(data=True):
            score = float(d.get('edge_mask', 0.0))
            width = 0.5 + 1.0 * score          # 2–8 px
            alpha = 0.45 + 0.55 * score        # 0.45–1.0
            nx.draw_networkx_edges(
                H, pos, edgelist=[(u, v)],
                edge_color='gray', width=width, alpha=alpha,
                arrows=True, arrowstyle='-|>', arrowsize=12, node_size=520,
                connectionstyle='arc3,rad=0.12' if u != v else 'arc3,rad=0.3',
                ax=ax
            )

        # edge labels from attribute; white bbox improves contrast
        edge_labels = {(u, v): f"{float(d.get('edge_mask', 0.0)):.2f}"
                    for u, v, d in H.edges(data=True)}
        nx.draw_networkx_edge_labels(
            H, pos, edge_labels=edge_labels, font_family='Times New Roman', font_size=11, label_pos=0.82,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85),
            rotate=False, ax=ax
        )

        ax.axis('off')
        fig.tight_layout()
        fig.savefig('log/graph/subgraph_node8.pdf', dpi=300)

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import degree
from torch_geometric.data import Data
from typing import Dict, Optional, Tuple


class Graph_Analysis:
    def __init__(self):
        # units: mm, N, MPa, ton, second
        config = 'configs/config.yaml'
        conc_prop, dmg_para, geo_para, mesh_para, contact_para = load_config(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Geometry
        geom = Layered_Contact_Geometry(para=geo_para, device=device)
        # Network
        net = GraphSAGE(input_dim=3, hidden_dim=3, output_dim=9)
        # PDE
        mat = Quadrilinear_Concrete(prop=conc_prop)
        pde = PDE_Graph(net=net, E=mat.prop.E_pinn, nu=mat.prop.nu)
        # PINN
        pinn = PINN_Graph(net=net, pde=pde, geom=geom, num_domain=8, num_bc=16, device=device)
        pinn.generate_points(layer=10, method='lhs', seed=1, k=8)
        self.pinn = pinn
        self.pinn_explanation = PINN_Graph(net=net, pde=pde, geom=geom, num_domain=8, num_bc=3, device=device)

        model = pinn.net
        layer = 10
        weight_path = f"log/graph/3x2/weights/Step0_layer{layer}_weight.pth"
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
    
    # =========================================================================
    # NEW METHOD: Analyze GNN vs MLP Advantage Parameters
    # =========================================================================
    def analyze_gnn_parameters(
        self,
        data: Data,
        q: float = 3.0,
        save_path: str = 'log/graph/gnn_analysis',
        verbose: bool = True
    ) -> Dict:
        """
        Analyze graph parameters to estimate theoretical GNN advantage over MLP.
        
        This implements the analysis from "Quantifying the Optimization and 
        Generalization Advantages of Graph Neural Networks Over MLPs" 
        (Huang et al., AISTATS 2025).
        
        The key formula is:
            SNR_MLP_min / SNR_GNN_min = D^((q-2)/(2q))
        
        where:
            - D: expected node degree (average number of neighbors)
            - q: power of polynomial ReLU activation (heuristic for GELU)
        
        IMPORTANT CAVEATS:
            - Theory assumes classification; your PINN is regression
            - Theory assumes polynomial ReLU; you use GELU (q≈3 is heuristic)
            - Theory assumes signal-noise data model; you have PDE constraints
            - Results provide INTUITION, not guarantees
        
        Args:
            data: PyTorch Geometric Data object containing:
                  - x: node features [n, input_dim]
                  - edge_index: graph connectivity [2, num_edges]
            q: Effective activation power. For GELU, use q≈3 as heuristic.
               Must be > 2 for the theory to apply.
            save_path: Directory to save analysis results and plots.
            verbose: Whether to print detailed analysis to console.
        
        Returns:
            Dictionary containing all computed parameters and metrics.
        
        Example:
            >>> explainer = Graph_Explainer()
            >>> data = explainer.pinn.data  # or however you access your graph data
            >>> results = explainer.analyze_gnn_parameters(data, q=3.0)
            >>> print(f"GNN advantage factor: {results['advantage']:.2f}x")
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # =====================================================================
        # STEP 1: Extract basic graph statistics
        # =====================================================================
        
        # Number of nodes (collocation points in your PINN)
        n = data.x.shape[0]
        
        # Number of features per node (should be 3: x, y, z coordinates)
        input_dim = data.x.shape[1]
        
        # Edge index shape: [2, num_edges]
        # For undirected graphs, edges are stored twice (i->j and j->i)
        num_edges_directed = data.edge_index.shape[1]
        
        # =====================================================================
        # STEP 2: Compute node degree statistics
        # =====================================================================
        
        # degree() counts outgoing edges from each node
        # For undirected graphs stored with both directions, this gives true degree
        deg = degree(
            data.edge_index[0],  # Source nodes
            num_nodes=n,
            dtype=torch.float
        )
        
        # Key parameter D: expected (average) node degree
        D = deg.mean().item()
        
        # Additional degree statistics for understanding graph structure
        D_min = deg.min().item()
        D_max = deg.max().item()
        D_std = deg.std().item()
        D_median = deg.median().item()
        
        # =====================================================================
        # STEP 3: Compute graph density
        # =====================================================================
        
        # For undirected graph: actual unique edges = directed edges / 2
        # (assuming edges stored as both i->j and j->i)
        num_edges_undirected = num_edges_directed / 2
        
        # Maximum possible edges in undirected graph: n*(n-1)/2
        max_possible_edges = n * (n - 1) / 2
        
        # Graph density: fraction of possible edges that exist
        # This is analogous to (p + s) in the paper's notation
        density = num_edges_undirected / max_possible_edges if max_possible_edges > 0 else 0
        
        # Verify consistency: D should approximately equal (n-1) * density
        # Because each node connects to density fraction of other (n-1) nodes
        D_from_density = (n - 1) * density
        
        # =====================================================================
        # STEP 4: Compute theoretical GNN advantage
        # =====================================================================
        
        # The advantage formula from the paper:
        # SNR_MLP_min / SNR_GNN_min = D^((q-2)/(2q))
        #
        # This means GNN can achieve low test error with lower SNR than MLP
        
        if D > 1 and q > 2:
            # Compute the exponent
            exponent = (q - 2) / (2 * q)
            
            # Compute the advantage factor
            advantage = D ** exponent
            
            # Alternative interpretation: noise tolerance ratio
            # GNN can handle advantage^2 times more noise variance
            noise_tolerance_ratio = advantage ** 2
            
            # Sample complexity improvement
            # GNN needs 1/D^((q-2)/2) times fewer samples for same error
            sample_complexity_factor = D ** ((q - 2) / 2)
        else:
            exponent = None
            advantage = None
            noise_tolerance_ratio = None
            sample_complexity_factor = None
        
        # =====================================================================
        # STEP 5: Analyze graph connectivity patterns
        # =====================================================================
        
        # For PINNs, understanding spatial connectivity is important
        # Check if graph is connected and analyze clustering
        
        # Compute degree distribution histogram
        deg_np = deg.cpu().numpy()
        unique_degrees, degree_counts = np.unique(deg_np.astype(int), return_counts=True)
        
        # Check for isolated nodes (degree = 0)
        num_isolated = (deg == 0).sum().item()
        
        # Check for hub nodes (degree > 2 * average)
        num_hubs = (deg > 2 * D).sum().item()
        
        # =====================================================================
        # STEP 6: Estimate effective parameters for different q values
        # =====================================================================
        
        # Since GELU doesn't have exact q, compute advantage for range of q
        q_values = [2.5, 3.0, 3.5, 4.0]
        advantages_by_q = {}
        
        for q_val in q_values:
            if D > 1 and q_val > 2:
                exp_val = (q_val - 2) / (2 * q_val)
                advantages_by_q[q_val] = D ** exp_val
            else:
                advantages_by_q[q_val] = None
        
        # =====================================================================
        # STEP 7: Compile results dictionary
        # =====================================================================
        
        results = {
            # Basic graph statistics
            'n': n,                           # Number of nodes
            'input_dim': input_dim,           # Feature dimension
            'num_edges': int(num_edges_undirected),  # Number of unique edges
            
            # Degree statistics (KEY PARAMETER)
            'D': D,                           # Average degree (main parameter)
            'D_min': D_min,                   # Minimum degree
            'D_max': D_max,                   # Maximum degree  
            'D_std': D_std,                   # Standard deviation of degree
            'D_median': D_median,             # Median degree
            
            # Density
            'density': density,               # Graph density (p+s equivalent)
            'D_from_density': D_from_density, # D computed from density (should match)
            
            # Connectivity analysis
            'num_isolated': int(num_isolated),  # Nodes with no edges
            'num_hubs': int(num_hubs),          # High-degree nodes
            
            # Theoretical advantage (MAIN RESULTS)
            'q': q,                           # Activation power used
            'exponent': exponent,             # (q-2)/(2q)
            'advantage': advantage,           # SNR_MLP / SNR_GNN ratio
            'noise_tolerance_ratio': noise_tolerance_ratio,  # advantage^2
            'sample_complexity_factor': sample_complexity_factor,  # D^((q-2)/2)
            
            # Sensitivity analysis
            'advantages_by_q': advantages_by_q,  # Advantage for different q values
            
            # Degree distribution
            'degree_distribution': {
                'unique_degrees': unique_degrees.tolist(),
                'counts': degree_counts.tolist()
            }
        }
        
        # =====================================================================
        # STEP 8: Print detailed analysis report
        # =====================================================================
        
        if verbose:
            self._print_gnn_analysis_report(results)
        
        # =====================================================================
        # STEP 9: Generate and save visualizations
        # =====================================================================
        
        self._plot_gnn_analysis(results, deg_np, save_path)
        
        # =====================================================================
        # STEP 10: Save results to CSV
        # =====================================================================
        
        self._save_gnn_analysis_results(results, save_path)
        
        return results
    
    def _print_gnn_analysis_report(self, results: Dict) -> None:
        """
        Print a formatted analysis report to console.
        
        Args:
            results: Dictionary from analyze_gnn_parameters()
        """
        print("\n" + "=" * 70)
        print("GNN vs MLP ADVANTAGE ANALYSIS REPORT")
        print("Based on Huang et al. (AISTATS 2025)")
        print("=" * 70)
        
        # Section 1: Graph Statistics
        print("\n" + "-" * 70)
        print("1. GRAPH STATISTICS")
        print("-" * 70)
        print(f"   Number of nodes (n):              {results['n']:,}")
        print(f"   Number of edges:                  {results['num_edges']:,}")
        print(f"   Input feature dimension:          {results['input_dim']}")
        print(f"   Graph density:                    {results['density']:.6f}")
        
        # Section 2: Degree Statistics
        print("\n" + "-" * 70)
        print("2. NODE DEGREE STATISTICS (Key Parameter: D)")
        print("-" * 70)
        print(f"   Average degree (D):               {results['D']:.2f}  ← Main parameter")
        print(f"   Minimum degree:                   {results['D_min']:.0f}")
        print(f"   Maximum degree:                   {results['D_max']:.0f}")
        print(f"   Median degree:                    {results['D_median']:.0f}")
        print(f"   Std deviation:                    {results['D_std']:.2f}")
        print(f"   Isolated nodes (degree=0):        {results['num_isolated']}")
        print(f"   Hub nodes (degree>2D):            {results['num_hubs']}")
        
        # Section 3: Theoretical Advantage
        print("\n" + "-" * 70)
        print("3. THEORETICAL GNN ADVANTAGE")
        print("-" * 70)
        print(f"   Activation power (q):             {results['q']} (heuristic for GELU)")
        
        if results['advantage'] is not None:
            print(f"   Exponent (q-2)/(2q):              {results['exponent']:.4f}")
            print(f"\n   ┌─────────────────────────────────────────────────────────┐")
            print(f"   │  SNR_MLP_min / SNR_GNN_min = D^((q-2)/(2q))            │")
            print(f"   │                            = {results['D']:.2f}^{results['exponent']:.4f}              │")
            print(f"   │                            = {results['advantage']:.4f}                  │")
            print(f"   └─────────────────────────────────────────────────────────┘")
            print(f"\n   Interpretation:")
            print(f"   • GNN can work with {results['advantage']:.2f}× lower SNR than MLP")
            print(f"   • GNN can tolerate {results['noise_tolerance_ratio']:.2f}× more noise variance")
            print(f"   • Sample complexity improved by factor of {results['sample_complexity_factor']:.2f}×")
        else:
            print("   Cannot compute advantage (need D > 1 and q > 2)")
        
        # Section 4: Sensitivity to q
        print("\n" + "-" * 70)
        print("4. SENSITIVITY TO ACTIVATION POWER (q)")
        print("-" * 70)
        print("   Since GELU has no exact q, here's the advantage for different q:")
        print(f"\n   {'q':<6} {'Exponent':<12} {'Advantage':<12} {'Noise Tolerance':<15}")
        print(f"   {'-'*6} {'-'*12} {'-'*12} {'-'*15}")
        
        for q_val, adv in results['advantages_by_q'].items():
            if adv is not None:
                exp = (q_val - 2) / (2 * q_val)
                noise_tol = adv ** 2
                print(f"   {q_val:<6.1f} {exp:<12.4f} {adv:<12.4f} {noise_tol:<15.4f}")
        
        # Section 5: Caveats
        print("\n" + "-" * 70)
        print("5. IMPORTANT CAVEATS")
        print("-" * 70)
        print("   ⚠ Theory assumes binary classification; you have regression")
        print("   ⚠ Theory assumes polynomial ReLU; you use GELU (q≈3 is heuristic)")
        print("   ⚠ Theory assumes signal-noise model; you have PDE constraints")
        print("   ⚠ These numbers provide INTUITION, not guarantees")
        print("   ✓ Higher D generally means better noise suppression in GNN")
        print("   ✓ Graph convolution acts as spatial smoothing for your PINN")
        
        print("\n" + "=" * 70)
        print("END OF REPORT")
        print("=" * 70 + "\n")
    
    def _plot_gnn_analysis(
        self, 
        results: Dict, 
        deg_np: np.ndarray, 
        save_path: str
    ) -> None:
        """
        Generate and save visualization plots for the analysis.
        
        Args:
            results: Dictionary from analyze_gnn_parameters()
            deg_np: Numpy array of node degrees
            save_path: Directory to save plots
        """
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # ----- Plot 1: Degree Distribution Histogram -----
        ax1 = axes[0, 0]
        ax1.hist(deg_np, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(results['D'], color='red', linestyle='--', linewidth=2, 
                    label=f'Mean D = {results["D"]:.2f}')
        ax1.axvline(results['D_median'], color='orange', linestyle=':', linewidth=2,
                    label=f'Median = {results["D_median"]:.0f}')
        ax1.set_xlabel('Node Degree', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Node Degree Distribution', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ----- Plot 2: Advantage vs q -----
        ax2 = axes[0, 1]
        q_range = np.linspace(2.1, 5, 100)
        D = results['D']
        
        if D > 1:
            advantages = D ** ((q_range - 2) / (2 * q_range))
            ax2.plot(q_range, advantages, 'b-', linewidth=2)
            ax2.axvline(results['q'], color='red', linestyle='--', linewidth=2,
                       label=f'Your q = {results["q"]}')
            if results['advantage']:
                ax2.axhline(results['advantage'], color='green', linestyle=':', linewidth=1,
                           label=f'Advantage = {results["advantage"]:.2f}')
                ax2.scatter([results['q']], [results['advantage']], color='red', s=100, zorder=5)
        
        ax2.set_xlabel('Activation Power (q)', fontsize=12)
        ax2.set_ylabel('SNR Advantage Factor', fontsize=12)
        ax2.set_title(f'GNN Advantage vs Activation Power (D = {D:.2f})', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(2, 5)
        
        # ----- Plot 3: Advantage vs D -----
        ax3 = axes[1, 0]
        D_range = np.linspace(2, 100, 100)
        q = results['q']
        
        if q > 2:
            advantages_D = D_range ** ((q - 2) / (2 * q))
            ax3.plot(D_range, advantages_D, 'b-', linewidth=2)
            ax3.axvline(D, color='red', linestyle='--', linewidth=2,
                       label=f'Your D = {D:.2f}')
            if results['advantage']:
                ax3.axhline(results['advantage'], color='green', linestyle=':', linewidth=1,
                           label=f'Advantage = {results["advantage"]:.2f}')
                ax3.scatter([D], [results['advantage']], color='red', s=100, zorder=5)
        
        ax3.set_xlabel('Average Degree (D)', fontsize=12)
        ax3.set_ylabel('SNR Advantage Factor', fontsize=12)
        ax3.set_title(f'GNN Advantage vs Graph Density (q = {q})', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # ----- Plot 4: Summary Box -----
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary text box
        summary_text = f"""
        SUMMARY OF GNN vs MLP ANALYSIS
        ══════════════════════════════════════
        
        Key Parameters:
        ─────────────────────────────────────
        • Nodes (n):           {results['n']:,}
        • Edges:               {results['num_edges']:,}
        • Average Degree (D):  {results['D']:.2f}
        • Activation (q):      {results['q']} (GELU heuristic)
        
        Theoretical Advantage:
        ─────────────────────────────────────
        • SNR Advantage:       {results['advantage']:.2f}× 
          (GNN needs {results['advantage']:.2f}× lower SNR)
        
        • Noise Tolerance:     {results['noise_tolerance_ratio']:.2f}×
          (GNN handles {results['noise_tolerance_ratio']:.2f}× more noise)
        
        Formula:
        ─────────────────────────────────────
        SNR_MLP / SNR_GNN = D^((q-2)/(2q))
                         = {results['D']:.2f}^{results['exponent']:.4f}
                         = {results['advantage']:.4f}
        
        ⚠️ Results are intuitive estimates for regression PINN
        """
        
        ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/gnn_advantage_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}/gnn_advantage_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {save_path}/gnn_advantage_analysis.pdf")
    
    def _save_gnn_analysis_results(self, results: Dict, save_path: str) -> None:
        """
        Save analysis results to CSV files.
        
        Args:
            results: Dictionary from analyze_gnn_parameters()
            save_path: Directory to save CSV files
        """
        # Save main parameters
        main_params = [
            ['Parameter', 'Value', 'Description'],
            ['n', results['n'], 'Number of nodes'],
            ['num_edges', results['num_edges'], 'Number of edges'],
            ['input_dim', results['input_dim'], 'Feature dimension'],
            ['D', results['D'], 'Average degree (main parameter)'],
            ['D_min', results['D_min'], 'Minimum degree'],
            ['D_max', results['D_max'], 'Maximum degree'],
            ['D_median', results['D_median'], 'Median degree'],
            ['D_std', results['D_std'], 'Degree standard deviation'],
            ['density', results['density'], 'Graph density'],
            ['q', results['q'], 'Activation power (heuristic)'],
            ['exponent', results['exponent'], '(q-2)/(2q)'],
            ['advantage', results['advantage'], 'SNR advantage factor'],
            ['noise_tolerance', results['noise_tolerance_ratio'], 'Noise tolerance ratio'],
            ['sample_complexity', results['sample_complexity_factor'], 'Sample complexity factor'],
        ]
        
        np.savetxt(
            f'{save_path}/gnn_analysis_parameters.csv',
            main_params,
            delimiter=',',
            fmt='%s',
            comments=''
        )
        
        # Save degree distribution
        degree_dist = results['degree_distribution']
        dist_data = np.column_stack([
            degree_dist['unique_degrees'],
            degree_dist['counts']
        ])
        np.savetxt(
            f'{save_path}/degree_distribution.csv',
            dist_data,
            delimiter=',',
            header='degree,count',
            comments='',
            fmt=['%d', '%d']
        )
        
        # Save sensitivity analysis
        sensitivity_data = [['q', 'exponent', 'advantage', 'noise_tolerance']]
        for q_val, adv in results['advantages_by_q'].items():
            if adv is not None:
                exp = (q_val - 2) / (2 * q_val)
                noise_tol = adv ** 2
                sensitivity_data.append([q_val, exp, adv, noise_tol])
        
        np.savetxt(
            f'{save_path}/sensitivity_analysis.csv',
            sensitivity_data,
            delimiter=',',
            fmt='%s',
            comments=''
        )
        
        print(f"Results saved to {save_path}/")
    

if __name__ == '__main__':
    ge = Graph_Explainer()
    top = ge.pinn.top_points
    bottom = ge.pinn.bottom_points
    left = ge.pinn.left_points
    ge.analyze_nodes_feature_importance(data=top, pos='top')