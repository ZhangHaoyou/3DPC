# main.py


import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt


from fem.config_loader import load_config
from fem.fem.materials import Quadrilinear_Concrete, Five_Line_Concrete
from fem.fem.damage import Mazars_Original_Damage, Mazars_Original_Damage_Torch, Mu_Damage, Modified_Mazars_Damage




def main_fem():
    from fem.fem.geo_mesh import Geometry, Mesh
    from fem.fem.finite_element_method import Finite_Element_Method
    
    start_time = time.time()
    
    # units: mm, N, MPa, ton, second
    config = 'configs/config.yaml'
    conc_prop, dmg_para, geo_para, mesh_para, contact_para = load_config(config)
    
    # Material instantiation
    mat = Quadrilinear_Concrete(prop=conc_prop)
    # mat = Five_Line_Concrete(prop=conc_prop)
    
    # Mesh
    geo = Geometry(para=geo_para)
    mesh = Mesh(geo=geo, para=mesh_para)
    
    dmg = Mazars_Original_Damage(mat=mat, para=dmg_para)
    # dmg = Mu_Damage(mat=mat, para=dmg_para)
    # dmg = Modified_Mazars_Damage(mat=mat, mesh=mesh, para=dmg_para)
    
    # FEM solver initialization
    fem = Finite_Element_Method(mat=mat, mesh=mesh, dmg=dmg)
    fem.configure_material()
    fem.configure_geometry(gap_between_layers=0.0, plot_geometry=False)
    fem.analyze_contacts(contact_para=contact_para,
            pressure_per_step=-1.0,
            n_steps=40,
            move_vertically=False, B_bar=True)
    fem.post_process(show=False)
    
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"\nAnalysis done. Elapsed time: {elapsed_minutes:.2f} minutes.")

def main_pinn() -> None:
    from dl.pinn.pde import PDE
    from dl.pinn.pinn import PINN
    from dl.pinn.networks import MLP
    from dl.pinn.geometry import Layered_Contact_Geometry
    from dl.pinn.incremental_pressure_applier import Incremental_Pressure_Applier
    
    start_time = time.time()
    
    SEED = 2025
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    
    # units: mm, N, MPa, ton, second
    config = 'configs/config.yaml'
    conc_prop, dmg_para, geo_para, mesh_para, contact_para = load_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Geometry
    geom = Layered_Contact_Geometry(para=geo_para, device=device)
    
    # Network
    # layer_size = [3] + [50] * 15 + [9]
    layer_size = [3] + [4] * 5 + [9]
    net = MLP(layer_size=layer_size)
    
    # PDE
    mat = Quadrilinear_Concrete(prop=conc_prop)
    dmg = Mazars_Original_Damage_Torch(mat=mat, para=dmg_para)
    pde = PDE(net=net, E=mat.prop.E_pinn, nu=mat.prop.nu)
    
    # PINN
    pinn = PINN(net=net, pde=pde, geom=geom, num_domain=8, num_bc=16, device=device)
    
    n_step = 40
    ipa = Incremental_Pressure_Applier(pinn=pinn, dmg=dmg)
    ipa.apply_load(pressure=-1.0, n_step=n_step, num_epochs=30000)
    ipa.save_to_vtu(n_step=n_step)
    ipa.plot_load_disp()
    
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    pinn.logger.debug(f"Done. Elapsed time: {elapsed_minutes:.2f} minutes.")

def main_graph():
    from dl.graph.pde import PDE_Graph
    from dl.graph.pinn import PINN_Graph
    from dl.graph.incremental_pressure_applier import Incremental_Pressure_Applier_Graph
    from dl.pinn.networks import GraphSAGE
    from dl.pinn.geometry import Layered_Contact_Geometry
    
    start_time = time.time()
    
    SEED = 2025
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    
    # units: mm, N, MPa, ton, second
    config = 'configs/config.yaml'
    conc_prop, dmg_para, geo_para, mesh_para, contact_para = load_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Geometry
    geom = Layered_Contact_Geometry(para=geo_para, device=device)
    # Network
    net = GraphSAGE() # ChebNetLite()
    # PDE
    mat = Quadrilinear_Concrete(prop=conc_prop)
    dmg = Mazars_Original_Damage_Torch(mat=mat, para=dmg_para)
    pde = PDE_Graph(net=net, E=mat.prop.E_pinn, nu=mat.prop.nu)
    # PINN
    pinn = PINN_Graph(net=net, pde=pde, geom=geom, num_domain=8, num_bc=16, device=device)
    n_step = 40
    ipa = Incremental_Pressure_Applier_Graph(pinn=pinn, dmg=dmg)
    ipa.apply_load(pressure=-1.0, n_step=n_step, num_epochs=10000)
    ipa.save_to_vtu(n_step=n_step)
    fig, ax = ipa.plot_load_disp()
    plt.close(fig)
    
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    pinn.logger.debug(f"Done. Elapsed time: {elapsed_minutes:.2f} minutes.")

def main_graph_interpretation() -> None:
    from dl.graph.explainer import Graph_Explainer
    
    SEED = 2025
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    
    ge = Graph_Explainer()
    top = ge.pinn.top_points
    bottom = ge.pinn.bottom_points
    left = ge.pinn.left_points
    inside = ge.pinn.inside_points
    # ge.analyze_nodes_feature_importance(data=top, pos='top')
    # ge.analyze_nodes_feature_importance(data=bottom, pos='bottom')
    # ge.analyze_nodes_feature_importance(data=left, pos='left')
    # ge.analyze_nodes_feature_importance(data=inside, pos='inside')
    # ge.plot_subgraph(node_index=0, data=top, path='log/graph/subgraph_top_0.pdf')
    # ge.plot_pruned_subgraph(node_index=2, data=top, path='log/graph/pruned_subgraph_top_0.pdf')
    ge.plot_subgraph_test()

def main_graph_analysis() -> None:
    from dl.graph.explainer import Graph_Analysis
    
    ga = Graph_Analysis()
    ga.pinn.generate_points(layer=0, method='lhs', seed=1, k=8)
    data = ga.pinn.domain_points
    print('Data shape:', data.x.shape)
    results = ga.analyze_gnn_parameters(
        data=data,
        q=3.0,                              # Heuristic for GELU
        save_path='log/graph/gnn_analysis', # Where to save outputs
        verbose=True                         # Print detailed report
    )
    
    # Access specific results
    print(f"\nQuick Summary:")
    print(f"  Average degree D = {results['D']:.2f}")
    print(f"  GNN advantage = {results['advantage']:.2f}x")
    print(f"  GNN can tolerate {results['noise_tolerance_ratio']:.2f}x more noise")


if __name__ == '__main__':
    # main_fem()
    # main_pinn()
    # main_graph()
    main_graph_interpretation()
    # main_graph_analysis()
    pass
