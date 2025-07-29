# test/pc_fem/test_boundary_conditions.py

from pc_fem.fem.geo_mesh import Geometry, Mesh
from pc_fem.fem.boundary_conditions import Boundary_Conditions

from pc_fem.config_loader import load_config

_, _, geo_para, mesh_para, _ = load_config()

def test_bc() -> None:
    geo = Geometry(para=geo_para)
    geo.print_parameters()
    
    mesh = Mesh(geo=geo, para=mesh_para)
    mesh.print_parameters()
    mesh.mesh_layers(gap_between_layers=0.0, plot_geometry=False)
    
    bc = Boundary_Conditions(mesh=mesh, move_vertically=False)
    nodal_forces = bc.pressure_to_nodal_forces(pressure=-1.0)
    print('Nodal forces:\n', nodal_forces)
    print('Nodal coords:\n', mesh.nodes[mesh.loading_node_tags])

if __name__ == '__main__':
    test_bc()