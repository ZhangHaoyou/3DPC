# test/pc_fem/geo_mesh.py

from pc_fem.fem.geo_mesh import Geometry, Mesh
from pc_fem.config_loader import load_config

_, _, geo_para, mesh_para, _ = load_config()

def test_geometry_mesh() -> None:
    geo = Geometry(para=geo_para)
    geo.print_parameters()
    
    mesh = Mesh(geo=geo, para=mesh_para)
    mesh.print_parameters()
    mesh.mesh_layers(gap_between_layers=0.0, plot_geometry=True)


if __name__ == '__main__':
    test_geometry_mesh()