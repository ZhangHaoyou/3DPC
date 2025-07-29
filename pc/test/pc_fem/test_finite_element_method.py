# test/pc_fem/test_finite_element_method.py

from pc_fem.fem.materials import Quadrilinear_Concrete, Five_Line_Concrete
from pc_fem.fem.damage import Mazars_Original_Damage, Mu_Damage, Modified_Mazars_Damage
from pc_fem.fem.geo_mesh import Geometry, Mesh
from pc_fem.fem.finite_element_method import Finite_Element_Method

from pc_fem.config_loader import load_config

# units: mm, N, MPa, ton, second
conc_prop, dmg_para, geo_para, mesh_para, contact_para = load_config()

def test_a_brick() -> None:
    # Material instantiation
    mat = Quadrilinear_Concrete(prop=conc_prop)
    
    # Mesh
    geo = Geometry(para=geo_para)
    mesh = Mesh(geo=geo, para=mesh_para)

    dmg = Mazars_Original_Damage(mat=mat, para=dmg_para)
    
    # FEM solver initialization
    fem = Finite_Element_Method(mat=mat, mesh=mesh, dmg=dmg)
    fem.configure_material()
    nodes, elements, contacts = fem.configure_geometry(gap_between_layers=0.0, plot_geometry=False)
    fem.analyze_bricks()

def test_contact() -> None:
    # Material instantiation
    mat = Quadrilinear_Concrete(prop=conc_prop)
    
    # Mesh
    geo = Geometry(para=geo_para)
    mesh = Mesh(geo=geo, para=mesh_para)

    dmg = Mazars_Original_Damage(mat=mat, para=dmg_para)
    
    # FEM solver initialization
    fem = Finite_Element_Method(mat=mat, mesh=mesh, dmg=dmg)
    fem.configure_material()
    nodes, elements, contacts = fem.configure_geometry(gap_between_layers=0.0, plot_geometry=False)
    fem.analyze_contacts(contact_para=contact_para)


if __name__ == '__main__':
    test_a_brick()
    test_contact()