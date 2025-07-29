# test/pc_fem/test_materials.py

from pc_fem.fem.materials import Quadrilinear_Concrete, Five_Line_Concrete
from pc_fem.config_loader import load_config

conc_prop, *_ = load_config()

def test_quadrilinear() -> None:
    # Material instantiation
    mat_ql = Quadrilinear_Concrete(prop=conc_prop)
    mat_ql.print_properties()
    
    eps_c_max = -0.015
    eps_t_max = 1e-2
    d_eps = 1e-6
    mat_ql.plot_stress_strain_compression(eps_c_max=eps_c_max, d_eps=d_eps)
    mat_ql.plot_stress_strain_tension(eps_t_max=eps_t_max, d_eps=d_eps)
    mat_ql.plot_stress_strain(eps_c_max=eps_c_max , eps_t_max=eps_t_max, d_eps=d_eps)

def test_five_line() -> None:
    # Material instantiation
    mat_fl = Five_Line_Concrete(prop=conc_prop)
    mat_fl.print_properties()
    
    eps_c_max = -0.015
    eps_t_max = 1e-2
    d_eps = 1e-6
    # mat_fl.plot_stress_strain_compression(eps_c_max=eps_c_max, d_eps=d_eps)
    mat_fl.plot_stress_strain_tension(eps_t_max=eps_t_max, d_eps=d_eps)
    # mat_fl.plot_stress_strain(eps_c_max=eps_c_max , eps_t_max=eps_t_max, d_eps=d_eps)


if __name__ == '__main__':
    test_quadrilinear()
    # test_five_line()
    pass