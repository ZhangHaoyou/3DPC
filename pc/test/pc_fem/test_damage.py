# test/pc_fem/test_damage.py


import matplotlib
import platform

if platform.system() != 'Windows':
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from typing import Tuple


from pc_fem.fem.geo_mesh import Geometry, Mesh
from pc_fem.fem.materials import Quadrilinear_Concrete, Five_Line_Concrete
from pc_fem.fem.damage import Mazars_Original_Damage, Mu_Damage, Modified_Mazars_Damage
from pc_fem.config_loader import load_config


def test_mazars_original_damage() -> None:
    conc_prop, dmg_para, *_ = load_config(path='configs/config_mo.yaml')
    
    # Material instantiation
    mat_ql = Quadrilinear_Concrete(prop=conc_prop)

    # Damage instantiation
    mo = Mazars_Original_Damage(mat=mat_ql, para=dmg_para)
    mo.print_parameters()
    
    mo.plot_damage_evolution_from_elastic_eps()

    Ac_cal, Bc_cal, At_cal, Bt_cal = mo.calculate_Ac_Bc_At_Bt()

    # Identify At Bt
    initial_At_Bt = (mo.para.At, mo.para.Bt)
    At_fit, Bt_fit = mo.identify_A_B_from_stress(mode='tension', initial_AB=initial_At_Bt, AB_bounds=((0.5, 1e3), (2.0, 5e4)), eps_max=0.004)

    # # Identify Ac Bc
    initial_Ac_Bc = (mo.para.Ac, mo.para.Bc)
    Ac_fit, Bc_fit = mo.identify_A_B_from_stress(mode='compression', initial_AB=initial_Ac_Bc, AB_bounds=((0.5, 1e3), (3.0, 5e4)), eps_max=0.01)

    eps_c_max = -0.015
    eps_t_max = 1e-2
    mo.plot_uniaxial_compression_damage_evolution(eps_c_max=eps_c_max, d_eps=1e-4, Ac=Ac_cal, Bc=Bc_cal)
    mo.plot_uniaxial_compression_damage_evolution(eps_c_max=eps_c_max, d_eps=1e-4, Ac=1.0, Bc=Bc_cal)
    mo.plot_uniaxial_tension_damage_evolution(eps_t_max=eps_t_max, d_eps=1e-6, At=At_cal, Bt=Bt_cal)
    
    mo.plot_damaged_uniaxial_tensile_stress(eps_t_max=eps_t_max, d_eps=1e-6, At=mo.para.At, Bt=mo.para.Bt)
    eps_d0 = mat_ql.prop.eps_tp
    mo.plot_damaged_uniaxial_compressive_stress(eps_c_max=eps_c_max, eps_d0=eps_d0, d_eps=1e-5, Ac=mo.para.Ac, Bc=mo.para.Bc)

def test_mo_parameters_from_papers(config_path, At, Bt, Ac, Bc) -> Tuple[plt.Figure, plt.Axes, plt.Figure, plt.Axes]:
    
    conc_prop, dmg_para, *_ = load_config(path=config_path)
    
    # Material instantiation
    mat_ql = Quadrilinear_Concrete(prop=conc_prop)

    # Damage instantiation
    mo = Mazars_Original_Damage(mat=mat_ql, para=dmg_para)
    mo.print_parameters()
    
    eps_c_max = -0.015
    eps_t_max = 1e-2
    fig_t, ax_t = mo.plot_damaged_uniaxial_tensile_stress(eps_t_max=eps_t_max, d_eps=1e-6, At=At, Bt=Bt, show=False, close=False)
    param_text = (f"$A_{{\\mathrm{{t}}}}$ = {At:.2f}, $B_{{\\mathrm{{t}}}}$ = {Bt:.0f}\n")
    ax_t.text(0.95, 0.7, param_text, transform=ax_t.transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    
    eps_d0 = mat_ql.prop.eps_tp
    fig_c, ax_c = mo.plot_damaged_uniaxial_compressive_stress(eps_c_max=eps_c_max, eps_d0=eps_d0, d_eps=1e-5, Ac=Ac, Bc=Bc, show=False, close=False)
    param_text = (f"$A_{{\\mathrm{{c}}}}$ = {Ac:.2f}, $B_{{\\mathrm{{c}}}}$ = {Bc:.0f}\n")
    ax_c.text(0.95, 0.7, param_text, transform=ax_c.transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    
    return fig_t, ax_t, fig_c, ax_c

def test_mo_parameters_from_pijaudier() -> None:
    config_path = 'configs/config_mo_pijaudier_cabot.yaml'
    At = 1.0
    Bt = 15000
    Ac = 1.2
    Bc = 1500
    fig_t, ax_t, fig_c, ax_c = test_mo_parameters_from_papers(config_path=config_path, At=At, Bt=Bt, Ac=Ac, Bc=Bc)
    ax_t.set_ylim(0, 4)
    ax_c.set_ylim(0, 40)
    plt.show()

def test_mo_parameters_from_mazars() -> None:
    config_path = 'configs/config_mazars_2015_tab1.yaml'
    At = 1.0
    Bt = 10000
    Ac = 1.25
    Bc = 517
    fig_t, ax_t, fig_c, ax_c = test_mo_parameters_from_papers(config_path=config_path, At=At, Bt=Bt, Ac=Ac, Bc=Bc)
    ax_t.set_ylim(0, 4)
    ax_c.set_ylim(0, 100)
    plt.show()
    
    config_path = 'configs/config_mazars_2015_tab2.yaml'
    At = 0.8
    Bt = 7000
    Ac = 1.25
    Bc = 395
    fig_t, ax_t, fig_c, ax_c = test_mo_parameters_from_papers(config_path=config_path, At=At, Bt=Bt, Ac=Ac, Bc=Bc)
    ax_t.set_ylim(0, 4)
    ax_c.set_ylim(0, 120)
    plt.show()
    
    config_path = 'configs/config_mazars_2017_tab1.yaml'
    At = 0.9
    Bt = 7000
    Ac = 1.25
    Bc = 400
    fig_t, ax_t, fig_c, ax_c = test_mo_parameters_from_papers(config_path=config_path, At=At, Bt=Bt, Ac=Ac, Bc=Bc)
    ax_t.set_ylim(0, 6)
    ax_c.set_ylim(0, 200)
    plt.show()

def test_mu_damage() -> None:
    conc_prop, dmg_para, *_ = load_config(path='configs/config_mu.yaml')
    
    # Material instantiation
    mat_ql = Quadrilinear_Concrete(prop=conc_prop)
    
    # Damage instantiation
    mu = Mu_Damage(mat=mat_ql, para=dmg_para)
    mu.print_parameters()
    
    # mu.plot_damage_evolution_from_elastic_eps()
    
    initial_AB = (mu.para.Ac, mu.para.Bc, mu.para.At, mu.para.Bt)
    AB_bounds = ((0.2, 1e2, 0.2, 1e2), (3.0, 5e4, 2.0, 5e4))
    Ac_fit_t, Bc_fit_t, At_fit_t, Bt_fit_t = mu.identify_A_B_from_stress(
            mode='tension', initial_AB=initial_AB, AB_bounds=AB_bounds, eps_max=0.004)
    Ac_fit_c, Bc_fit_c, At_fit_c, Bt_fit_c = mu.identify_A_B_from_stress(
            mode='compression', initial_AB=initial_AB, AB_bounds=AB_bounds, eps_max=0.01)
    
    eps_c_max = -0.015
    eps_t_max = 1e-2
    mu.plot_uniaxial_compression_damage_evolution(eps_c_max=eps_c_max, d_eps=1e-4,
            Ac=Ac_fit_c, Bc=Bc_fit_c, At=At_fit_c, Bt=Bt_fit_c)
    mu.plot_uniaxial_tension_damage_evolution(eps_t_max=eps_t_max, d_eps=1e-6,
            Ac=Ac_fit_t, Bc=Bc_fit_t, At=At_fit_t, Bt=Bt_fit_t)
    
    fig_t, ax_t = mu.plot_damaged_uniaxial_tensile_stress(eps_t_max=eps_t_max, d_eps=1e-6,
            Ac=Ac_fit_c, Bc=Bc_fit_c, At=At_fit_c, Bt=Bt_fit_c, show=False, close=False, clip=False)
    param_text = (f"$A_{{\\mathrm{{t}}}}$ = {At_fit_t:.2f}, $B_{{\\mathrm{{t}}}}$ = {Bt_fit_t:.0f}\n")
    ax_t.text(0.95, 0.7, param_text, transform=ax_t.transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    plt.show()
    
    fig_c, ax_c = mu.plot_damaged_uniaxial_compressive_stress(eps_c_max=eps_c_max, d_eps=1e-5,
            Ac=Ac_fit_c, Bc=Bc_fit_c, At=At_fit_c, Bt=Bt_fit_c, show=False, close=False, clip=False)
    param_text = (f"$A_{{\\mathrm{{c}}}}$ = {Ac_fit_c:.2f}, $B_{{\\mathrm{{c}}}}$ = {Bc_fit_c:.0f}\n")
    ax_c.text(0.95, 0.7, param_text, transform=ax_c.transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    plt.show()

def test_mu_parameters_from_papers(config_path, At, Bt, Ac, Bc) -> Tuple[plt.Figure, plt.Axes, plt.Figure, plt.Axes]:
    
    conc_prop, dmg_para, *_ = load_config(path=config_path)
    
    # Material instantiation
    mat_ql = Quadrilinear_Concrete(prop=conc_prop)

    # Damage instantiation
    mu = Mu_Damage(mat=mat_ql, para=dmg_para)
    mu.print_parameters()
    
    eps_c_max = -0.015
    eps_t_max = 1e-2
    fig_t, ax_t = mu.plot_damaged_uniaxial_tensile_stress(eps_t_max=eps_t_max, d_eps=1e-6,
            Ac=Ac, Bc=Bc, At=At, Bt=Bt, show=False, close=False, clip=False)
    param_text = (f"$A_{{\\mathrm{{t}}}}$ = {At:.2f}, $B_{{\\mathrm{{t}}}}$ = {Bt:.0f}\n"
                f"$A_{{\\mathrm{{c}}}}$ = {Ac:.2f}, $B_{{\\mathrm{{c}}}}$ = {Bc:.0f}\n")
    ax_t.text(0.97, 0.7, param_text, transform=ax_t.transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    
    fig_c, ax_c = mu.plot_damaged_uniaxial_compressive_stress(eps_c_max=eps_c_max, d_eps=1e-5,
            Ac=Ac, Bc=Bc, At=At, Bt=Bt, show=False, close=False, clip=False)
    param_text = (f"$A_{{\\mathrm{{t}}}}$ = {At:.2f}, $B_{{\\mathrm{{t}}}}$ = {Bt:.0f}\n"
                f"$A_{{\\mathrm{{c}}}}$ = {Ac:.2f}, $B_{{\\mathrm{{c}}}}$ = {Bc:.0f}\n")
    ax_c.text(0.97, 0.7, param_text, transform=ax_c.transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    
    return fig_t, ax_t, fig_c, ax_c

def test_mu_parameters_from_mazars():
    config_path = 'configs/config_mu.yaml'
    At = 1.0
    Bt = 41085
    Ac = 1.0
    Bc = 3500
    fig_t, ax_t, fig_c, ax_c = test_mu_parameters_from_papers(config_path=config_path, At=At, Bt=Bt, Ac=Ac, Bc=Bc)
    ax_t.set_ylim(0, 5)
    ax_c.set_ylim(0, 60)
    plt.show()
    
    config_path = 'configs/config_mazars_2015_tab1.yaml'
    At = 1.0
    Bt = 10000
    Ac = 1.25
    Bc = 517
    fig_t, ax_t, fig_c, ax_c = test_mu_parameters_from_papers(config_path=config_path, At=At, Bt=Bt, Ac=Ac, Bc=Bc)
    ax_t.set_ylim(0, 4)
    ax_c.set_ylim(0, 100)
    plt.show()

def test_modified_mazars_damage() -> None:
    config_path = 'configs/config_Debuisne.yaml'
    conc_prop, dmg_para, geo_para, mesh_para, _ = load_config()
    
    # Material instantiation
    mat_fl = Five_Line_Concrete(prop=conc_prop)
    
    # Mesh
    geo = Geometry(para=geo_para)
    mesh = Mesh(geo=geo, para=mesh_para)
    
    # Damage instantiation
    mm = Modified_Mazars_Damage(mat=mat_fl, mesh=mesh, para=dmg_para)
    mm.print_parameters()
    
    # mm.plot_damage_evolution_from_elastic_eps()
    
    eps_c_max = -0.015
    eps_t_max = 1e-2
    mm.plot_uniaxial_compression_damage_evolution(eps_c_max=eps_c_max, d_eps=1e-5, clip=True)
    # mm.plot_uniaxial_tension_damage_evolution(eps_t_max=eps_t_max, d_eps=1e-6)

if __name__ == '__main__':
    # test_mazars_original_damage()
    # test_mo_parameters_from_papers()
    # test_mo_parameters_from_pijaudier()
    # test_mo_parameters_from_mazars()
    # test_mu_damage()
    test_mu_parameters_from_mazars()
    # test_modified_mazars_damage()
    pass

