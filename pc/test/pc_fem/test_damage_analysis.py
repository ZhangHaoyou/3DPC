# test/pc_fem/test_damage_analysis.py

import numpy as np


from pc_fem.fem.materials import Quadrilinear_Concrete, Five_Line_Concrete
from pc_fem.fem.damage import Mazars_Original_Damage, Mu_Damage, Modified_Mazars_Damage
from pc_fem.fem.damage_analysis import MO_Evolution_Analysis
from pc_fem.config_loader import load_config


conc_prop, dmg_para, *_ = load_config()

Ac_cal, Bc_cal, At_cal, Bt_cal = 2.45, 1286, 1.0, 41086 # ib = 1.0

def analyze_mo_damage() -> None:
    # Material instantiation
    mat = Quadrilinear_Concrete(prop=conc_prop)

    # Damage instantiation
    mo = Mazars_Original_Damage(mat=mat, para=dmg_para)
    Ac_cal, Bc_cal, At_cal, Bt_cal = mo.calculate_Ac_Bc_At_Bt()
    
    dea = MO_Evolution_Analysis(dmg=mo)
    
    eps_c_max = 6e-3
    """Smaller Ac -> greater d; d < 0 when Ac > 1; d > 1 when kappa is further greater"""
    Bc = 2000
    Acs = np.array([0.5, 1.0, 1.5, 2.0])
    dea.analyze_Ac(eps_c_max=eps_c_max, Acs=Acs, Bc=Bc)
    
    """Larger Bc -> greater d; no outliers"""
    Ac = 1.0
    Bcs = np.array([500, 1000, 2000, 3000])
    dea.analyze_Bc(eps_c_max=eps_c_max, Bcs=Bcs, Ac=Ac)
    
    pass

if __name__ == '__main__':
    analyze_mo_damage()