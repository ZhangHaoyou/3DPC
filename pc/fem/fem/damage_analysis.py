# fem/fem/damage_analysis.py
# Need to update if used


import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple


from fem.fem.materials import Quadrilinear_Concrete, Five_Line_Concrete
from fem.fem.damage import Damage, Mazars_Original_Damage, Mu_Damage, Modified_Mazars_Damage
from fem.utils.utils import plot_template


class MO_Evolution_Analysis:
    """Class for visualizing the influence of damage parameters on the damage evolution curve."""

    def __init__(self, dmg: Mazars_Original_Damage) -> None:
        """Initializes the Damage Evolution Analyzer.

        Args:
            dmg (Damage): An instance of a Mazars_Original_Damage model.
        """
        self.dmg = dmg
    
    def analyze_Ac(self, eps_c_max: float, Acs: np.ndarray, Bc: float) -> None:
        """Analyzes and visualizes the influence of different A_c values on compressive damage evolution.

        Args:
            eps_c_max (float): Maximum equivalent compressive strain to evaluate.
            Acs (np.ndarray): Array of A_c values (e.g., [0.2, 0.4, 0.6, 0.8, 1.0]).
            Bc (float): Fixed B_c value to pair with all A_c values.
        """
        # Equivalent strain range: κ ∈ [0, |eps_c_max|]
        eps = np.arange(0.0, np.abs(eps_c_max), 1e-5)
        d_curves = []
        for Ac in Acs:
            d_vals = []
            for ep in eps:
                d = self.dmg.calculate_d_from_equivalent_strain(eps_eq=ep, eps_d0=self.dmg.mat.prop.eps_tp, A=Ac, B=Bc)
                self.dmg.validate_damage_value(value=d, label='dc')
                d_vals.append(d)
            d_curves.append(d_vals)
        
        # Plot settings
        labels = [f"$A_c = {A:.2f}$" for A in Acs]
        fig, ax = plot_template(
            *[(eps, d_vals) for d_vals in d_curves],
            xlabel='Equivalent Strain $\\kappa$',
            ylabel='Damage $d$',
            legend_labels=labels,
            title=(f'Varying $A_c$ at $B_c$ = {Bc}')
        )
        Ac_min, Ac_max = np.round(Acs[[0, -1]], 2)
        save_path = f'log/fem/damage/mo/Acs_{Ac_min}_{Ac_max}__Bc_{Bc:.0e}.png'
        fig.savefig(save_path, dpi=300)
        plt.show()
    
    def analyze_Bc(self, eps_c_max: float, Bcs: np.ndarray, Ac: float) -> None:
        """
        Analyzes and visualizes the influence of different B_c values on compressive damage evolution,
        keeping A_c constant.

        Args:
            eps_c_max (float): Maximum equivalent compressive strain to evaluate.
            Bcs (np.ndarray): Array of B_c values to test (e.g., [1000, 2000]).
            Ac (float): Fixed A_c value applied to all curves.
        """
        # Equivalent strain range: κ ∈ [0, |eps_c_max|]
        eps = np.arange(0.0, np.abs(eps_c_max), 1e-5)
        d_curves = []
        for Bc in Bcs:
            d_vals = []
            for ep in eps:
                d = self.dmg.calculate_d_from_equivalent_strain(eps_eq=ep, eps_d0=self.dmg.mat.prop.eps_tp, A=Ac, B=Bc)
                self.dmg.validate_damage_value(value=d, label='dc')
                d_vals.append(d)
            d_curves.append(d_vals)
        
        # Plot settings
        labels = [f"$B_c = {B:.2f}$" for B in Bcs]
        fig, ax = plot_template(
            *[(eps, d_vals) for d_vals in d_curves],
            xlabel='Equivalent Strain $\\kappa$',
            ylabel='Damage $d$',
            legend_labels=labels,
            title=(f'Varying $B_c$ at $A_c$ = {Ac}')
        )
        Bc_min, Bc_max = np.round(Bcs[[0, -1]], 0)
        save_path = f'log/fem/damage/mo/Bcs_{Bc_min}_{Bc_max}__Ac_{Ac:.2f}.png'
        fig.savefig(save_path, dpi=300)
        plt.show()
    
    def plot_combined_d(self,):
        pass