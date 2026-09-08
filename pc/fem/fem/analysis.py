# fem/fem/analysis.py
# Need to update if used


import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple


from fem.utils.utils import plot_template, macaulay_bracket


class Damage_Evolution_Analysis:
    """
    Class for analyzing the evolution of damage in concrete using the Mazars isotropic damage model.

    This class implements the damage growth function:
        d(kappa) = 1 - (kappa0 * (1 - A)) / kappa - A * exp(-B * (kappa - kappa0))

    The model describes how material stiffness degrades with increasing equivalent strain (kappa),
    controlled by parameters A and B.

    Attributes:
        None
    """
    def __init__(self) -> None:
        """Initializes the Damage_Evolution_Analysis class."""
        # Set Matplotlib font styles
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'mathtext.fontset': 'custom',
            'mathtext.rm': 'Times New Roman',
            'mathtext.it': 'Times New Roman:italic',
            'mathtext.bf': 'Times New Roman:bold',
        })
    
    def compute_d(self, kappa: float, A: float, B: float, kappa0: float, clip: bool = True) -> float:
        """
        Computes scalar damage variable `d` given equivalent strain `kappa` and model parameters.

        Args:
            kappa (float): Current equivalent strain (must be ≥ 0).
            A (float): Asymptotic damage factor. Affects long-term residual stiffness.
            B (float): Damage rate factor. Controls how quickly damage grows with strain.
            kappa0 (float): Threshold strain where damage initiates (i.e., elastic limit).

        Returns:
            float: Damage value `d` ∈ [0, 1], where 0 means no damage and 1 means fully damaged.

        Raises:
            ValueError: If `kappa` is negative.
        """
        if kappa < 0:
            raise ValueError("⚠️ Equivalent strain (kappa) must be non-negative.")

        if kappa < kappa0:
            return 0.0
        
        # Mazars damage function
        d_raw = 1.0 - (kappa0 * (1 - A)) / kappa - A / np.exp(B * (kappa - kappa0))

        if d_raw < 0 or d_raw > 1:
            print("⚠️ Warning: Computed damage d = {:.4f} is outside [0, 1].".format(d_raw))
            print("    Inputs: kappa = {:.4e}, A = {:.2f}, B = {:.1f}, kappa0 = {:.4e}".format(kappa, A, B, kappa0))

        # Clip to ensure it's in [0, 1]
        if clip:
            d_clipped = np.clip(d_raw, 0.0, 1.0)
            return d_clipped
        else:
            return d_raw
    
    def parameter_A_analysis(self, eps_p: np.ndarray, As: np.ndarray, B: float, kappa0: float) -> Tuple[plt.Figure, plt.Axes]:
        """
        Analyzes the influence of damage parameter A on the Mazars damage evolution.

        This function computes the scalar damage variable `d` across a range of equivalent
        strain values (`eps_p`) and varying values of the material parameter `A`, keeping
        `B` and `kappa0` fixed. The results are plotted as a 3D surface.

        Args:
            eps_p (np.ndarray): Array of equivalent strain (kappa) values.
            As (np.ndarray): Array of A values to test (damage parameter A).
            B (float): Damage evolution rate parameter.
            kappa0 (float): Initial threshold equivalent strain.

        Returns:
            Tuple[plt.Figure, plt.Axes]: The generated Matplotlib figure and axes.
        """
        # Initialize damage matrix: rows → A values, cols → kappa values
        d_surface = np.zeros((len(As), len(eps_p)))

        # Compute damage surface
        for i, A in enumerate(As):
            for j, kappa in enumerate(eps_p):
                d_surface[i, j] = self.compute_d(kappa=kappa, A=A, B=B, kappa0=kappa0)

        # Prepare grid for plotting
        A_grid, K_grid = np.meshgrid(eps_p, As)

        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(K_grid, A_grid, d_surface, cmap='viridis', edgecolor='none', alpha=0.95)

        # Axis formatting
        ax.set_xlabel('Damage Parameter $A$', labelpad=10)
        ax.set_ylabel('Equivalent Strain $\\kappa$', labelpad=10)
        ax.set_zlabel('Damage $d$', labelpad=10)
        ax.set_title(f'Mazars Damage Evolution with Varying $A$ at $B$ = {B}')
        
        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, label='Damage Level')
        
        # Save and display
        Amin, Amax = np.round(As[[0, -1]], 2)
        save_path = f'log/damage/As_{Amin}_{Amax}__B_{B:.0e}.png'
        fig.savefig(save_path, dpi=300)
        plt.tight_layout()
        plt.show()

        return fig, ax
    
    def parameter_B_analysis(self, eps_p: np.ndarray, Bs: np.ndarray, A: float, kappa0: float) -> Tuple[plt.Figure, plt.Axes]:
        """
        Analyzes the influence of damage parameter B on the Mazars damage evolution.

        This function computes the scalar damage variable `d` across a range of equivalent
        strain values (`eps_p`) and varying values of the material parameter `B`, keeping
        `A` and `kappa0` fixed. The results are plotted as a 3D surface.

        Args:
            eps_p (np.ndarray): Array of equivalent strain (kappa) values.
            Bs (np.ndarray): Array of A values to test (damage parameter A).
            A (float): Damage evolution rate parameter.
            kappa0 (float): Initial threshold equivalent strain.

        Returns:
            Tuple[plt.Figure, plt.Axes]: The generated Matplotlib figure and axes.
        """
        # Initialize damage matrix: rows → A values, cols → kappa values
        d_surface = np.zeros((len(Bs), len(eps_p)))

        # Compute damage surface
        for i, B in enumerate(Bs):
            for j, kappa in enumerate(eps_p):
                d_surface[i, j] = self.compute_d(kappa=kappa, A=A, B=B, kappa0=kappa0)

        # Prepare grid for plotting
        A_grid, K_grid = np.meshgrid(eps_p, Bs)

        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(K_grid, A_grid, d_surface, cmap='viridis', edgecolor='none', alpha=0.95)

        # Axis formatting
        ax.set_xlabel('Damage Parameter $B$', labelpad=10)
        ax.set_ylabel('Equivalent Strain $\\kappa$', labelpad=10)
        ax.set_zlabel('Damage $d$', labelpad=10)
        ax.set_title(f'Mazars Damage Evolution with Varying $B$ at $A$ = {A}')
        
        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, label='Damage Level')
        
        # Save and display
        Bmin, Bmax = np.round(Bs[[0, -1]], 2)
        save_path = f'log/damage/Bs_{Bmin}_{Bmax}__A_{A:.1}.png'
        fig.savefig(save_path, dpi=300)
        plt.tight_layout()
        plt.show()

        return fig, ax
    
    def parameter_AB_analysis(self, d: float, As: np.ndarray, Bs: np.ndarray, kappa0: float) -> Tuple[plt.Figure, plt.Axes]:
        """
        Analyzes the equivalent strain required to reach a given damage level `d` 
        across combinations of Mazars parameters A and B.

        This function inverts the damage law to find the equivalent strain `kappa`
        that produces the specified damage `d` for each (A, B) pair, assuming fixed `kappa0`.

        Args:
            d (float): Target damage level (0 < d < 1).
            As (np.ndarray): Range of A values to analyze.
            Bs (np.ndarray): Range of B values to analyze.
            kappa0 (float): Initial damage threshold strain.

        Returns:
            Tuple[plt.Figure, plt.Axes]: The generated 3D plot of equivalent strain vs A & B.
        """
        # Initialize result surface
        K_eq_surface = np.zeros((len(As), len(Bs)))

        for i, A in enumerate(As):
            for j, B in enumerate(Bs):
                if d == 0:
                    K_eq_surface[i, j] = 0.0
                elif d >= 1.0:
                    K_eq_surface[i, j] = np.nan
                else:
                    # Solve for kappa using numerical root-finding
                    def objective(kappa):
                        return self.compute_d(kappa, A, B, kappa0, clip=False) - d
                    
                    try:
                        from scipy.optimize import root_scalar
                        result = root_scalar(objective, bracket=[kappa0, 100 * kappa0], method='brentq')
                        K_eq_surface[i, j] = result.root if result.converged else np.nan
                    except Exception:
                        K_eq_surface[i, j] = np.nan

        # Create meshgrid for plotting
        A_grid, B_grid = np.meshgrid(As, Bs, indexing='ij')

        # Plot the surface
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(A_grid, B_grid, K_eq_surface, cmap='plasma', edgecolor='none', alpha=0.95)

        # Labels and title
        ax.set_xlabel('Damage Parameter $A$', labelpad=10)
        ax.set_ylabel('Damage Parameter $B$', labelpad=10)
        ax.set_zlabel('Equivalent Strain $\\kappa$', labelpad=10)
        ax.set_title(f'Equivalent Strain to Reach Damage $d={d}$')
        fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label='$\\kappa$')

        # Save and show
        Amin, Amax = np.round([As[0], As[-1]], 2)
        Bmin, Bmax = np.round([Bs[0], Bs[-1]], -2)
        fig.savefig(f'log/fem/damage/kappa_at_d_{d:.2f}_A_{Amin}_{Amax}_B_{Bmin}_{Bmax}.png', dpi=300)
        plt.tight_layout()
        plt.show()

        return fig, ax

    def show_uniaxial_compressive_equivalent_strain(self,
            eps0: float = 0.0,
            eps1: float = 0.004,
            d_eps: float = 1e-5,
            nu: float = 0.25
            ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualizes the equivalent strain under uniaxial compression loading,
        based on the Mazars damage model using only positive (tensile) principal strains.

        This function assumes uniaxial compression in the X-direction and applies
        Poisson's effect in the Y and Z directions. It converts the strain state into
        an equivalent tensile strain using the Euclidean norm of positive principal strains.

        Args:
            eps0 (float, optional): Initial compressive strain (positive scalar). Defaults to 0.0.
            eps1 (float, optional): Final compressive strain. Defaults to 0.01.
            d_eps (float, optional): Incremental strain step. Defaults to 1e-5.
            nu (float, optional): Poisson’s ratio. Defaults to 0.25.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Matplotlib figure and axes objects.

        Saves:
            A figure is saved to 'log/damage/uniaxial_compressive_equivalent_strain.png'.
        """
        eps_range = np.arange(eps0, eps1, d_eps)
        eq_strains = []
        
        for ep in np.abs(eps_range):  # Use absolute value to keep strain positive for compression
            # Construct diagonal strain tensor: [ε, -ν·ε, -ν·ε]
            strain_tensor = np.diag([-ep, nu * ep, nu * ep])
            
            # Compute principal strains
            principal_strains = compute_principals(strain_tensor)

            # Equivalent strain for damage (only tensile parts)
            eq_strain = np.linalg.norm(macaulay_bracket(principal_strains))
            eq_strains.append(eq_strain)
        
        # Plot using a standard plotting utility
        fig, ax = plot_template([eps_range, eq_strains],
                    figsize=(10,6),
                    xlabel='Compressive strain',
                    ylabel='Equivalent strain',
                    xlim=(eps0, eps1),
                    ylim=(0, 0.0015),
                    title='Equivalent Strain under Uniaxial Compression'
                    )
        
        # Save and show figure
        fig.savefig('log/fem/damage/uniaxial_compressive_equivalent_strain.png', dpi=300)
        plt.show()
    
    def show_uniaxial_tensile_equivalent_strain(self,
            eps0: float = 0.0,
            eps1: float = 0.001,
            d_eps: float = 1e-5,
            nu: float = 0.25) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots the equivalent strain evolution under uniaxial tension based on the Mazars damage model.

        This function simulates a uniaxial tensile strain state in the X-direction,
        applying lateral contractions in Y and Z using Poisson's ratio. It computes
        the equivalent strain as the norm of the positive principal strains.

        Args:
            eps0 (float, optional): Initial tensile strain. Defaults to 0.0.
            eps1 (float, optional): Final tensile strain. Defaults to 0.001.
            d_eps (float, optional): Strain increment. Defaults to 1e-5.
            nu (float, optional): Poisson’s ratio. Defaults to 0.25.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Matplotlib figure and axes objects.

        Saves:
            Figure saved to: 'log/damage/uniaxial_tensile_equivalent_strain.png'
        """
        eps_range = np.arange(eps0, eps1, d_eps)
        eq_strains = []

        for ep in eps_range:
            # Construct strain tensor under uniaxial tension with lateral contraction
            strain_tensor = np.diag([ep, -nu * ep, -nu * ep])

            # Compute principal strains
            principal_strains = compute_principals(strain_tensor)

            # Compute Mazars equivalent strain: norm of positive principal strains
            eq_strain = np.linalg.norm(macaulay_bracket(principal_strains))
            eq_strains.append(eq_strain)

        # Plot result
        fig, ax = plot_template(
            [eps_range, eq_strains],
            figsize=(10,6),
            xlabel='Tensile Strain',
            ylabel='Equivalent Strain',
            xlim=(eps0, eps1),
            ylim=(0, 0.001),
            title='Equivalent Strain under Uniaxial Tension'
        )

        # Save and show
        fig.savefig('log/fem/damage/uniaxial_tensile_equivalent_strain.png', dpi=300)
        plt.show()

        return fig, ax
    
    def show_damage2D_A(self, kappas: np.ndarray, As: np.ndarray, B: float, kappa0: float, clip: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot 2D curves of Mazars damage evolution for different A values.

        This function evaluates and visualizes the damage variable `d` across a range
        of equivalent strains (`kappas`) for various values of the damage parameter `A`,
        with fixed `B` and threshold `kappa0`. Useful for studying the influence of A
        on the softening rate in Mazars damage model.

        Args:
            kappas (np.ndarray): Array of equivalent strain values.
            As (np.ndarray): Array of A values to be evaluated (damage parameters).
            B (float): Damage exponent parameter.
            kappa0 (float): Threshold strain where damage begins.
            clip (bool, optional): Whether to clip damage between [0, 1]. Defaults to True.

        Returns:
            Tuple[plt.Figure, plt.Axes]: The generated matplotlib figure and axis objects.
        """
        damage_curves = []
        for A in As:
            d_vals = [self.compute_d(kappa, A, B, kappa0, clip) for kappa in kappas]
            damage_curves.append(d_vals)

        # Prepare legend labels for each A
        labels = [f"$A = {A:.2f}$" for A in As]

        # Unpack all curves into the plotting function
        fig, ax = plot_template(
            *[(kappas, d_vals) for d_vals in damage_curves],
            xlabel='Equivalent Strain $\\kappa$',
            ylabel='Mazars Damage $d$',
            legend_labels=labels,
            title=(f'Varying $A$ at $B$ = {B}')
        )
        
        # Save and display
        Amin, Amax = np.round(As[[0, -1]], 2)
        save_path = f'log/fem/damage/As_{Amin}_{Amax}__B_{B:.0e}.png'
        fig.savefig(save_path, dpi=300)
        plt.show()
        
        return fig, ax
    
    def show_damage2D_B(self, kappas: np.ndarray, Bs: np.ndarray, A: float, kappa0: float, clip: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot 2D curves of Mazars damage evolution for different A values.

        This function evaluates and visualizes the damage variable `d` across a range
        of equivalent strains (`kappas`) for various values of the damage parameter `B`,
        with fixed `A` and threshold `kappa0`. Useful for studying the influence of B
        on the softening rate in Mazars damage model.

        Args:
            kappas (np.ndarray): Array of equivalent strain values.
            Bs (np.ndarray): Array of A values to be evaluated (damage parameters).
            A (float): Damage exponent parameter.
            kappa0 (float): Threshold strain where damage begins.
            clip (bool, optional): Whether to clip damage between [0, 1]. Defaults to True.

        Returns:
            Tuple[plt.Figure, plt.Axes]: The generated matplotlib figure and axis objects.
        """
        damage_curves = []
        for B in Bs:
            d_vals = [self.compute_d(kappa, A, B, kappa0, clip) for kappa in kappas]
            damage_curves.append(d_vals)

        # Prepare legend labels for each B
        labels = [f"$B = {B}$" for B in Bs]

        # Unpack all curves into the plotting function
        fig, ax = plot_template(
            *[(kappas, d_vals) for d_vals in damage_curves],
            xlabel='Equivalent Strain $\\kappa$',
            ylabel='Mazars Damage $d$',
            legend_labels=labels,
            title=(f'Varying $B$ at $A$ = {A}')
        )
        
        # Save and display
        Bmin, Bmax = np.round(Bs[[0, -1]], 2)
        save_path = f'log/fem/damage/Bs_{Bmin}_{Bmax}__A_{A:.2f}.png'
        fig.savefig(save_path, dpi=300)
        plt.show()
        
        return fig, ax
    
    def show_damage2D_AB(self,
            kappas: np.ndarray,
            As: np.ndarray,
            Bs: np.ndarray,
            kappa0: float,
            clip: bool = True
        ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot 2D curves of Mazars damage evolution for different A values and multiple B values.

        Args:
            compute_d (Callable): Function to compute damage given kappa, A, B, and kappa0.
            kappas (np.ndarray): Array of equivalent strain values.
            As (np.ndarray): Array of A values to be evaluated (damage parameters).
            Bs (np.ndarray): Array of B values to evaluate.
            kappa0 (float): Threshold strain where damage begins.
            clip (bool): Whether to clip damage between [0, 1]. Defaults to True.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Matplotlib figure and axis.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.rcParams['font.family'] = 'Times New Roman'

        markers = [None, 'o', '^', 's']
        linestyles = ['-', '--', '--', '--']
        colors = plt.cm.viridis(np.linspace(0, 1, len(As)))

        for b_idx, B in enumerate(Bs):
            marker = markers[b_idx] if b_idx < len(markers) else 'x'
            linestyle = linestyles[b_idx] if b_idx < len(linestyles) else '--'

            for a_idx, A in enumerate(As):
                d_vals = [self.compute_d(kappa, A, B, kappa0, clip) for kappa in kappas]
                label = f"$A={A:.2f}, B={B:.0f}$"
                ax.plot(kappas, d_vals, label=label, linestyle=linestyle,
                        marker=marker if marker else None,
                        markerfacecolor='none',
                        markersize=5,
                        color=colors[a_idx])

        ax.set_xlabel('Equivalent Strain $\\kappa$', fontsize=14)
        ax.set_ylabel('Mazars Damage $d$', fontsize=14)
        ax.set_title('Varying $A$ and $B$', fontsize=16)
        ax.legend(fontsize=15)
        ax.grid(True)
        fig.tight_layout()
        
        # Save and display
        Amin, Amax = np.round(As[[0, -1]], 2)
        Bmin, Bmax = np.round(Bs[[0, -1]], 0)
        save_path = f'log/fem/damage/As_{Amin}_{Amax}__Bs_{Bmin}_{Bmax}.png'
        fig.savefig(save_path, dpi=300)
        plt.show()
        
        return fig, ax

def dt_analysis(kappa0=0.5e-4):
    dea = Damage_Evolution_Analysis()
    
    eps_p = np.linspace(0, 0.005, 100)
    
    # As
    Ats = np.linspace(0.7, 1.2, 50)
    Bt = 1e4
    dea.parameter_A_analysis(eps_p=eps_p, As=Ats, B=Bt, kappa0=kappa0)
    
    # Bs
    Bts = np.linspace(1e4, 5e4, 50)
    At = 0.7
    dea.parameter_B_analysis(eps_p=eps_p, Bs=Bts, A=At, kappa0=kappa0)
    
    # As, Bs
    ds = np.linspace(0.1, 0.9, 9)
    As = np.linspace(0.7, 1.5, 50)
    Bs = np.linspace(1e3, 5e4, 100)
    for d in ds:
        dea.parameter_AB_analysis(d=d, As=As, Bs=Bs, kappa0=kappa0)

def dc_analysis(kappa0=0.5e-4):
    dea = Damage_Evolution_Analysis()
    
    eps_p = np.linspace(0, 0.005, 100)
    
    Acs = np.linspace(1.0, 1.2, 50)
    Bc = 1e3
    dea.parameter_A_analysis(eps_p=eps_p, As=Acs, B=Bc, kappa0=kappa0)
    
    Bcs = np.linspace(1e3, 2e3, 50)
    Ac = 1.0
    dea.parameter_B_analysis(eps_p=eps_p, Bs=Bcs, A=Ac, kappa0=kappa0)
    dea.show_uniaxial_compressive_equivalent_strain()

def main():
    dt_analysis()
    dc_analysis()
    pass

if __name__ == '__main__':
    main()