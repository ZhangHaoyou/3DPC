# fem/damage.py


import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy.optimize import curve_fit
from typing import Tuple, Optional
from abc import ABC, abstractmethod


from pc_fem.parameters import Damage_Parameter
from pc_fem.fem.materials import Material
from pc_fem.fem.geo_mesh import Mesh
from pc_fem.utils.log_config import setup_logger
from pc_fem.utils.utils import plot_template, macaulay_bracket, calculate_equivalent_strain_from_voigt, calculate_principal_values



def clamp(d: torch.Tensor, min: torch.Tensor, max: float) -> torch.Tensor:
    d = torch.maximum(d, min)
    d = torch.minimum(d, torch.tensor(max, device=d.device))
    return d

class Damage(ABC):
    """Abstract base class for damage models in continuum materials.

    This class defines a general interface for damage evolution models,
    which compute a scalar damage variable based on input strain tensors
    and material parameters.

    Attributes:
        mat (Material): The constitutive material model associated with this damage formulation.
        para (Damage_Parameter): Parameters specific to the damage evolution law.
    """
    
    def __init__(self, mat: Material, para: Damage_Parameter):
        """Initializes the damage model with material properties and damage parameters.

        Args:
            mat (Material): The constitutive material model.
            para (Damage_Parameter): Damage parameters governing the evolution law.
        """
        self.mat = mat
        self.para = para
    
    def print_parameters(self) -> None:
        """Prints the material and damage parameters in a formatted layout.

        Iterates through both the material properties (`mat.prop`) and damage
        parameters (`para`), displaying them with aligned formatting for clarity.
        """
        print(f"\nDamage and Material Parameters ({self.__class__.__name__}):")
        for key, value in vars(self.mat.prop).items():
            print(f"{key:<20}: {value}")
        for key, value in vars(self.para).items():
            print(f"{key:<20}: {value}")
    
    def validate_damage_value(self, value: float, label: str = "d") -> bool:
        """Validates whether a scalar damage value lies within the range [0, 1].

        This utility checks if a given damage variable (e.g., `d`, `dt`, or `dc`) 
        is within the physically valid range [0, 1]. If not, it prints a warning message.

        Args:
            value (float): The damage variable to check.
            label (str, optional): Name of the variable for warning messages. Defaults to "d".

        Returns:
            bool: True if the value is in [0, 1], False otherwise.
        """
        if value > 1.0:
            print(f"⚠️  Warning: Damage `{label}` = {value:.4f} exceeds 1.0")
            return False
        elif value < 0.0:
            print(f"⚠️  Warning: Damage `{label}` = {value:.4f} is negative")
            return False
        return True
    
    def calculate_Ac_Bc_At_Bt(self) -> Tuple[float, float, float, float]:
        """Calculates the Mazars damage model parameters Ac, Bc, At, and Bt.

        This method computes the damage evolution parameters based on Eurocode 2
        and the formulation from the original Mazars isotropic damage model.
        The compressive parameters Ac and Bc are derived using an exponential 
        softening law, and the tensile parameters At and Bt are calculated 
        using a bilinear law based on the brittleness index.

        Returns:
            Tuple[float, float, float, float]: 
                - Ac: Compressive softening coefficient.
                - Bc: Compressive brittleness coefficient.
                - At: Tensile coefficient (typically 1.0).
                - Bt: Tensile brittleness coefficient (1 + ib) / ε_tp.
        """
        # Material properties
        E = self.mat.prop.E           # Young's modulus (MPa)
        nu = self.mat.prop.nu         # Poisson's ratio
        fc = self.mat.prop.fc         # Compressive strength (MPa)
        eps_cp = self.mat.prop.eps_cp # Peak compressive strain
        eps_tp = self.mat.prop.eps_tp # Peak tensile strain
        
        # Compressive damage model parameters
        Bc = 1.0 / (np.sqrt(2) * nu * eps_cp)
        numerator = eps_tp - fc * np.sqrt(2) * nu / E
        denominator = Bc * eps_tp - np.exp(Bc * eps_tp - 1.0)
        Ac = Bc * numerator / denominator

        # Tensile damage model parameters
        ib = self.para.ib
        At = 1.0
        Bt = (1 + ib) / eps_tp
        
        print(f'Calculated Mazars parameters: Ac = {Ac:.2f}, Bc = {Bc:.2f}, At = {At:.2f}, Bt = {Bt:.2f}')
        
        return float(Ac), float(Bc), float(At), float(Bt)
    
    def calculate_alpha(self, eps: np.ndarray, d0: float) -> Tuple[float, float]:
        """Computes tensile and compressive damage weights (α_t, α_c) from principal strains.

        This method implements the decomposition strategy from Mazars (1986) and Ramirez (2017),
        based on strain tensors reconstructed from Voigt components. The resulting weights are
        used to proportionally attribute damage to tension and compression.

        Args:
            eps (np.ndarray): A 6-component engineering strain vector in Voigt notation.
            d0 (float): Damage value from the previous increment (0 ≤ d < 1).

        Returns:
            Tuple[float, float]: 
                - alpha_t (float): Tensile damage weighting factor (α_t).
                - alpha_c (float): Compressive damage weighting factor (α_c).

        Raises:
            ValueError: If `strain_voigt` is not a 1D NumPy array with 6 components.

        Notes:
            - Principal values are computed from reconstructed tensors.
            - Decomposition uses secant stiffness: ε = (1 / (1 - d)) * Γ⁻¹ * σ.
            - Equivalent strain is defined as the norm of positive principal strains.
            - Weighting formulation:
                α_t = Σ(ε⁺_i * ε_ti) / ||ε⁺||²
                α_c = Σ(ε⁺_i * ε_ci) / ||ε⁺||²
        """
        # Reference: Mazars, 1986; Ramirez, 2017
        if eps.shape != (6,):
            raise ValueError("Input `eps` must be a 6-component Voigt vector (shape: (6,))")
        
        # Material stiffness matrix
        E = self.mat.prop.E
        nu = self.mat.prop.nu
        D = self.mat.calculate_D(E=E)

        # Compute stress using secant formulation
        sig = (1-d0) * D @ eps # (6,)
        
        # Get principal values
        sig_p = calculate_principal_values(vec=sig, quantity='stress') # (3,) 
        eps_p = calculate_principal_values(vec=eps, quantity='strain') # (3,) 
        
        # Tension-only principal stress and total sum
        sig_p_pos = macaulay_bracket(sig_p) # (3,) Positive (tensile) part of principal stresses: σ⁺ = max(σ, 0)
        sigma_pos_sum = np.sum(sig_p_pos)   # Scalar

        # Reconstruct tensile strain (ε_tensile) from stress (Ramirez 2017, Eq. 5)
        eps_t = ((1.0 + nu) * sig_p_pos - nu * sigma_pos_sum) / E  # (3,)

        # Compute ε_compressive as residual
        eps_c = eps_p - eps_t  # (3,)
        
        # Equivalent strain: norm of positive principal strains
        eps_pos = macaulay_bracket(eps_p)  # (3,) eps_plus[i] = max(eps_total[i], 0)
        eps_eq = np.linalg.norm(eps_pos) # eq = norm(eps_plus)
        
        if eps_eq == 0.0:
            return 0.0, 0.0
        
        # Compute alpha weights
        alpha_t = np.sum(eps_t * eps_pos) / (eps_eq * eps_eq)
        alpha_c = np.sum(eps_c * eps_pos) / (eps_eq * eps_eq)
        
        # Warning if sum > 1 (may indicate inconsistency)
        if alpha_t + alpha_c > 1.0 + 1e-5:
            print("⚠️ Warning: α_t + α_c exceeds 1.0")
            print(f"  alpha_t = {alpha_t:.4f}, alpha_c = {alpha_c:.4f}")

        return float(alpha_t), float(alpha_c)
    
    @abstractmethod
    def calculate_damage(self, eps: np.ndarray, d0: np.ndarray) -> np.ndarray:
        """Calculates the updated scalar damage variable based on the current strain state.

        This method defines the interface for computing the damage evolution.
        Implementations should define the damage update law based on the current
        strain state and the previous damage value.

        Args:
            eps (np.ndarray): The current 6-component strain vector in Voigt notation.
            d0 (np.ndarray): Previous damage state (3,), [d_tc, d_t, d_c].

        Returns:
            np.ndarray: Updated damage vector [d_tc, d_t, d_c].
        """
        pass
    
    def plot_damage_evolution_from_elastic_eps(self, clip: bool = True,
                save_path: str ='log/damage/mo/Elastic_eps_damage_evolution.png',
                show: bool = True, close: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots damage evolution against compression strain using precomputed elastic strain data.

        This function loads a list of elastic strain vectors from file, computes corresponding
        damage values using the current damage model, and visualizes the damage evolution
        with respect to the compressive strain component (ε_z).

        Args:
            clip (bool): Whether to clip damage values to [d0, 0.99]. Default is False.
            save_path (str): Path to save the generated figure. Default is predefined.
            show (bool): Whether to display the plot interactively. Default is True.
            close (bool): Whether to close the figure after plotting. Default is True.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Matplotlib figure and axes objects.
        """
        filepath = 'log/kfu/elastic_eps_voigt.csv'

        try:
            eps_array = np.loadtxt(filepath, delimiter=',')
        except Exception as e:
            raise IOError(f"Failed to load strain data from '{filepath}': {e}")

        if eps_array.ndim == 1:
            eps_array = eps_array[np.newaxis, :]  # Ensure shape (N, 6)

        d0 = np.zeros(3)
        d_results = []

        for eps_voigt in eps_array:
            d = self.calculate_damage(eps=eps_voigt, d0=d0, clip=clip)
            d0 = d
            d_results.append(d)

        d_results = np.array(d_results)
        strain_z = -eps_array[:, 2]  # Compression is negative → flip for plotting

        fig, ax = plot_template(
            [strain_z, d_results[:, 0]],
            xlabel='Compression strain (-ε_z)',
            ylabel='Damage',
            title='Damage Evolution from Elastic Strains'
        )

        fig.savefig(save_path, dpi=300)
        self.logger.info(f"[Elastic Damage Plot] Saved damage evolution plot to: {save_path}")

        if show:
            plt.show()
        if close:
            plt.close(fig)

        return fig, ax
    
    def calculate_d_from_equivalent_strain(self, eps_eq: float, eps_d0: float, A: float, B: float):
        """Computes the scalar damage variable `d` using the Mazars isotropic damage law.

        The damage variable `d` quantifies stiffness degradation as a function of the
        maximum equivalent strain. This formulation follows the exponential softening 
        rule defined by the Mazars model and applies only to scalar inputs.

        Args:
            eps_eq (float): Current maximum equivalent strain.
            eps_d0 (float): Damage initiation threshold (e.g., peak elastic strain).
            A (float): Damage shape parameter. Controls asymptotic behavior 
                (e.g., 0.7–1.0 for tension, 1.0–2.0 for compression).
            B (float): Damage growth rate parameter. Controls exponential decay 
                (e.g., 1000–20000).

        Returns:
            float: Scalar damage value `d` in the range [0, 1].

        Raises:
            ValueError: If input strains are negative or A/B are non-positive.
        """
        pass
    
    def calculate_uniaxial_compression_damage(self, eps: np.ndarray, Ac: float, Bc: float) -> np.ndarray:
        """Computes the damage variable `d` for uniaxial compression.

        This method evaluates the Mazars isotropic damage variable `d` for a 
        given array of compressive strains, using the equivalent strain 
        definition for compression: 
        ε_eq = √2 * ν * |ε|.

        Args:
            eps (np.ndarray): Array of compressive strain values (should be ≤ 0).
            Ac (float): Damage shape parameter in compression.
            Bc (float): Damage curvature parameter in compression.

        Returns:
            np.ndarray: Array of computed damage values corresponding to input strains.
        """
        pass
    
    def calculate_uniaxial_tension_damage(self, eps: np.ndarray, At: float, Bt: float) -> np.ndarray:
        """Computes the damage variable `d` for uniaxial tension.

        This method calculates the damage evolution based on the Mazars isotropic 
        damage model for a given array of tensile strains. Equivalent strain is 
        taken as the absolute value of the tensile strain.

        Args:
            eps (np.ndarray): Array of tensile strain values (should be ≥ 0).
            At (float): Damage shape parameter in tension.
            Bt (float): Damage curvature parameter in tension.

        Returns:
            np.ndarray: Array of computed damage values corresponding to input strains.
        """
        pass
    
    def plot_right_damage_evolution(self, ax: plt.Axes, eps: np.ndarray, d_vals: np.ndarray) -> plt.Axes:
        """Plots damage evolution curve (d vs. strain) on the right y-axis.

        This method overlays a secondary y-axis on the input Matplotlib axis `ax` to
        display the damage variable `d` against strain. The damage curve is rendered
        as a red dashed line with ticks and labels styled for visual clarity.

        Args:
            ax (plt.Axes): Primary Matplotlib axis (typically with stress-strain curve).
            eps (np.ndarray): Strain values corresponding to damage values.
            d_vals (np.ndarray): Damage values (range [0, 1]) to be plotted.

        Returns:
            plt.Axes: The created secondary y-axis displaying the damage evolution curve.
        """
        # Create secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(
            eps, d_vals,
            linestyle='--',
            color=(192 / 255, 0, 0),
            label='Damage $d$'
        )
        ax2.set_ylabel("Damage $d$", color=(192 / 255, 0, 0), fontsize=12)

        # Axis tick styling
        ax2.set_ylim(0.0, 1.05)
        ax2.tick_params(axis='y', direction='in', which='major', length=4)
        ax2.tick_params(axis='y', direction='in', which='minor', length=3)
        ax2.minorticks_on()
        ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
        
        return ax2
    
    def plot_uniaxial_compression_damage_evolution(self, eps_c_max: float, d_eps: float, Ac: float, Bc: float,
                eps_c_min: float, save_path: str, show: bool, close: bool) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
        """Plots uniaxial compressive stress-strain curve with damage evolution.

        This abstract method overlays the damage variable curve (d vs. ε) on top of the 
        compressive stress-strain curve, using a secondary y-axis. The damage curve 
        is typically plotted as a red dashed line to distinguish it from the stress response.

        Args:
            eps_c_max (float): Maximum compressive strain (should be negative).
            d_eps (float): Strain step size.
            Ac (float): Damage shape parameter in compression.
            Bc (float): Damage curvature parameter in compression.
            eps_c_min (float): Minimum compressive strain.
            save_path (str): Path to save the generated figure.
            show (bool): If True, displays the plot in an interactive window.
            close (bool): If True, closes the figure after saving.

        Returns:
            Tuple[plt.Figure, plt.Axes, plt.Axes]: The generated Matplotlib figure, the primary (stress) axis and (damage) axis.
        """
        pass
    
    def plot_uniaxial_tension_damage_evolution(self, eps_t_max: float, d_eps: float, At: float, Bt: float,
                eps_t_min: float, save_path: str, show: bool, close: bool) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
        """Plots uniaxial tensile stress-strain curve with damage evolution.

        This method overlays the tensile damage variable curve (d vs. ε) on top of 
        the uniaxial tensile stress-strain response, using a secondary y-axis. 
        The damage curve is rendered as a red dashed line for clear visual distinction.

        Args:
            eps_t_max (float): Maximum tensile strain.
            d_eps (float): Strain step size.
            At (float): Damage shape parameter in tension.
            Bt (float): Damage curvature parameter in tension.
            eps_t_min (float, optional): Minimum tensile strain.
            save_path (str, optional): File path to save the generated figure.
            show (bool, optional): If True, displays the figure interactively.
            close (bool, optional): If True, closes the figure after saving to avoid memory leaks.

        Returns:
            Tuple[plt.Figure, plt.Axes, plt.Axes]: The generated Matplotlib figure, the primary (stress) axis and (damage) axis.
        """
        pass
    
class Mazars_Original_Damage(Damage):
    """Mazars isotropic damage model for concrete.

    Implements the original Mazars scalar damage formulation to simulate
    stiffness degradation in concrete under tension and compression.
    Damage evolves separately in tension and compression using exponential-type
    laws, with parameters controlling the softening rate.

    Attributes:
        mat (Material): Material instance with elastic and strength properties.
        para (Damage_Parameter): Damage parameters including Ac, Bc, At, Bt.
        logger (logging.Logger): Logger for tracking model initialization and events.
    """
    
    def __init__(self, mat: Material, para: Damage_Parameter) -> None:
        """Initializes the Mazars damage model using material properties and 
        damage evolution parameters.

        Args:
            mat (Material): Instance of the base material class containing 
                elastic and strength parameters.
            para (Damage_Parameter): Instance of damage coefficients including:
                - Ac (float): Compression damage coefficient.
                - Bc (float): Compression damage exponent.
                - At (float): Tension damage coefficient.
                - Bt (float): Tension damage exponent.
                - ib (float): Optional shape parameter for Bt calculation if Bt is not given.
                - control (str): Control strategy, one of:
                    - 't' / 'tension'
                    - 'c' / 'compression'
                    - 'ct', 'tc', or 'compression-tension' (combined mode).
        """
        super().__init__(mat, para)
        self.mat = mat
        self.para = para
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info(
            "Initialized Mazars Original Damage with parameters: %s",
            vars(self.para)
        )
    
    def calculate_d_from_equivalent_strain(self, eps_eq: float, eps_d0: float, A: float, B: float):
        """Computes the scalar damage variable `d` using the Mazars isotropic damage law.

        The damage variable `d` quantifies stiffness degradation as a function of the
        maximum equivalent strain. This formulation follows the exponential softening 
        rule defined by the Mazars model and applies only to scalar inputs.

        Args:
            eps_eq (float): Current maximum equivalent strain.
            eps_d0 (float): Damage initiation threshold (e.g., peak elastic strain).
            A (float): Damage shape parameter. Controls asymptotic behavior 
                (e.g., 0.7–1.0 for tension, 1.0–2.0 for compression).
            B (float): Damage growth rate parameter. Controls exponential decay 
                (e.g., 1000–20000).

        Returns:
            float: Scalar damage value `d` in the range [0, 1].

        Raises:
            ValueError: If input strains are negative or A/B are non-positive.
        """
        if eps_eq < 0 or eps_d0 < 0:
            raise ValueError("Strains must be non-negative.")
        if A <= 0 or B <= 0:
            raise ValueError("Parameters A and B must be positive.")

        if eps_eq < eps_d0:
            return 0.0

        d = 1 - (1 - A) * eps_d0 / eps_eq - A * np.exp(-B * (eps_eq - eps_d0))
        
        return float(d)
    
    def calculate_uniaxial_compression_damage(self, eps: np.ndarray, Ac: float, Bc: float) -> np.ndarray:
        """Computes the damage variable `d` for uniaxial compression.

        This method evaluates the Mazars isotropic damage variable `d` for a 
        given array of compressive strains, using the equivalent strain 
        definition for compression: 
        ε_eq = √2 * ν * |ε|.

        Args:
            eps (np.ndarray): Array of compressive strain values (should be ≤ 0).
            Ac (float): Damage shape parameter in compression.
            Bc (float): Damage curvature parameter in compression.

        Returns:
            np.ndarray: Array of computed damage values corresponding to input strains.
        """
        # Compute equivalent strain under compression (Mazars formulation)
        eps_eq = np.sqrt(2) * self.mat.prop.nu * np.abs(eps)
        
        # Compute damage variable d for each equivalent strain
        d_vals = np.array([
            self.calculate_d_from_equivalent_strain(eps_eq=e, eps_d0=self.mat.prop.eps_tp, A=Ac, B=Bc)
            for e in eps_eq
        ])
        
        return d_vals
    
    def calculate_uniaxial_tension_damage(self, eps: np.ndarray, At: float, Bt: float) -> np.ndarray:
        """Computes the damage variable `d` for uniaxial tension.

        This method calculates the damage evolution based on the Mazars isotropic 
        damage model for a given array of tensile strains. Equivalent strain is 
        taken as the absolute value of the tensile strain.

        Args:
            eps (np.ndarray): Array of tensile strain values (should be ≥ 0).
            At (float): Damage shape parameter in tension.
            Bt (float): Damage curvature parameter in tension.

        Returns:
            np.ndarray: Array of computed damage values corresponding to input strains.
        """
        # Compute equivalent strain in tension (absolute value)
        eps_eq = np.abs(eps)

        # Compute damage variable d for each equivalent strain
        d_vals = np.array([
            self.calculate_d_from_equivalent_strain(eps_eq=e, eps_d0=self.mat.prop.eps_tp, A=At, B=Bt)
            for e in eps_eq
        ])
        
        return d_vals
    
    def calculate_damage(self, eps: np.ndarray, d0: np.ndarray, clip: bool = True) -> np.ndarray:
        """
        Computes scalar damage based on the current strain using the Mazars model.

        The damage model supports three control strategies:
        - Tension-only ('t', 'tension')
        - Compression-only ('c', 'compression')
        - Combined tension-compression ('ct', 'tc', 'compression-tension')

        Each component (tensile or compressive) is calculated using equivalent strain and
        exponential evolution laws. The final damage value is clipped in [d0, 0.99] by default.

        Args:
            eps (np.ndarray): Current strain in Voigt notation (6,).
            d0 (np.ndarray): Previous damage state (3,), [d_tc, d_t, d_c].
            clip (bool, optional): Whether to clip updated damage to [d0, 0.99]. Defaults to True.

        Returns:
            np.ndarray: Updated damage vector [d_tc, d_t, d_c].

        Raises:
            ValueError: If `eps` is not a 6-vector or if control mode is invalid.
        """
        # Ensure input is a 1D vector in Voigt notation
        eps = eps.ravel()
        if not isinstance(eps, np.ndarray) or eps.shape != (6,):
            raise ValueError("Input `eps` must be a NumPy array with shape (6,)")
        
        # Early return if strain is zero
        if np.allclose(eps, 0.0, atol=1e-12):
            return d0
        
        # Unpack previous damage state
        d0_tc, d0_t, d0_c = d0 
        
        # Retrieve material damage parameters
        Ac = self.para.Ac
        Bc = self.para.Bc
        At = self.para.At
        Bt = self.para.Bt
        control = self.para.control.lower().strip()
        
        # Compute equivalent strain
        eps_eq = calculate_equivalent_strain_from_voigt(vec=eps)
        
        # Compute tensile damage
        dt = self.calculate_d_from_equivalent_strain(eps_eq=eps_eq, eps_d0=self.mat.prop.eps_tp, A=At, B=Bt)
        is_valid = self.validate_damage_value(value=dt, label='dt')
        if not is_valid:
            self.logger.debug(
                "[DEBUG] Invalid `dc` detected in `calculate_damage()` (Mazars): "
                "dt = %.6f | eps = %s | eps_eq = %.6f | eps_d0 = %.6f | At = %.3f | Bt = %.3f",
                dt, np.round(eps, 6).tolist(), eps_eq, self.mat.prop.eps_tp, At, Bt
            )
        
        if control in {"t", "tension"}:
            dt = np.clip(dt, d0_t, 0.99) if clip else dt
            return np.array([d0_tc, dt, d0_c])
        
        # Compute compressive damage
        dc = self.calculate_d_from_equivalent_strain(eps_eq=eps_eq, eps_d0=self.mat.prop.eps_tp, A=Ac, B=Bc)
        is_valid = self.validate_damage_value(value=dc, label='dc')
        if not is_valid:
            self.logger.debug(
                "[DEBUG] Invalid `dc` detected in `calculate_damage()` (Mazars): "
                "dt = %.6f | eps = %s | eps_eq = %.6f | eps_d0 = %.6f | Ac = %.3f | Bc = %.3f",
                dc, np.round(eps, 6).tolist(), eps_eq, self.mat.prop.eps_cp, Ac, Bc
            )
        
        if control in {"c", "compression"}:
            dc = np.clip(dc, d0_c, 0.99) if clip else dc
            return np.array([d0_tc, d0_t, dc])
        
        if control in {"ct", "tc", "compression-tension", "tension-compression"}:
            # Calculate weighting factors
            alpha_t, alpha_c = self.calculate_alpha(eps=eps, d0=d0[0])
            
            dt = float(np.clip(dt, d0_t, 0.99) if clip else dt)
            dc = float(np.clip(dc, d0_c, 0.99) if clip else dc)
            
            # Combine weighted damage contributions
            d = alpha_t * dt + alpha_c * dc
            
            is_valid = self.validate_damage_value(value=d, label="d")
            
            if not is_valid:
                print(f'\tdc = {dc:.2f}, dt = {dt:.0f}, alpha_c = {alpha_c:.2f}, alpha_t = {alpha_t:.2f}')
                self.logger.debug(
                    "[DEBUG] Invalid `d` detected in `calculate_damage()` (Mazars): "
                    "d = %.6f | dc = %.6f | dt = %.6f | alpha_c = %.3f | alpha_t = %.3f | "
                    "d0 = [%.6f, %.6f, %.6f] | eps = %s | eps_eq = %.6f | eps_d0 = %.6f | "
                    "At = %.3f | Bt = %.3f | Ac = %.3f | Bc = %.3f",
                    d, dc, dt, alpha_c, alpha_t,
                    d0_tc, d0_t, d0_c,
                    np.round(eps, 6).tolist(), eps_eq, self.mat.prop.eps_tp,
                    At, Bt, Ac, Bc
                )
            
            d = np.clip(d, d0_tc, 0.99) if clip else d
            
            return np.array([d, dt, dc])
        
        raise ValueError(
            f"Invalid control mode '{control}'. "
            "Expected one of: 't', 'c', 'ct', 'tc', 'tension', 'compression', 'compression-tension'."
        )
    
    def calculate_damaged_uniaxial_compressive_stress(self, eps: np.ndarray, Ac: float, Bc: float) -> np.ndarray:
        """Calculates uniaxial compressive stress from strain using the Mazars model.

        This method evaluates the stress response for an array of compressive strains
        based on the isotropic Mazars damage formulation.

        Args:
            eps (np.ndarray): Array of compressive strain values (should be ≤ 0).
            Ac (float): Damage shape parameter in compression.
            Bc (float): Damage curvature parameter in compression.

        Returns:
            np.ndarray: Computed compressive stress values (positive).
        """
        # Convert to positive strain magnitude
        eps_abs = np.abs(eps)

        # Evaluate damage for each ε
        d_vals = self.calculate_uniaxial_compression_damage(eps=eps_abs, Ac=Ac, Bc=Bc)

        # Compute compressive stress: σ = (1 - d) * E * ε
        sig = (1.0 - d_vals) * self.mat.prop.E * eps_abs

        return sig
    
    def calculate_damaged_uniaxial_tensile_stress(self, eps: np.ndarray, At: float, Bt: float) -> np.ndarray:
        """Calculates uniaxial tensile stress from strain using the Mazars model.

        This method evaluates the stress response for an array of tensile strains
        using the isotropic Mazars damage model formulation in tension.

        Args:
            eps (np.ndarray): Array of tensile strain values (should be ≥ 0).
            At (float): Damage shape parameter in tension.
            Bt (float): Damage curvature parameter in tension.

        Returns:
            np.ndarray: Computed tensile stress values.
        """
        # Use absolute strain values to ensure positivity
        eps_abs = np.abs(eps)

        # Evaluate damage values for each tensile strain
        d_vals = self.calculate_uniaxial_tension_damage(eps=eps_abs, At=At, Bt=Bt)

        # Compute stress using: σ = (1 - d) * E * ε
        sig = (1.0 - d_vals) * self.mat.prop.E * eps_abs

        return sig
    
    def calculate_damage_from_elastic_eps(self, clip: bool) -> np.ndarray:
        """
        Computes damage values for a series of elastic strain vectors loaded from file.

        This method reads a CSV file containing engineering strain vectors (in Voigt notation),
        applies the damage model to each vector, and returns the resulting damage array.

        Args:
            clip (bool): Whether to clip the damage value within valid range [d0, 0.99]. Default is True.

        Returns:
            np.ndarray: Array of damage values, shape (n_samples, 3)
        """
        filepath = 'log/kfu/elastic_eps_voigt.csv'
        try:
            eps_array = np.loadtxt(filepath, delimiter=',')
        except Exception as e:
            raise IOError(f"Failed to load strain data from '{filepath}': {e}")

        if eps_array.ndim == 1:
            eps_array = eps_array[np.newaxis, :]  # Ensure shape (N, 6)

        d0 = np.zeros(3)
        d_results = []

        for eps_voigt in eps_array:
            d_vec = self.calculate_damage(eps=eps_voigt, d0=d0, clip=clip)
            d_results.append(d_vec)

        return np.array(d_results)
    
    def identify_A_B_from_stress(self, mode: str, initial_AB: Tuple[float, float], AB_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
            eps_max: float, d_eps: float = 1e-6) -> Tuple[float, float]:
        """Identifies Mazars damage parameters A and B via curve fitting.

        This method performs nonlinear least squares fitting of the Mazars model
        to synthetic stress-strain data in either uniaxial tension or compression.

        Args:
            mode (str): Type of loading. Must be either 'compression' or 'tension'.
            initial_AB (Tuple[float, float]): Initial guess for (A, B) parameters.
            AB_bounds (Tuple[Tuple[float, float], Tuple[float, float]]): Lower and upper bounds for (A, B).
                Format: ((A_min, B_min), (A_max, B_max))
            eps_max (float): Maximum strain for generating synthetic data.
            d_eps (float, optional): Strain step size. Defaults to 1e-6.

        Returns:
            Tuple[float, float]: Fitted damage parameters (A, B).

        Raises:
            ValueError: If `mode` is not 'compression' or 'tension'.
        """
        # Generate strain range
        eps = np.arange(0.0, np.abs(eps_max), d_eps)
        
        # Generate stress from material model
        if mode == 'compression':
            sig = -self.mat.calculate_stress(eps=-eps)  # Ensure stress is positive
            fit_func = self.calculate_damaged_uniaxial_compressive_stress
        elif mode == 'tension':
            sig = self.mat.calculate_stress(eps=eps)
            fit_func = self.calculate_damaged_uniaxial_tensile_stress
        else:
            raise ValueError("Mode must be 'compression' or 'tension'")

        # Fit (A, B) to minimize error between synthetic stress and model prediction
        popt, pcov = curve_fit(
            f=fit_func,
            xdata=eps,
            ydata=sig,
            p0=initial_AB,
            bounds=AB_bounds
        )

        A_fit, B_fit = popt
        print(f'[Fit] Mode = {mode} | A = {A_fit:.4f}, B = {B_fit:.1f}')
        print(f'[Fit] Covariance matrix:\n{pcov}')
        self.logger.info(f'[Fit] Mode = {mode} | A = {A_fit:.4f}, B = {B_fit:.1f}')
        self.logger.info(f'[Fit] Covariance matrix:\n{pcov}')

        # Optional: Plot results
        sig_init = fit_func(eps, *initial_AB)
        sig_fit = fit_func(eps, A_fit, B_fit)

        fig, ax = plot_template(
            [eps, sig], [eps, sig_init], [eps, sig_fit],
            figsize=(9/2.54, 5.5/2.54),
            legend_labels=[
                "Target (from model)",
                f"Initial Guess (A={initial_AB[0]:.2f}, B={initial_AB[1]:.0f})",
                f"Fitted (A={A_fit:.2f}, B={B_fit:.0f})"
            ],
            colors=['black', (0, 47 / 255, 167 / 255), (192 / 255, 0, 0)],
            linestyles=['-', '--', '--'],
            xlabel=f"{mode.title()} Strain",
            ylabel="Stress (MPa)",
        )

        # Save and close
        fig.savefig(f"log/damage/MO_parameters_fitted_from_stress_{mode}.png", dpi=300)
        plt.show()
        plt.close(fig)

        return A_fit, B_fit
    
    def plot_uniaxial_tension_damage_evolution(self, eps_t_max: float, d_eps: float, At: float, Bt: float,
                eps_t_min: float = 0.0,
                save_path: str ='log/damage/MO_uniaxial_tension_damage_evolution.png',
                show: bool = True, close: bool = True) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
        """Plots uniaxial tensile stress-strain curve with damage evolution.

        This method overlays the tensile damage variable curve (d vs. ε) on top of 
        the uniaxial tensile stress-strain response, using a secondary y-axis. 
        The damage curve is rendered as a red dashed line for clear visual distinction.

        Args:
            eps_t_max (float): Maximum tensile strain.
            d_eps (float): Strain step size.
            At (float): Damage shape parameter in tension.
            Bt (float): Damage curvature parameter in tension.
            eps_t_min (float, optional): Minimum tensile strain. Defaults to 0.0.
            save_path (str, optional): File path to save the generated figure. Defaults to 
                'log/damage/MO_uniaxial_tension_damage_evolution.png'.
            show (bool, optional): If True, displays the figure interactively. Defaults to True.
            close (bool, optional): If True, closes the figure after saving to avoid memory leaks. Defaults to True.

        Returns:
            Tuple[plt.Figure, plt.Axes, plt.Axes]: The generated Matplotlib figure, the primary (stress) axis and (damage) axis.
        """
        # Generate tensile strain values (positive for calculation)
        eps = np.arange(np.abs(eps_t_min), np.abs(eps_t_max), d_eps)
        
        # Evaluate damage values for each tensile strain
        d_vals = self.calculate_uniaxial_tension_damage(eps=eps, At=At, Bt=Bt)
        
        # Generate base stress-strain plot
        fig, ax1 = self.mat.plot_stress_strain_tension(eps_t_max=eps_t_max, d_eps=d_eps, save_path='', show=False, close=False)
        
        # Overlay damage curve on secondary y-axis
        ax2 = self.plot_right_damage_evolution(ax=ax1, eps=eps, d_vals=d_vals)

        # Adjust layout for clarity
        fig.tight_layout()
        
        # Save the figure
        fig.savefig(save_path, dpi=300)
        
        # Show the plot if requested
        if show:
            plt.show()

        # Close the figure if requested
        if close:
            plt.close(fig)
    
        return fig, ax1, ax2
    
    def plot_uniaxial_compression_damage_evolution(self, eps_c_max: float, d_eps: float, Ac: float, Bc: float,
                eps_c_min: float = 0.0,
                save_path: str ='log/damage/MO_uniaxial_compression_damage_evolution.png',
                show: bool = True, close: bool = True) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
        """Plots uniaxial compression stress-strain curve with damage evolution.

        This function overlays the damage evolution curve (d vs. ε) on top of the
        stress-strain curve using a secondary y-axis. The damage curve is plotted
        as a red dashed line for visual clarity.

        Args:
            eps_c_max (float): Maximum compressive strain (should be negative).
            d_eps (float): Strain step size.
            Ac (float): Damage shape parameter in compression.
            Bc (float): Damage curvature parameter in compression.
            eps_c_min (float, optional): Minimum compressive strain. Defaults to 0.0.
            show: If True, displays the plot in an interactive window.
            close: If True, closes the figure after saving (prevents memory accumulation).

        Returns:
            Tuple[plt.Figure, plt.Axes, plt.Axes]: The generated Matplotlib figure, the primary (stress) axis and (damage) axis.
        """
        # Generate compressive strain values (positive for calculation)
        eps = np.arange(np.abs(eps_c_min), np.abs(eps_c_max), d_eps)
        
        # Compute damage variable d for each compressive strain
        d_vals = self.calculate_uniaxial_compression_damage(eps=eps, Ac=Ac, Bc=Bc)
        
        # Generate base stress-strain plot
        fig, ax1 = self.mat.plot_stress_strain_compression(eps_c_max=eps_c_max, d_eps=d_eps, save_path='', show=False, close=False)
        
        # Overlay damage curve on secondary y-axis
        ax2 = self.plot_right_damage_evolution(ax=ax1, eps=eps, d_vals=d_vals)

        # Adjust layout for clarity
        fig.tight_layout()
        
        # Save the figure
        fig.savefig(save_path, dpi=300)
        
        # Show the plot if requested
        if show:
            plt.show()

        # Close the figure if requested
        if close:
            plt.close(fig)
    
        return fig, ax1, ax2
    
    def plot_damaged_uniaxial_tensile_stress(self, eps_t_max: float, d_eps: float, At: float, Bt: float,
                show: bool = True, close: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """Plots the damaged uniaxial tensile stress-strain curve using the specified damage parameters.

        Args:
            eps_t_max (float): Maximum tensile strain (positive) to evaluate.
            d_eps (float): Strain increment for evaluation.
            At (float): Tensile damage shape parameter.
            Bt (float): Tensile damage evolution rate parameter.
            show: Whether to display the figure interactively.
            close: Whether to close the figure after saving (to free memory).

        Returns:
            Tuple[plt.Figure, plt.Axes]: The matplotlib figure and axes of the generated plot.
        """
        # Define tensile strain range (positive values only)
        eps = np.arange(0.0, np.abs(eps_t_max), d_eps)
        
        # Compute corresponding stress values
        sig = self.calculate_damaged_uniaxial_tensile_stress(eps=eps, At=At, Bt=Bt)
        
        fig, ax = self.mat.plot_stress_strain_tension(eps_t_max=eps_t_max, d_eps=d_eps, save_path='', show=False, close=False)
        ax.plot(eps, sig)
        # Plot using the configured template
        # fig, ax = plot_template([eps, sig],
        #             xlabel='Tensile Strain',
        #             ylabel='Stress (MPa)',
        #             )
        
        # Show the plot if requested
        if show:
            plt.show()

        # Close the figure if requested
        if close:
            plt.close(fig)
        
        return fig, ax
    
    def plot_damaged_uniaxial_compressive_stress(self, eps_c_max: float, eps_d0:float, d_eps: float, Ac: float, Bc: float,
                show: bool = True, close: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """Plots the damaged uniaxial compressive stress-strain curve for a given damage model.

        Args:
            eps_c_max (float): Maximum compressive strain (positive value).
            d_eps (float): Strain increment for evaluation.
            Ac (float): Compressive damage shape parameter.
            Bc (float): Compressive damage evolution rate parameter.
            show: Whether to display the figure interactively.
            close: Whether to close the figure after saving (to free memory).

        Returns:
            Tuple[plt.Figure, plt.Axes]: Matplotlib figure and axes objects of the plot.
        """
        # Define compressive strain range (positive for evaluation)
        eps = np.arange(0.0, np.abs(eps_c_max), d_eps)
        
        # Compute corresponding stress values
        sig = self.calculate_damaged_uniaxial_compressive_stress(eps=eps, Ac=Ac, Bc=Bc)
        
        fig, ax = self.mat.plot_stress_strain_compression(eps_c_max=eps_c_max, d_eps=d_eps, save_path='', show=False, close=False)
        ax.plot(eps, sig)
        
        # Create stress-strain plot
        # fig, ax = plot_template([eps, sig],
        #             xlabel='Compressive Strain',
        #             ylabel='Stress (MPa)',
        #             )
        
        # Show the plot if requested
        if show:
            plt.show()

        # Close the figure if requested
        if close:
            plt.close(fig)
        
        return fig, ax

class Mazars_Original_Damage_Torch(Damage):
    """Mazars isotropic damage model for concrete, implemented in PyTorch.

    Implements the original Mazars scalar damage formulation to simulate
    stiffness degradation in concrete under tension and compression.
    Damage evolves separately in tension and compression via exponential‐type
    laws, with parameters controlling the softening rate.

    Attributes:
        mat (Material):
            Base material instance containing elastic and strength properties.
        para (Damage_Parameter):
            Damage parameters including:
                - Ac (float): compression damage coefficient
                - Bc (float): compression damage exponent
                - At (float): tension damage coefficient
                - Bt (float): tension damage exponent
                - ib (float): optional shape parameter for Bt if Bt not provided
                - control (str): one of 't', 'c', 'ct' etc.
        logger (logging.Logger):
            Logger for tracking model initialization and events.
    """
    def __init__(self, mat: Material, para: Damage_Parameter) -> None:
        """Initializes the Mazars damage model using material properties and 
        damage evolution parameters.

        Args:
            mat (Material): Instance of the base material class containing 
                elastic and strength parameters.
            para (Damage_Parameter): Instance of damage coefficients including:
                - Ac (float): Compression damage coefficient.
                - Bc (float): Compression damage exponent.
                - At (float): Tension damage coefficient.
                - Bt (float): Tension damage exponent.
                - ib (float): Optional shape parameter for Bt calculation if Bt is not given.
                - control (str): Control strategy, one of:
                    - 't' / 'tension'
                    - 'c' / 'compression'
                    - 'ct', 'tc', or 'compression-tension' (combined mode).
        """
        super().__init__(mat, para)
        self.mat = mat
        self.para = para
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info(
            "Initialized Mazars Original Damage PyTorch with parameters: %s",
            vars(self.para)
        )
    
    def macaulay_bracket(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the Macaulay bracket ⟨x⟩ = max(x, 0)."""
        return torch.clamp(x, min=0.0)
    
    def calculate_principal_values(self, tensor: torch.Tensor) -> torch.Tensor:
        """Computes principal values (eigenvalues) from symmetric tensors.
        
        Args:
            tensor: Shape (N, 3, 3) symmetric tensors
            
        Returns:
            torch.Tensor: Shape (N, 3) principal values sorted in descending order
        """
        # Use torch.linalg.eigvalsh for symmetric tensors (guaranteed real eigenvalues)
        eigenvalues = torch.linalg.eigvalsh(tensor)  # (N, 3)
        
        # Sort in descending order
        eigenvalues_sorted, _ = torch.sort(eigenvalues, dim=-1, descending=True)
        
        return eigenvalues_sorted
    
    def calculate_equivalent_strain_from_tensor(self, eps_tensor: torch.Tensor) -> torch.Tensor:
        """Computes Mazars-type equivalent strain from strain tensors.
        
        Args:
            eps_tensor: Shape (N, 3, 3) strain tensors
            
        Returns:
            torch.Tensor: Shape (N,) equivalent strains
        """
        # Get principal strains
        eps_principal = self.calculate_principal_values(tensor=eps_tensor)  # (N, 3)
        
        # Apply Macaulay bracket to get tensile components
        eps_pos = self.macaulay_bracket(x=eps_principal)  # (N, 3)
        
        # Compute equivalent strain: sqrt(sum of squares)
        eq_strain = torch.sqrt(torch.sum(eps_pos**2, dim=-1))  # (N,)
        
        return eq_strain
    
    def calculate_d_from_equivalent_strain(self, eps_eq: torch.Tensor, eps_d0: float, A: float, B: float) -> torch.Tensor:
        """Calculates scalar damage from equivalent strain using Mazars’ exponential law,
        which enforces zero damage whenever the strain is below the threshold.

        Args:
            eps_eq:  FloatTensor of shape (N,), the equivalent strains at N points.
            eps_d0:  Damage threshold strain (below this, d = 0).
            A:       Damage coefficient (must be > 0).
            B:       Damage exponent   (must be > 0).

        Returns:
            FloatTensor of shape (N,) with damage values in [0, 1).

        Raises:
            ValueError: If A <= 0 or B <= 0, or if any eps_eq is negative.
        """
        if A <= 0 or B <= 0:
            raise ValueError("Parameters A and B must be positive.")
        if (eps_eq < 0).any():
            raise ValueError("Equivalent strains must be non‐negative.")
        
        # for strains below the threshold, damage remains zero
        # otherwise compute d = 1 − (eps_d0/eps_eq) * exp(−B*(eps_eq − eps_d0))
        damage = torch.where(
            eps_eq <= eps_d0,
            torch.zeros_like(eps_eq),
            1.0 - (1.0 - A) * eps_d0 / eps_eq - A * torch.exp(-B * (eps_eq - eps_d0))
        )
        
        return damage
    
    def validate_damage_value(self, value: torch.Tensor, label: str = "d") -> torch.Tensor:
        """Validates damage values and returns a boolean mask for valid values.
        
        Args:
            value: Shape (N,) damage values
            label: Name for warning messages
            
        Returns:
            torch.Tensor: Shape (N,) boolean mask (True for valid values)
        """
        valid_mask = (value >= 0.0) & (value <= 1.0)
        
        # Optional: print warnings for invalid values
        if not torch.all(valid_mask):
            invalid_count = torch.sum(~valid_mask).item()
            self.logger.warning(f"Warning: {invalid_count} invalid damage values `{label}` detected")
            
        return valid_mask
    
    def calculate_stress_tensor(self, eps: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Compute the batch of Cauchy stress tensors via Hooke’s law with damage.

        For each of N points, we have a small-strain tensor ε (3×3) and a scalar
        damage d. We first reduce the Young’s modulus E by (1–d), then form the
        Lamé constants μ and λ and assemble

            σ = 2 μ ε + λ (tr ε) I.

        Args:
            eps: FloatTensor of shape (N, 3, 3).  
                The symmetric small-strain tensor at N sampling points.
            d:   FloatTensor of shape (N,).  
                Scalar damage variable in [0,1), one per point.

        Returns:
            FloatTensor of shape (N, 3, 3), the Cauchy stress tensor σ at each point.
        """
        # 1) material + damage
        E  = self.mat.prop.E_pinn   # nondimensional PINN modulus
        nu = self.mat.prop.nu
        E_d = (1.0 - d) * E         # effective Young’s modulus per point
        
        # 2) Lamé parameters per point
        factor = E_d / ((1 + nu) * (1 - 2 * nu))  # shape (N,)
        lam    = nu * factor                     # shape (N,)
        mu     = 0.5 * (1.0 - 2.0 * nu) * factor  # shape (N,)

        # 3) trace of strain: tr(ε)
        tr_eps = torch.einsum('nii->n', eps)     # shape (N,)

        # 4) assemble σ = 2μ ε + λ (tr ε) I
        I = torch.eye(3, device=eps.device, dtype=eps.dtype)  # (3,3)
        sigma = 2.0 * mu[:, None, None] * eps + lam[:, None, None] * tr_eps[:, None, None] * I  # shape (N,3,3)

        return sigma
    
    def calculate_alpha(self, eps_tensor: torch.Tensor, d0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes tensile and compressive damage weights (α_t, α_c).
        
        Args:
            eps_tensor: Shape (N, 3, 3) strain tensors
            d0:   FloatTensor of shape (N,), previous damage state per point.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (alpha_t, alpha_c) each shape (N,)
        """
        E = self.mat.prop.E_pinn
        nu = self.mat.prop.nu
        
        # Get principal values
        eps_p = self.calculate_principal_values(tensor=eps_tensor)  # (N, 3)
        sig_tensor = self.calculate_stress_tensor(eps=eps_tensor, d=d0)
        sig_p = self.calculate_principal_values(tensor=sig_tensor)  # (N, 3)
        
        # Tension-only principal stress
        sig_p_pos = self.macaulay_bracket(sig_p)  # (N, 3)
        sigma_pos_sum = torch.sum(sig_p_pos, dim=-1, keepdim=True)  # (N, 1)
        
        # Reconstruct tensile strain from stress (Ramirez 2017, Eq. 5)
        eps_t = ((1.0 + nu) * sig_p_pos - nu * sigma_pos_sum) / E  # (N, 3)
        
        # Compute compressive strain as residual
        eps_c = eps_p - eps_t  # (N, 3)
        
        # Equivalent strain: norm of positive principal strains
        eps_pos = self.macaulay_bracket(eps_p)  # (N, 3)
        eps_eq = torch.sqrt(torch.sum(eps_pos**2, dim=-1))  # (N,)
        
        # Avoid division by zero
        eps_eq_safe = torch.clamp(eps_eq, min=1e-12)
        
        # Compute alpha weights
        alpha_t = torch.sum(eps_t * eps_pos, dim=-1) / (eps_eq_safe * eps_eq_safe)  # (N,)
        alpha_c = torch.sum(eps_c * eps_pos, dim=-1) / (eps_eq_safe * eps_eq_safe)  # (N,)
        
        # Handle zero equivalent strain case
        zero_mask = eps_eq < 1e-12
        alpha_t[zero_mask] = 0.0
        alpha_c[zero_mask] = 0.0
        
        # Warning for inconsistent weights (optional)
        inconsistent_mask = (alpha_t + alpha_c) > 1.0 + 1e-5
        if torch.any(inconsistent_mask):
            inconsistent_count = torch.sum(inconsistent_mask).item()
            self.logger.warning(f"Warning: {inconsistent_count} cases where α_t + α_c > 1.0")
        
        return alpha_t, alpha_c
    
    
    def calculate_damage(self, eps: torch.Tensor, d0: torch.Tensor, clip: bool = True) -> torch.Tensor:
        """Compute updated scalar damage for a batch of points.

        This uses Mazars’ isotropic damage law.  We take a batch of N strains
        in Voigt notation and return N×3 damage states:

            - index 0: combined tension/compression damage d_tc
            - index 1: tensile‐only damage d_t
            - index 2: compressive‐only damage d_c

        Args:
            eps:  FloatTensor of shape (N,6), current strain components
                    [ε_xx, ε_yy, ε_zz, ε_xy, ε_yz, ε_xz] at N points.
            d0:   FloatTensor of shape (N,3), previous damage state per point.
            clip: If True, enforce d_new ≤ 0.99 and d_new ≥ d0.

        Returns:
            FloatTensor of shape (N,3), updated damage states [d_tc, d_t, d_c].
        """
        if eps.dim() != 3 or eps.shape[1:] != (3, 3):
            raise ValueError("Input `eps` must have shape (N, 3, 3)")
        
        if d0.dim() != 2 or d0.shape[1] != 3:
            raise ValueError("Input `d0` must have shape (N, 3)")
        
        # Identify zero strain cases
        N = eps.shape[0]
        eps_norm = torch.norm(eps.view(N, -1), dim=1)
        zero_mask = eps_norm < 1e-12 # True where strain ≃ 0
        # Early return if all strains are zero
        if zero_mask.all():
            return d0
        
        # pick out only the non‐zero strains & previous damage
        nonzero_mask = ~zero_mask    # True where strain ≠ 0
        eps_nz = eps[nonzero_mask]   # (M,3,3) or however you’re representing it
        d0_nz  = d0[nonzero_mask]    # (M,3)
        
        # Unpack previous damage state
        d0_tc, d0_t, d0_c = d0[:, 0], d0[:, 1], d0[:, 2]
        d0_tc_nz, d0_t_nz, d0_c_nz = d0_nz[:, 0], d0_nz[:, 1], d0_nz[:, 2]
        
        # Retrieve material damage parameters
        Ac = self.para.Ac
        Bc = self.para.Bc
        At = self.para.At
        Bt = self.para.Bt
        control = self.para.control.lower().strip()
        eps_tp  = self.mat.prop.eps_tp
        
        # Compute equivalent strain
        eps_eq_nz = self.calculate_equivalent_strain_from_tensor(eps_tensor=eps_nz)  # (M,)
        
        # Compute tensile damage
        dt_nz = self.calculate_d_from_equivalent_strain(eps_eq=eps_eq_nz, eps_d0=eps_tp, A=At, B=Bt)  # (M,)
        
        # Validate tensile damage
        self.validate_damage_value(dt_nz, 'dt')
        
        mask_t = eps_eq_nz < eps_tp
        dt_nz = torch.where(mask_t, d0_t_nz, dt_nz)
        
        # Handle different control modes
        if control in {"t", "tension"}:
            if clip:
                dt_nz = clamp(d=dt_nz, min=d0_t_nz, max=0.99)
            d_nz = torch.stack([d0_tc_nz, dt_nz, d0_c_nz], dim=1)
            d0[nonzero_mask] = d_nz
            return d0
        
        # Compute compressive damage
        dc_nz = self.calculate_d_from_equivalent_strain(eps_eq=eps_eq_nz, eps_d0=eps_tp, A=Ac, B=Bc)  # (M,)
        
        # Validate compressive damage
        self.validate_damage_value(dc_nz, 'dc')
        
        mask_c = eps_eq_nz < eps_tp
        dc_nz = torch.where(mask_c, d0_c_nz, dc_nz)
        
        if control in {"c", "compression"}:
            if clip:
                dc_nz = clamp(d=dc_nz, min=d0_c_nz, max=0.99)
            d_nz = torch.stack([d0_tc_nz, d0_t_nz, dc_nz], dim=1)
            d0[nonzero_mask] = d_nz
            return d0
        
        if control in {"ct", "tc", "compression-tension", "tension-compression"}:
            
            # Calculate weighting factors
            alpha_t, alpha_c = self.calculate_alpha(eps_tensor=eps_nz, d0=d0_nz[:,0])  # (M,), (M,)
            
            # Clip individual damage components
            if clip:
                dt_nz = clamp(d=dt_nz, min=d0_t_nz, max=0.99)
                dc_nz = clamp(d=dc_nz, min=d0_c_nz, max=0.99)
            
            # Combine weighted damage contributions
            d_tc_nz = alpha_t * dt_nz + alpha_c * dc_nz  # (M,)
            
            # Validate combined damage
            self.validate_damage_value(d_tc_nz, 'd')
            
            # Clip combined damage
            if clip:
                d_tc_nz = clamp(d=d_tc_nz, min=d0_tc_nz, max=0.99)
            
            d_nz = torch.stack([d_tc_nz, dt_nz, dc_nz], dim=1)
            d0[nonzero_mask] = d_nz
            return d0
        
        raise ValueError(
            f"Invalid control mode '{control}'. "
            "Expected one of: 't', 'c', 'ct', 'tc', 'tension', 'compression', 'compression-tension'."
        )


class Mu_Damage(Damage):
    """μ-Damage model for concrete (Mazars, 2015).

    This model extends classical isotropic damage by incorporating both tensile and compressive
    damage effects based on triaxial stress states and independent evolution laws for tension and compression.

    Attributes:
        mat (Material): Material object containing mechanical and constitutive properties.
        para (Damage_Parameter): Object containing shape and curvature parameters:
            At, Bt (for tension); Ac, Bc (for compression).
    """
    
    def __init__(self, mat: Material, para: Damage_Parameter) -> None:
        """Initializes the μ-damage model with specified material and damage parameters.

        Args:
            mat (Material): Material object with elastic constants and stress-strain behavior.
            para (Damage_Parameter): Damage parameters with keys:
                - At (float): Shape parameter in tension.
                - Bt (float): Curvature parameter in tension.
                - Ac (float): Shape parameter in compression.
                - Bc (float): Curvature parameter in compression.
                - k (float): Shape adjustment parameter for large strain
        """
        super().__init__(mat, para)
        self.mat = mat
        self.para = para
        
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info("Initialized Mu-Damage model with parameters: %s", vars(self.para))
    
    def calculate_r(self, d0: float, D: np.ndarray, eps: np.ndarray) -> float:
        """Computes the stress triaxiality ratio `r` for the μ-model.

        The ratio `r` reflects the proportion of tensile principal stresses to the total 
        magnitude of all principal stresses. It is used in the μ-model to weight damage 
        evolution between tension and compression.

        Args:
            d0 (float): Previous scalar damage value.
            D (np.ndarray): Constitutive stiffness matrix (6×6).
            eps (np.ndarray): Engineering strain vector (length 6).

        Returns:
            float: Stress triaxiality ratio `r` ∈ [0, 1].

        """
        # Compute stress vector (6,) using damaged stiffness
        sig = (1 - d0) * D @ eps # (6,)
        
        # Extract principal stresses from Voigt-form stress
        sig_p = calculate_principal_values(vec=sig, quantity='stress')
        
        # Macaulay bracket: keep only positive principal values (tension)
        sig_p_pos = macaulay_bracket(sig_p)
        
        # Compute triaxiality ratio: sum(tensile) / sum(abs)
        numer = np.sum(sig_p_pos)         # tensile part only
        denom = np.sum(np.abs(sig_p))     # total principal stress magnitude

        # Avoid division by zero or instability
        if denom < 1e-12:
            return 0.0

        r = numer / denom
        # is_valid = self.validate_damage_value(value=r, label='r')
        if r < 0.0 or r > 1.0:
            self.logger.debug(
                "[DEBUG] Invalid `r` detected in `calculate_r()` (Mu): "
                "r = %.6f | eps = %s | sig = %s | numer = %.6f | denom = %.3f | d0 = %.3f",
                r, np.round(eps, 6).tolist(), np.round(eps, 3).tolist(), numer, denom, d0
            )
        
        return float(np.clip(r, 0.0, 1.0))  # ensure it stays in [0, 1]
    
    def calculate_AB(self, r: float, k: float,
            Ac: Optional[float] = None,
            Bc: Optional[float] = None,
            At: Optional[float] = None,
            Bt: Optional[float] = None) -> Tuple[float, float]:
        """Computes interpolated A and B parameters for the μ-model based on 
        stress triaxiality `r` and asymptotic shape control parameter `k`.

        The interpolation blends the tensile (At, Bt) and compressive (Ac, Bc) 
        damage parameters to obtain equivalent A and B values for mixed stress states.

        Args:
            r (float): Stress triaxiality factor ∈ [0, 1], indicating the ratio of 
                    tensile stress to total stress (0 = pure compression, 1 = pure tension).
            k (float): Asymptotic shape adjustment factor that affects the curvature 
                    of the transition between tension and compression (typically ~0.75).
            Ac (float, optional): Compressive damage shape parameter. Defaults to `self.para.Ac`.
            Bc (float, optional): Compressive damage curvature parameter. Defaults to `self.para.Bc`.
            At (float, optional): Tensile damage shape parameter. Defaults to `self.para.At`.
            Bt (float, optional): Tensile damage curvature parameter. Defaults to `self.para.Bt`.

        Returns:
            Tuple[float, float]: Interpolated values (A, B) based on r and k.

        Raises:
            ValueError: If `r` is not in the range [0, 1].
        """
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"Triaxiality factor `r` = {r} must be in the range [0, 1].")
        
        # Use input parameters if provided, otherwise fall back to defaults from self.para
        Ac = Ac if Ac is not None else self.para.Ac
        Bc = Bc if Bc is not None else self.para.Bc
        At = At if At is not None else self.para.At
        Bt = Bt if Bt is not None else self.para.Bt
        
        # Interpolated shape parameter A using μ-model formulation
        A = At * (2 * r**2 * (1 - 2 * k) - r * (1 - 4 * k)) + Ac * (2 * r**2 - 3 * r + 1)
        
        # Smooth transition between Bt and Bc
        weight = r**(r**2 - 2 * r + 2)
        B = weight * Bt + (1 - weight) * Bc
        
        return float(A), float(B)
    
    def calculate_Yt_Yc(self, eps_p: np.ndarray, eps_0t: float, eps_0c: float) -> float:
        """Computes the equivalent strain measures Yₜ (tensile) and Yc (compressive)
        from the principal strains using energy-based definitions.

        These measures are used to compare the current strain state against 
        tension and compression thresholds in the μ-model.

        Args:
            eps_p (np.ndarray): Principal strains (shape: (3,)).
            eps_0t (float): Threshold strain for tension evolution.
            eps_0c (float): Threshold strain for compression evolution.

        Returns:
            Tuple[float, float]: 
                - Yt (float): Equivalent tensile strain (≥ eps_0t).
                - Yc (float): Equivalent compressive strain (≥ eps_0c).

        Raises:
            ValueError: If `eps_p` is not a 1D NumPy array of length 3.
        """
        if not isinstance(eps_p, np.ndarray) or eps_p.shape != (3,):
            raise ValueError("`eps_p` must be a NumPy array of shape (3,) representing principal strains.")

        nu = self.mat.prop.nu # Poisson's ratio
        
        # Unpack principal strains
        eps1, eps2, eps3 = eps_p
        
        # Second strain invariant (deviatoric component)
        J = 0.5 * ((eps1 - eps2)**2 + (eps2 - eps3)**2 + (eps3 - eps1)**2)
        
        # First strain invariant (volumetric component)
        I = eps1 + eps2 + eps3
        
        # Equivalent strains for tension and compression (Mazars μ-formulation)
        eps_t = I / (2 * (1 - 2 * nu)) + np.sqrt(J) / (2 * (1 + nu))
        eps_c = I / (5 * (1 - 2 * nu)) + 6 * np.sqrt(J) / (5 * (1 + nu))
        
        # Ensure damage-driving strain is always ≥ threshold
        Yt = max(eps_t, eps_0t)
        Yc = max(eps_c, eps_0c)
        
        return float(Yt), float(Yc)
    
    def calculate_damage(self, eps: np.ndarray, d0: np.ndarray,
            Ac: Optional[float] = None, Bc: Optional[float] = None,
            At: Optional[float] = None, Bt: Optional[float] = None, clip: bool = True) -> np.ndarray:
        """Computes the scalar damage variable `d` using the μ-model of isotropic damage.

        This method evaluates the evolution of material damage based on the current
        strain state `eps`, considering the combined effect of tension and compression.
        The μ-model achieves this by:
            - Computing a stress triaxiality factor `r` from the current strain and stiffness.
            - Interpolating the tensile (At, Bt) and compressive (Ac, Bc) damage parameters.
            - Constructing an equivalent strain measure `Y` from principal strains.
            - Evaluating the damage evolution law based on `Y` and its threshold `Y0`.

        This implementation supports optional override of damage parameters.

        Args:
            eps (np.ndarray): Engineering strain vector in Voigt notation (shape: (6,)).
            d0 (np.ndarray): Previous damage state (3,), [d_tc, d_t, d_c].
            Ac (float, optional): Compressive damage shape parameter. Defaults to `self.para.Ac`.
            Bc (float, optional): Compressive damage curvature parameter. Defaults to `self.para.Bc`.
            At (float, optional): Tensile damage shape parameter. Defaults to `self.para.At`.
            Bt (float, optional): Tensile damage curvature parameter. Defaults to `self.para.Bt`.
            clip (bool, optional): Whether to clip the output to [d0, 0.99]. Defaults to True.

        Returns:
            np.ndarray: Updated damage vector `d` [d_tc, d_t, d_c], constrained to [d0, 0.99] if clipping is enabled.

        Raises:
            ValueError: If `eps` is not a 1D NumPy array of shape (6,).
        """
        # Validate input shape
        eps = eps.ravel()
        if not isinstance(eps, np.ndarray) or eps.shape != (6,):
            raise ValueError("Input `eps` must be a NumPy array with shape (6,)")

        # Early return if strain is zero
        if np.allclose(eps, 0.0, atol=1e-12):
            return d0
        
        # Extract Material properties
        E = self.mat.prop.E                       # Young's modulus
        eps_0c = self.mat.prop.eps_0c             # Compressive damage threshold
        eps_tp = self.mat.prop.eps_tp             # Tensile damage threshold
        k = self.para.k 
        
        # Calculate constitutive matrix
        D_matrix = self.mat.calculate_D(E=E)      # Elastic stiffness matrix (6x6)
        
        # Calculate stress triaxiality
        r = self.calculate_r(d0=d0[0], D=D_matrix, eps=eps)
        
        #  Calculate interpolated damage parameters
        A, B = self.calculate_AB(r=r, k=k, Ac=Ac, Bc=Bc, At=At, Bt=Bt)
        
        # Calculate principal strain-based equivalent strain
        eps_p = calculate_principal_values(vec=eps, quantity='strain')
        Yt, Yc = self.calculate_Yt_Yc(eps_p=eps_p, eps_0t=eps_tp, eps_0c=eps_0c)
        Y = r * Yt + (1 - r) * Yc           # Weighted energy norm
        Y0 = r * eps_tp + (1 - r) * eps_0c  # Weighted threshold
        
        # μ-model evolution law
        if Y < Y0:
            return d0
        d = 1 - (1 - A) * Y0 / Y - A * np.exp(-B * (Y - Y0))
        
        # Damage validation and clipping
        is_valid = self.validate_damage_value(value=d, label='d')
        if not is_valid:
            print(f"μ-model: d = {d:.4f}, A = {A:.3f}, B = {B:.1f}, r = {r:.3f}, Y = {Y:.5f}, Y0 = {Y0:.5f}")
            self.logger.debug(
                "[DEBUG] Invalid `d` detected in `calculate_damage()` (Mu_Damage): "
                "d = %.6f | A = %.3f | B = %.1f | r = %.3f | Y = %.5f | Y0 = %.5f | "
                "eps = %s | eps_tp = %.6f | eps_0c = %.6f | d0 = %.4f",
                d, A, B, r, Y, Y0,
                np.round(eps, 6).tolist(), eps_tp, eps_0c, d0[0]
            )
        
        d_clip = float(np.clip(d, d0[0], 0.99)) if clip else float(d)
        
        return np.array([d_clip, d0[1], d0[2]])
    
    def calculate_uniaxial_compression_damage(self, eps: np.ndarray, Ac: Optional[float] = None, Bc: Optional[float] = None,
            At: Optional[float] = None, Bt: Optional[float] = None, clip: bool = False) -> np.ndarray:
        """Computes damage `d` under uniaxial compression using the μ-model for an array of strain values.

        This method constructs 3D strain states for uniaxial compression in the z-direction (ε_zz < 0),
        assuming lateral expansion (ε_xx = ε_yy > 0) based on Poisson’s ratio, and no shear strain.
        The resulting 6-component strain vectors are used to compute scalar damage values using
        the μ-model with interpolated parameters (A, B) from the given Ac/Bc (compression) and At/Bt (tension).

        Args:
            eps (np.ndarray): Array of uniaxial compressive strains (should be ≤ 0).
            Ac (float, optional): Compressive damage shape parameter. Defaults to `self.para.Ac`.
            Bc (float, optional): Compressive damage curvature parameter. Defaults to `self.para.Bc`.
            At (float, optional): Tensile damage shape parameter. Defaults to `self.para.At`.
            Bt (float, optional): Tensile damage curvature parameter. Defaults to `self.para.Bt`.
            clip (bool, optional): If True, clamps damage values to [d0, 0.99]. Defaults to False.

        Returns:
            np.ndarray: Array of computed scalar damage values.

        Raises:
            ValueError: If `eps` is not a 1D NumPy array.
        """
        if not isinstance(eps, np.ndarray) or eps.ndim != 1:
            raise ValueError("`eps` must be a 1D NumPy array.")

        nu = self.mat.prop.nu
        d0 = np.zeros((3,))
        d_vals = []
        
        # Compute synthetic strain tensors for uniaxial compression in z-direction
        eps_lat = nu * np.abs(eps)  # lateral positive strain in x/y
        eps_shear = np.zeros((len(eps), 3))      # no shear components

        # Build full strain in Voigt notation: [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_zx]
        eps_voigt = np.column_stack((eps_lat, eps_lat, -np.abs(eps), eps_shear))
        
        # Evaluate damage for each strain in Voigt notation
        for e_voigt in eps_voigt:
            d = self.calculate_damage(eps=e_voigt, d0=d0, Ac=Ac, Bc=Bc, At=At, Bt=Bt, clip=clip)
            d_vals.append(d[0])
        
        return np.array(d_vals)
    
    def calculate_uniaxial_tension_damage(self, eps: np.ndarray, Ac: Optional[float] = None, Bc: Optional[float] = None,
            At: Optional[float] = None, Bt: Optional[float] = None, clip: bool = False) -> np.ndarray:
        """Computes damage `d` under uniaxial tension using the μ-model for an array of strain values.

        This method constructs 3D strain states for uniaxial tension in the z-direction (ε_zz > 0),
        with lateral contraction (ε_xx = ε_yy < 0) computed from Poisson’s ratio, and zero shear components.
        The resulting strain vectors are passed to the μ-model damage law, which interpolates between
        tensile and compressive damage parameters using stress triaxiality.

        Args:
            eps (np.ndarray): Array of uniaxial tensile strain values (should be ≥ 0).
            Ac (float, optional): Shape parameter in compression damage evolution.
            Bc (float, optional): Curvature parameter in compression damage evolution.
            At (float, optional): Shape parameter in tension damage evolution.
            Bt (float, optional): Curvature parameter in tension damage evolution.
            clip (bool, optional): If True, clamps damage values to [d0, 0.99]. Defaults to False.

        Returns:
            np.ndarray: Array of computed scalar damage values.

        Raises:
            ValueError: If `eps` is not a 1D NumPy array.
        """
        if not isinstance(eps, np.ndarray) or eps.ndim != 1:
            raise ValueError("`eps` must be a 1D NumPy array.")

        nu = self.mat.prop.nu
        d0 = np.zeros((3,))
        d_vals = []
        
        # Compute synthetic strain tensors for uniaxial tension in z-direction
        eps_lat = -nu * np.abs(eps)  # lateral contraction in x/y
        eps_shear = np.zeros((len(eps), 3))      # no shear components

        # Build full strain tensor in Voigt notation: [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_zx]
        eps_voigt = np.column_stack((eps_lat, eps_lat, np.abs(eps), eps_shear))
        
        # Evaluate damage for each strain in Voigt notation
        for e_voigt in eps_voigt:
            d = self.calculate_damage(eps=e_voigt, d0=d0, Ac=Ac, Bc=Bc, At=At, Bt=Bt, clip=clip)
            d_vals.append(d[0])
        
        return np.array(d_vals)
    
    def calculate_damaged_uniaxial_compressive_stress(self, eps: np.ndarray, Ac: float, Bc: float, At: float, Bt: float, clip: bool = False) -> np.ndarray:
        """Computes uniaxial compressive stress incorporating damage effects via the μ-model.

        This method evaluates the scalar damage `d` for each compressive strain using the
        μ-model, and computes the corresponding degraded stress values using the damaged
        constitutive law: σ = (1 - d) · E · |ε|.

        The strain is assumed to be applied in the z-direction (ε_zz < 0), and the damage
        evolution accounts for both compressive and tensile contributions via (Ac, Bc, At, Bt).

        Args:
            eps (np.ndarray): Array of uniaxial compressive strains (negative or zero).
            Ac (float): Compressive damage shape parameter.
            Bc (float): Compressive damage curvature parameter.
            At (float): Tensile damage shape parameter (used for interpolation).
            Bt (float): Tensile damage curvature parameter (used for interpolation).
            clip (bool, optional): Whether to clamp damage values to [d0, 0.99]. Defaults to False.

        Returns:
            np.ndarray: Array of computed degraded stress values for each input strain.
        """
        # Compute damage values from uniaxial compressive strain history
        d_vals = self.calculate_uniaxial_compression_damage(eps=eps, Ac=Ac, Bc=Bc, At=At, Bt=Bt, clip=clip)
        
        # Compute corresponding stress: σ = (1 - d) · E · |ε|
        sig = (1 - d_vals) * self.mat.prop.E * np.abs(eps)
        
        return sig
    
    def calculate_damaged_uniaxial_tensile_stress(self, eps: np.ndarray, Ac: float, Bc: float, At: float, Bt: float, clip: bool = False) -> np.ndarray:
        """Computes uniaxial tensile stress with damage effects based on the μ-model.

        This method calculates the scalar damage `d` for each tensile strain value using the
        μ-model formulation and computes the corresponding degraded tensile stress as:
        σ = (1 - d) · E · |ε|.

        Although the strain is tensile (ε ≥ 0), the model interpolates between tensile and
        compressive damage parameters (At, Bt) and (Ac, Bc), enabling smooth transitions for
        mixed stress states.

        Args:
            eps (np.ndarray): Array of uniaxial tensile strains (non-negative).
            Ac (float): Compressive damage shape parameter.
            Bc (float): Compressive damage curvature parameter.
            At (float): Tensile damage shape parameter.
            Bt (float): Tensile damage curvature parameter.
            clip (bool, optional): Whether to clamp damage values to [d0, 0.99]. Defaults to False.

        Returns:
            np.ndarray: Array of degraded tensile stress values corresponding to input strains.
        """
        # Compute damage values from uniaxial tensile strain history
        d_vals = self.calculate_uniaxial_tension_damage(eps=eps, Ac=Ac, Bc=Bc, At=At, Bt=Bt, clip=clip)
        
        # Compute degraded stress response: σ = (1 - d) · E · |ε|
        sig = (1 - d_vals) * self.mat.prop.E * np.abs(eps)
        
        return sig
    
    def identify_A_B_from_stress(self, mode: str, initial_AB: Tuple[float, float, float, float],
            AB_bounds: Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]],
            eps_max: float, d_eps: float = 1e-6):
        """Identifies damage parameters (Ac, Bc, At, Bt) by fitting stress-strain data.

        This function fits the μ-model damage parameters to synthetic stress-strain data
        generated from the material model, using nonlinear least squares optimization.

        It supports both tension and compression modes and compares predicted stress
        with the stress generated from the baseline constitutive model. The best-fit
        parameters minimize the L2 loss between predicted and target stress.

        Args:
            mode (str): 'compression' or 'tension'. Defines which part of the curve to fit.
            initial_AB (Tuple[float, float, float, float]): Initial guess for (Ac, Bc, At, Bt).
            AB_bounds (Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]): 
                Lower and upper bounds for the four parameters.
            eps_max (float): Maximum strain value for curve fitting.
            d_eps (float, optional): Strain resolution. Defaults to 1e-6.

        Returns:
            Tuple[float, float, float, float]: Optimized values for (Ac, Bc, At, Bt).

        Raises:
            ValueError: If `mode` is not one of ['compression', 'tension'].
        """
        # Generate strain range (positive values for computation)
        eps = np.arange(0.0, np.abs(eps_max), d_eps)
        
        # Generate stress from material model
        if mode == 'compression':
            sig = -self.mat.calculate_stress(eps=-eps)  # Ensure stress is positive
            fit_func = self.calculate_damaged_uniaxial_compressive_stress
        elif mode == 'tension':
            sig = self.mat.calculate_stress(eps=eps)
            fit_func = self.calculate_damaged_uniaxial_tensile_stress
        else:
            raise ValueError("Mode must be 'compression' or 'tension'")

        # Fit (Ac, Bc, At, Bt) to match stress-strain response
        popt, pcov = curve_fit(
            f=fit_func,
            xdata=eps,
            ydata=sig,
            p0=initial_AB,
            bounds=AB_bounds
        )

        Ac_fit, Bc_fit, At_fit, Bt_fit = popt
        print(f'[Fit] Mode = {mode} | Ac = {Ac_fit:.2f}, Bc = {Bc_fit:.1f}, At = {At_fit:.2f}, Bt = {Bt_fit:.1f}')
        print(f'[Fit] Covariance matrix:\n{pcov}')
        self.logger.info(f'[Fit] Mode = {mode} | Ac = {Ac_fit:.2f}, Bc = {Bc_fit:.1f}, At = {At_fit:.2f}, Bt = {Bt_fit:.1f}')
        self.logger.info(f'[Fit] Covariance matrix:\n{pcov}')

        # Compute stress from initial and fitted parameters
        sig_init = fit_func(eps, *initial_AB)
        sig_fit = fit_func(eps, *popt)

        # Plot results
        fig, ax = plot_template(
            [eps, sig], [eps, sig_init], [eps, sig_fit],
            figsize=(12/2.54, 6/2.54),
            legend_labels=[
                "Target (from model)",
                f"Initial Guess (Ac={initial_AB[0]:.2f}, Bc={initial_AB[1]:.0f}, At={initial_AB[2]}, Bt={initial_AB[3]})",
                f"Fitted (Ac={Ac_fit:.2f}, Bc={Bc_fit:.0f}, At={At_fit:.2f}, Bt={Bt_fit:.0f})"
            ],
            colors=['black', (0, 47 / 255, 167 / 255), (192 / 255, 0, 0)],
            linestyles=['-', '--', '--'],
            xlabel=f"{mode.title()} Strain",
            ylabel="Stress (MPa)",
        )

        # Save and close
        fig.savefig(f"log/damage/Mu_parameters_fitted_from_stress_{mode}.png", dpi=300)
        plt.show()
        plt.close(fig)

        return Ac_fit, Bc_fit, At_fit, Bt_fit
    
    def plot_uniaxial_compression_damage_evolution(self, eps_c_max: float, d_eps: float, Ac: Optional[float] = None, Bc: Optional[float] = None,
            At: Optional[float] = None, Bt: Optional[float] = None,
            eps_c_min: float = 0.0,
            save_path: str ='log/damage/MU_uniaxial_compression_damage_evolution.png',
            show: bool = True, close: bool = True, clip: bool = False) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
        """Plots the uniaxial compressive stress-strain curve overlaid with the damage evolution curve.

        This function generates a composite figure consisting of:
            - The compressive stress-strain response computed from `self.mat`.
            - The corresponding scalar damage evolution curve `d(ε)` plotted on a secondary y-axis.

        The compressive strain is applied in the z-direction (ε_zz < 0), and lateral strains
        are generated assuming Poisson contraction. Damage is computed using the μ-model with
        interpolated parameters (A, B) for each strain.

        Args:
            eps_c_max (float): Maximum compressive strain (absolute value, positive input).
            d_eps (float): Strain increment used for sampling the strain range.
            Ac (float, optional): Compressive shape parameter. Defaults to `self.para.Ac`.
            Bc (float, optional): Compressive curvature parameter. Defaults to `self.para.Bc`.
            At (float, optional): Tensile shape parameter (used for interpolation).
            Bt (float, optional): Tensile curvature parameter (used for interpolation).
            eps_c_min (float, optional): Minimum compressive strain (positive value). Defaults to 0.0.
            save_path (str, optional): File path to save the plot. Defaults to `'log/damage/...png'`.
            show (bool, optional): Whether to display the plot interactively. Defaults to True.
            close (bool, optional): Whether to close the figure after saving. Defaults to True.
            clip (bool, optional): Whether to clamp damage values to [d0, 0.99]. Defaults to False.

        Returns:
            Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.axes.Axes]: 
                The generated figure, primary axis (stress), and secondary axis (damage).
        """
        # Generate compressive strain values (positive input for convenience)
        eps = np.arange(np.abs(eps_c_min), np.abs(eps_c_max), d_eps)
        
        # Generate compressive strain values (positive input for convenience)
        d_vals = self.calculate_uniaxial_compression_damage(eps=eps, Ac=Ac, Bc=Bc, At=At, Bt=Bt, clip=clip)
        
        # Generate base stress-strain plot
        fig, ax1 = self.mat.plot_stress_strain_compression(eps_c_max=eps_c_max, d_eps=d_eps, save_path='', show=False, close=False)
        
        # Overlay damage curve on secondary y-axis
        ax2 = self.plot_right_damage_evolution(ax=ax1, eps=eps, d_vals=d_vals)

        # Adjust layout for clarity
        fig.tight_layout()
        
        # Save the figure
        fig.savefig(save_path, dpi=300)
        
        # Show the plot if requested
        if show:
            plt.show()

        # Close the figure if requested
        if close:
            plt.close(fig)
    
        return fig, ax1, ax2
    
    def plot_uniaxial_tension_damage_evolution(self, eps_t_max: float, d_eps: float, Ac: Optional[float] = None, Bc: Optional[float] = None,
            At: Optional[float] = None, Bt: Optional[float] = None,
            eps_t_min: float = 0.0,
            save_path: str ='log/damage/MU_uniaxial_tension_damage_evolution.png',
            show: bool = True, close: bool = True, clip: bool = False) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
        """Plots the uniaxial tensile stress-strain curve with overlaid damage evolution curve.

        This function creates a two-axis plot showing:
            - The tensile stress-strain response generated from `self.mat`.
            - The corresponding scalar damage curve `d(ε)` computed using the μ-model
                and plotted on a secondary y-axis.

        Strain is applied in the z-direction (ε_zz > 0) with lateral contraction (ε_xx, ε_yy < 0)
        assumed from Poisson's effect. Damage parameters can be customized or default to `self.para`.

        Args:
            eps_t_max (float): Maximum tensile strain (positive).
            d_eps (float): Strain increment step.
            Ac (float, optional): Compressive damage shape parameter. Defaults to `self.para.Ac`.
            Bc (float, optional): Compressive damage curvature parameter. Defaults to `self.para.Bc`.
            At (float, optional): Tensile damage shape parameter. Defaults to `self.para.At`.
            Bt (float, optional): Tensile damage curvature parameter. Defaults to `self.para.Bt`.
            eps_t_min (float, optional): Minimum tensile strain. Defaults to 0.0.
            save_path (str, optional): Path to save the output figure. Defaults to `'log/damage/...png'`.
            show (bool, optional): Whether to show the plot interactively. Defaults to True.
            close (bool, optional): Whether to close the figure after saving. Defaults to True.
            clip (bool, optional): Whether to clip damage values to [d0, 0.99]. Defaults to False.

        Returns:
            Tuple[Figure, Axes, Axes]: Matplotlib figure, primary (stress) axis, and secondary (damage) axis.
        """
        # Generate tensile strain values (positive for calculation)
        eps = np.arange(np.abs(eps_t_min), np.abs(eps_t_max), d_eps)
        
        # Evaluate damage values for each tensile strain
        d_vals = self.calculate_uniaxial_tension_damage(eps=eps, Ac=Ac, Bc=Bc, At=At, Bt=Bt, clip=clip)
        
        # Generate base stress-strain plot
        fig, ax1 = self.mat.plot_stress_strain_tension(eps_t_max=eps_t_max, d_eps=d_eps, save_path='', show=False, close=False)
        
        # Overlay damage curve on secondary y-axis
        ax2 = self.plot_right_damage_evolution(ax=ax1, eps=eps, d_vals=d_vals)

        # Adjust layout for clarity
        fig.tight_layout()
        
        # Save the figure
        fig.savefig(save_path, dpi=300)
        
        # Show the plot if requested
        if show:
            plt.show()

        # Close the figure if requested
        if close:
            plt.close(fig)
    
        return fig, ax1, ax2
    
    def plot_damaged_uniaxial_tensile_stress(self, eps_t_max: float, d_eps: float, Ac: float, Bc: float, At: float, Bt: float,
                show: bool = True, close: bool = True, clip: bool = False) -> Tuple[plt.Figure, plt.Axes]:
        """Plots the damaged uniaxial tensile stress-strain curve using the specified damage parameters.

        Args:
            eps_t_max (float): Maximum tensile strain (positive) to evaluate.
            d_eps (float): Strain increment for evaluation.
            Ac (float): Compressive damage shape parameter.
            Bc (float): Compressive damage curvature parameter.
            At (float): Tensile damage shape parameter.
            Bt (float): Tensile damage curvature parameter.
            show: Whether to display the figure interactively.
            close: Whether to close the figure after saving (to free memory).
            clip (bool, optional): Whether to clamp damage values to [d0, 0.99]. Defaults to False.

        Returns:
            Tuple[plt.Figure, plt.Axes]: The matplotlib figure and axes of the generated plot.
        """
        # Define tensile strain range (positive values only)
        eps = np.arange(0.0, np.abs(eps_t_max), d_eps)
        
        # Compute corresponding stress values
        sig = self.calculate_damaged_uniaxial_tensile_stress(eps=eps, Ac=Ac, Bc=Bc, At=At, Bt=Bt, clip=clip)
        
        fig, ax = self.mat.plot_stress_strain_tension(eps_t_max=eps_t_max, d_eps=d_eps, save_path='', show=False, close=False)
        ax.plot(eps, sig)
        # Plot using the configured template
        # fig, ax = plot_template([eps, sig],
        #             xlabel='Tensile Strain',
        #             ylabel='Stress (MPa)',
        #             )
        
        # Show the plot if requested
        if show:
            plt.show()

        # Close the figure if requested
        if close:
            plt.close(fig)
        
        return fig, ax
    
    def plot_damaged_uniaxial_compressive_stress(self, eps_c_max: float, d_eps: float, Ac: float, Bc: float, At: float, Bt: float,
                show: bool = True, close: bool = True, clip: bool = False) -> Tuple[plt.Figure, plt.Axes]:
        """Plots the damaged uniaxial compressive stress-strain curve for a given damage model.

        Args:
            eps_c_max (float): Maximum compressive strain (positive value).
            d_eps (float): Strain increment for evaluation.
            Ac (float): Compressive damage shape parameter.
            Bc (float): Compressive damage curvature parameter.
            At (float): Tensile damage shape parameter.
            Bt (float): Tensile damage curvature parameter.
            show: Whether to display the figure interactively.
            close: Whether to close the figure after saving (to free memory).
            clip (bool, optional): Whether to clamp damage values to [d0, 0.99]. Defaults to False.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Matplotlib figure and axes objects of the plot.
        """
        # Define compressive strain range (positive for evaluation)
        eps = np.arange(0.0, np.abs(eps_c_max), d_eps)
        
        # Compute corresponding stress values
        sig = self.calculate_damaged_uniaxial_tensile_stress(eps=eps, Ac=Ac, Bc=Bc, At=At, Bt=Bt, clip=clip)
        
        fig, ax = self.mat.plot_stress_strain_compression(eps_c_max=eps_c_max, d_eps=d_eps, save_path='', show=False, close=False)
        ax.plot(eps, sig)
        
        # Create stress-strain plot
        # fig, ax = plot_template([eps, sig],
        #             xlabel='Compressive Strain',
        #             ylabel='Stress (MPa)',
        #             )
        
        # Show the plot if requested
        if show:
            plt.show()

        # Close the figure if requested
        if close:
            plt.close(fig)
        
        return fig, ax
    
class Modified_Mazars_Damage(Damage):
    """Implements the Modified Mazars damage model for concrete, integrating energy-based regularization 
    in both tension and compression as proposed by:

    - Arruda et al., *A modified Mazars damage model with energy regularization*, EFM, 2022.
    - Debuisine et al., *On the Need of Compressive Regularization in Damage Models for Concrete*, 2024.

    This model extends the original Mazars isotropic damage formulation by incorporating fracture 
    energy terms to control softening and localization. It introduces two additional parameters: 
    tensile fracture energy `G_ft` and compressive fracture energy `G_fc`, which enable objective 
    energy dissipation with respect to mesh size.

    Attributes:
        mat (Material): Material object with mechanical and constitutive properties.
        para (Damage_Parameter): Damage parameters including:
            - At (float): Tensile shape parameter.
            - Bt (float): Tensile curvature parameter.
            - Ac (float): Compressive shape parameter.
            - Bc (float): Compressive curvature parameter.
            - G_ft (float): Fracture energy in tension (N/mm).
            - G_fc (float): Fracture energy in compression (N/mm).
    """
    
    def __init__(self, mat: Material, mesh: Mesh, para: Damage_Parameter) -> None:
        """Initializes the Modified Mazars model with material and fracture energy-based damage parameters.

        Args:
            mat (Material): Material instance containing elastic and strain limit definitions.
            mesh (Mesh): Mesh object containing node, element and meshing data.
            para (Damage_Parameter): Struct containing damage evolution parameters including 
                {'At', 'Bt', 'Ac', 'Bc', 'G_ft', 'G_fc'}.
        """
        super().__init__(mat, para)
        self.mat = mat
        self.mesh = mesh
        self.para = para
        
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info("Initialized Modified_Mazars_Damage model with parameters: %s", vars(self.para))
    
    def calculate_dt_Debuisne(self, eps_eq: np.ndarray, eps_d0: float, G_ft: float, h: float) -> float:
        """Computes tensile damage `dt` based on the modified Mazars model with energy regularization.

        This method follows the formulation by Debuisine et al. (2024), specifically Equation (12), 
        which introduces a simplified energy-based expression for the tensile damage variable `dt` 
        and computes `Bt` as a function of fracture energy `G_ft` and characteristic length `h`.

        Args:
            eps_eq (float): Equivalent strain in tension.
            eps_d0 (float): Damage threshold strain in tension, ε₀ = f_t / E.
            G_ft (float): Fracture energy in tension (N/mm).
            h (float): Characteristic length of the element (mm).
            clip (bool, optional): Whether to clamp `dt` to the range [0, 0.99]. Defaults to False.

        Returns:
            float: Computed damage variable `dt` ∈ [0, 1].

        Reference:
            Debuisine, M. et al., *On the Need of Compressive Regularization in Damage Models for Concrete*,
            Applied Mechanics, 2024. Equation (12).
        """
        if eps_eq <= eps_d0:
            return 0.0
        
        ft = self.mat.prop.ft
        try:
            Bt = ft * h / (G_ft - ft * eps_d0 * h / 2)
        except ZeroDivisionError:
            raise ValueError("Invalid Bt calculation: denominator is zero. Check G_ft, f_t, ε_d0, and h.")
        
        dt = 1 - eps_d0 / eps_eq * np.exp(-Bt * (eps_eq - eps_d0))
        
        is_valid = self.validate_damage_value(value=dt, label='dt')
        if not is_valid:
            self.logger.debug(
                "[DEBUG] Invalid `dt` detected in `calculate_dt_Debuisne()`: "
                "dt = %.6f | eps_eq = %.6f | eps_d0 = %.6f | G_ft = %.3f | h = %.3f | Bt = %.3f",
                dt, eps_eq, eps_d0, G_ft, h, Bt
            )
        
        return dt
    
    def calculate_dt_Arruda(self, eps_eq: np.ndarray, eps_d0: float, G_ft: float, Lc: float) -> float:
        """Compute tensile damage `dt` based on the energy-regularized Arruda (Debuisine) formulation.

        This function implements the exponential and linear tension softening model derived from:
        Debuisine et al. (2024), Eq. (12). The fracture energy `G_ft` and characteristic length `Lc` 
        are used to regularize damage growth and avoid mesh dependency.

        Args:
            eps_eq (float): Equivalent tensile strain.
            eps_d0 (float): Initial damage threshold strain (ε₀ = f_t / E).
            G_ft (float): Fracture energy in tension (N/mm).
            Lc (float): Characteristic element length (mm).

        Returns:
            float: Tensile damage variable `dt`, in range [0, 1].
        """
        if eps_eq <= eps_d0:
            return 0.0
        
        # Material parameters
        ft = self.mat.prop.ft
        eps_t1 = self.mat.prop.eps_tp
        pt = self.mat.prop.omega_tu  # Residual stress ratio
        alpha_ct = self.mat.prop.alpha_ct
        soften_type = self.mat.prop.soften.lower()

        # Residual strain and regularized ultimate strain
        eps_t_res = self.mat.calculate_residual_tensile_eps()
        eps_tu = G_ft / (Lc * ft) + eps_d0  # Energy-regularized ε_tu
        
        # Compute dt based on exponential vs linear segment
        if eps_eq <= eps_t_res:
            dt = 1 - eps_d0 * np.exp((eps_t1 - eps_eq) / (eps_tu - eps_t1)) / eps_eq
        else:
            dt = 1 - pt * eps_d0 / eps_eq
        
        # Validation and debug logging
        is_valid = self.validate_damage_value(value=dt, label='dt')
        if not is_valid:
            self.logger.debug(
                "[DEBUG] Invalid `dt` in `calculate_dt_Arruda()`: "
                "dt = %.6f | eps_eq = %.6f | eps_d0 = %.6f | eps_t1 = %.6f | eps_t_res = %.6f | "
                "eps_tu = %.6f | G_ft = %.3f | Lc = %.3f | pt = %.3f | soften = '%s'",
                dt, eps_eq, eps_d0, eps_t1, eps_t_res, eps_tu, G_ft, Lc, pt, soften_type
            )
        
        return dt
    
    def calculate_dc_Debuisne(self, eps_eq: np.ndarray, G_fc: float, h: float) -> float:
        """Computes the compressive damage variable `dc` using the regularized 
        Debuisne-Mazars RTC model (2024), which includes plateau and softening phases.

        This function implements the piecewise damage law described in Debuisne et al. (2024)
        for uniaxial compression, based on equivalent strain `eps_eq`, fracture energy `G_fc`,
        and element characteristic length `h`.

        Args:
            eps_eq (np.ndarray): Equivalent strain under uniaxial compression (scalar).
            G_fc (float): Compressive fracture energy (N/mm).
            h (float): Characteristic element length (mm).
            clip (bool, optional): If True, clamps damage `dc` to [0, 0.99]. Defaults to False.

        Returns:
            float: Damage variable `dc` ∈ [0, 1).

        Reference:
            Debuisne, M., et al. (2024). On the Need of Compressive Regularization in Damage Models for Concrete.
            Applied Mechanics, 5(3), 490–512. doi:10.3390/applmech5030028
        """
        # Extract material constants
        E = self.mat.prop.E
        nu = self.mat.prop.nu
        fc = self.mat.prop.fc
        eps_c1 = self.mat.prop.eps_cp
        eps_c2 = self.mat.prop.eps_cp2
        
        # Compute derived fracture strain parameter (Equation 16)
        eps_cu = 2 * G_fc / (h * fc) - (eps_c2 - eps_c1)
        
        # Equivalent strain rescaling (κ_c and κ̄_c)
        kappa_c = eps_eq / (nu * np.sqrt(2)) # uniaxial equivalent strain
        kappa_c_bar = kappa_c / eps_c1
        
        # k, k1, k2 for piecewise damage law
        k = 1.05 * E * eps_c1 / fc   # shape factor
        k_1 = fc / (eps_cu - eps_c2)
        k_2 = fc + k_1 * eps_c2
        
        # Threshold to avoid nonphysical softening
        thres = 0.05 * k * eps_c1 / (1.05 + k * (k - 2))
        
        # Compute piecewise damage d_c (Equation 15)
        if kappa_c <= thres:
            dc = 0.0
        if thres < kappa_c <= eps_c1:
            num = (k * kappa_c_bar - kappa_c_bar**2) * fc
            den = (1 + (k - 2) * kappa_c_bar) * E * kappa_c
            dc = 1 - num / den
        elif eps_c1 < kappa_c <= eps_c2:
            dc = 1 - fc / (E * kappa_c)
        elif eps_c2 < kappa_c <= eps_cu:
            dc = 1 + k_1 / E - k_2 / (E * kappa_c)
        else:
            dc = 0.99
        
        is_valid = self.validate_damage_value(value=dc, label='dc')
        if not is_valid:
            self.logger.debug(
                "[DEBUG] Invalid `dc` detected in `calculate_dc_Debuisne()`:\n"
                "  dc        = %.6f\n"
                "  eps_eq    = %.6e\n"
                "  G_fc      = %.2f N/mm, h = %.3f mm\n"
                "  kappa_c   = %.6e, kappa_c_bar = %.6e\n"
                "  thres     = %.6e\n"
                "  E         = %.1f MPa, fc = %.2f MPa\n"
                "  eps_c1    = %.6e, eps_c2 = %.6e, eps_cu = %.6e\n"
                "  k         = %.3f, k_1 = %.3f, k_2 = %.3f\n",
                dc, eps_eq, G_fc, h, kappa_c, kappa_c_bar, thres,
                E, fc, eps_c1, eps_c2, eps_cu, k, k_1, k_2
            )
        
        return dc
    
    def calculate_dc_Arruda(self, eps_eq: np.ndarray, G_fc: float, Lc: float) -> float:
        """Computes the compressive damage variable `dc` based on the Arruda-type 
        regularized damage model (Shen et al., 2024) using a 4-part piecewise strain law 
        for uniaxial compression with energy-based softening.

        This implementation follows the explicit damage evolution laws defined by:
            - κ_c   = ε_eq / (ν * √2)
            - ε_cu  from energy-based Eq. (24)
            - Piecewise `dc = g_c(κ_c)` from Eq. (22)

        Args:
            eps_eq (np.ndarray): Equivalent strain under uniaxial compression.
            G_fc (float): Compressive fracture energy (N/mm).
            Lc (float): Characteristic length of the finite element (mm).

        Returns:
            float: Compressive damage value `dc` ∈ [0, 1).

        Reference:
            Shen, J., Arruda, T., Pagani, A. (2024).
            "A consistent crack bandwidth for higher-order beam theories: Application to concrete."
            International Journal of Damage Mechanics, 33(4), 269–292.
            https://doi.org/10.1177/10567895231215557
        """
        # Extract material constants
        E = self.mat.prop.E
        nu = self.mat.prop.nu
        fc = self.mat.prop.fc
        eps_c1 = self.mat.prop.eps_cp
        eps_c2 = self.mat.prop.eps_cp2
        pc = self.mat.prop.omega_cu
        
        # Derived strain limit based on fracture energy (Eq. 24)
        eps_cu = 2 * G_fc / (Lc * fc) - (eps_c2 - eps_c1) # Ultimate strain for softening slope
        
        # Equivalent strain mapping (Eq. 23)
        kappa_c = eps_eq / (nu * np.sqrt(2)) # Scaled equivalent strain κ_c
        kappa_c_bar = kappa_c / eps_c1       # Normalized κ̄_c = κ_c / ε_c1
        
        # k, k1, k2 for piecewise damage law
        k = 1.05 * E * eps_c1 / fc   # shape factor
        k_1 = fc / (eps_cu - eps_c2) # Softening slope
        k_2 = fc + k_1 * eps_c2      # Stress at ε_c2
        
        # Residual branch cutoff (ε_cres) based on post-peak residual
        eps_c_res = self.mat.calculate_residual_compressive_eps(eps_cu=eps_cu)
        
        # Piecewise damage function (Eq. 22)
        if kappa_c <= eps_c1:
            num = (k * kappa_c_bar - kappa_c_bar**2) * fc
            den = (1 + (k - 2) * kappa_c_bar) * E * kappa_c
            dc = 1 - num / den
        elif eps_c1 < kappa_c <= eps_c2:
            dc = 1 - fc / (E * kappa_c)
        elif eps_c2 < kappa_c <= eps_c_res:
            dc = 1 + k_1 / E - k_2 / (E * kappa_c)
        else:
            dc = 1 - pc * fc / (E * kappa_c)
        
        # --- Validate and debug logging if needed ---
        is_valid = self.validate_damage_value(value=dc, label='dc')
        if not is_valid:
            self.logger.debug(
                "[DEBUG] Invalid `dc` in `calculate_dc_Arruda()`:\n"
                "  Computed dc           = %.6f\n"
                "  Input eps_eq          = %.6e\n"
                "  G_fc                  = %.2f N/mm\n"
                "  Lc                    = %.3f mm\n"
                "  nu (Poisson)          = %.3f\n"
                "  kappa_c               = %.6e\n"
                "  kappa_c_bar           = %.6e\n"
                "  eps_c1 (peak strain)  = %.6e\n"
                "  eps_c2 (plateau end)  = %.6e\n"
                "  eps_cu (ultimate)     = %.6e\n"
                "  eps_cres (residual)   = %.6e\n"
                "  fc (strength)         = %.2f MPa\n"
                "  E  (modulus)          = %.1f MPa\n"
                "  k (pre-peak factor)   = %.3f\n"
                "  k1 (softening slope)  = %.3f\n"
                "  k2 (plateau exit)     = %.3f\n"
                "  pc (residual ratio)   = %.3f",
                dc, eps_eq, G_fc, Lc, nu, kappa_c, kappa_c_bar,
                eps_c1, eps_c2, eps_cu, eps_c_res,
                fc, E, k, k_1, k_2, pc
            )
        
        return dc
    
    def calculate_damage(self, eps: np.ndarray, d0: np.ndarray, clip: bool = True) -> np.ndarray:
        """Computes scalar damage `d` using Modified Mazars model with energy regularization 
        in both tension and compression.

        This method uses a combination of Debuisne et al.'s 2024 regularized 
        tension and compression laws, incorporating characteristic element size `h`.

        Args:
            eps (np.ndarray): Strain tensor in Voigt notation with shape (6,).
            d0 (float): Previous damage value.
            h (float): Characteristic element size for regularization.
            control (str): One of {"t", "c", "ct", "tc", ...} for mode selection.
            clip (bool, optional): Whether to clamp damage to [d0, 0.99]. Defaults to True.

        Returns:
            float: Computed scalar damage variable `d` ∈ [0, 1).

        Raises:
            ValueError: If input shape is invalid or control mode is unrecognized.
        """
        # Validate and sanitize input
        eps = eps.ravel()
        if not isinstance(eps, np.ndarray) or eps.shape != (6,):
            raise ValueError("Input `eps` must be a NumPy array with shape (6,)")

        # Early return if strain is zero
        if np.allclose(eps, 0.0, atol=1e-12):
            return d0
        
        # Unpack previous damage state
        d0_tc, d0_t, d0_c = d0 
        
        # Material parameters
        eps_tp = self.mat.prop.eps_tp
        G_ft = self.para.G_ft
        G_fc = self.para.G_fc
        control = self.para.control.lower().strip()
        h = self.mesh.para.mesh_size[0]
        
        model_type = self.para.model_type.lower()
        
        # Compute equivalent strain
        eps_eq = calculate_equivalent_strain_from_voigt(vec=eps)
        
        # Compute tension damage
        if model_type in {'d', 'debuisne'}:
            dt = self.calculate_dt_Debuisne(eps_eq=eps_eq, eps_d0=eps_tp, G_ft=G_ft, h=h)
        elif model_type in {'a', 'arruda'}:
            dt = self.calculate_dt_Arruda(eps_eq=eps_eq, eps_d0=eps_tp, G_ft=G_ft, Lc=h)
        else:
            raise ValueError(f"Expected `model_type` to be 'debuisne' or 'arruda', got '{model_type}'.")
        if control in {"t", "tension"}:
            dt = np.clip(dt, d0_t, 0.99) if clip else dt
            return np.array([d0_tc, dt, d0_c])
        
        # Compute compression damage
        if model_type in {'d', 'debuisne'}:
            dc = self.calculate_dc_Debuisne(eps_eq=eps_eq, G_fc=G_fc, h=h)
        elif model_type in {'a', 'arruda'}:
            dc = self.calculate_dc_Arruda(eps_eq=eps_eq, G_fc=G_fc, Lc=h)
        else:
            raise ValueError(f"Expected `model_type` to be 'debuisne' or 'arruda', got '{model_type}'.")
        if control in {"c", "compression"}:
            dc = np.clip(dc, d0, 0.99) if clip else dc
            return np.array([d0_tc, d0_t, dc])
        
        # Compute tension and compression damage
        if control in {"ct", "tc", "compression-tension", "tension-compression"}:
            # Calculate weighting factors for combined damage
            alpha_t, alpha_c = self.calculate_alpha(eps=eps, d0=d0[0])
            
            # Combine weighted damage contributions
            d = alpha_t * dt + alpha_c * dc
            
            # Validation and debug if invalid
            is_valid = self.validate_damage_value(value=d, label="d")
            if not is_valid:
                print(f'\tdc = {dc:.2f}, dt = {dt:.2f}, alpha_c = {alpha_c:.2f}, alpha_t = {alpha_t:.2f}')
                self.logger.debug(
                    "[DEBUG] Invalid `d` in `calculate_damage()` (Modified_Mazars_Damage):\n"
                    "  d       = %.6f\n"
                    "  dc      = %.6f\n"
                    "  dt      = %.6f\n"
                    "  alpha_c = %.3f\n"
                    "  alpha_t = %.3f\n"
                    "  d0_tc   = %.6f\n"
                    "  d0_t    = %.6f\n"
                    "  d0_c    = %.6f\n"
                    "  eps     = %s\n"
                    "  eps_eq  = %.6f\n"
                    "  eps_d0  = %.6f\n"
                    "  G_ft    = %.3f N/mm, G_fc = %.3f N/mm, h = %.3f mm",
                    d, dc, dt, alpha_c, alpha_t,
                    d0_tc, d0_t, d0_c,
                    np.round(eps, 6).tolist(), eps_eq, eps_tp,
                    G_ft, G_fc, h
                )
            
            d = np.clip(d, d0[0], 0.99) if clip else d
            
            return np.array([d, dt, dc])
        
        raise ValueError(
            f"Invalid control mode '{control}'. "
            "Expected one of: 't', 'c', 'ct', 'tc', 'tension', 'compression', 'compression-tension'."
        )
    
    def calculate_uniaxial_compression_damage(self, eps: np.ndarray, clip: bool = False) -> np.ndarray:
        """Computes uniaxial compression damage `d` for a series of strain values.

        This method constructs synthetic 3D strain tensors representing uniaxial 
        compression in the z-direction (ε_zz < 0) using Poisson's ratio to compute 
        lateral tensile strains (ε_xx = ε_yy > 0). It then evaluates the scalar 
        damage variable using the Modified Mazars damage law with energy regularization.

        Args:
            eps (np.ndarray): Array of uniaxial compressive strains (should be ≤ 0), shape (N,).
            h (float): Characteristic element length for energy regularization.
            clip (bool, optional): Whether to clamp damage values to [0, 0.99]. Defaults to False.

        Returns:
            np.ndarray: Array of computed damage values corresponding to input strains.

        Raises:
            ValueError: If `eps` is not a 1D NumPy array.
        """
        if not isinstance(eps, np.ndarray) or eps.ndim != 1:
            raise ValueError("`eps` must be a 1D NumPy array.")
        
        nu = self.mat.prop.nu
        d0 = np.zeros((3,))
        d_vals = []
        
        # Compute synthetic strain tensors for uniaxial compression in z-direction
        eps_lat = np.sqrt(2) * nu * np.abs(eps)  # lateral positive strain in x/y
        eps_shear = np.zeros((len(eps), 3))      # no shear components

        # Build full strain in Voigt notation: [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_zx]
        eps_voigt = np.column_stack((eps_lat, eps_lat, -np.abs(eps), eps_shear))
        
        # Evaluate damage for each strain in Voigt notation
        for e_voigt in eps_voigt:
            d = self.calculate_damage(eps=e_voigt, d0=d0, clip=clip)
            d0 = d
            d_vals.append(d[0])
        
        return np.array(d_vals)
    
    def calculate_uniaxial_tension_damage(self, eps: np.ndarray, clip: bool = False) -> np.ndarray:
        """Computes uniaxial tension damage `d` for a series of axial strain values.

        This method simulates uniaxial tension in the z-direction (ε_zz > 0) by
        constructing synthetic 3D strain tensors. Lateral compressive strains 
        (ε_xx = ε_yy < 0) are derived using Poisson's ratio, and shear components 
        are assumed to be zero. The Modified Mazars damage model is then used 
        to compute the damage variable for each strain state.

        Args:
            eps (np.ndarray): Array of uniaxial tensile strain values (should be ≥ 0), shape (N,).
            h (float): Characteristic element length for energy regularization.
            clip (bool, optional): Whether to clamp damage values to [0, 0.99]. Defaults to False.

        Returns:
            np.ndarray: Array of computed damage values corresponding to input strains.

        Raises:
            ValueError: If `eps` is not a 1D NumPy array.
        """
        if not isinstance(eps, np.ndarray) or eps.ndim != 1:
            raise ValueError("`eps` must be a 1D NumPy array.")
        
        nu = self.mat.prop.nu
        d0 = np.zeros((3,))
        d_vals = []
        
        # Compute synthetic strain tensors for uniaxial compression in z-direction
        eps_lat = -np.sqrt(2) * nu * np.abs(eps)  # lateral compressive strain in x/y
        eps_shear = np.zeros((len(eps), 3))       # no shear components

        # Build full strain in Voigt notation: [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_zx]
        eps_voigt = np.column_stack((eps_lat, eps_lat, np.abs(eps), eps_shear))
        
        # Evaluate damage for each strain in Voigt notation
        for e_voigt in eps_voigt:
            d = self.calculate_damage(eps=e_voigt, d0=d0, clip=clip)
            d0 = d
            d_vals.append(d[0])
        
        return np.array(d_vals)
    
    def plot_uniaxial_compression_damage_evolution(self, eps_c_max: float, d_eps: float,
            eps_c_min: float = 0.0,
            save_path: str ='log/damage/MM_uniaxial_compression_damage_evolution.png',
            show: bool = True, close: bool = True, clip: bool = False) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
        """Plots uniaxial compression stress-strain and corresponding damage evolution curves.

        This method overlays the Modified Mazars damage evolution curve on top of 
        the stress-strain plot for uniaxial compression. The stress-strain curve is 
        generated from the material model, while the damage curve is computed using 
        the Debuisne regularized formulation based on element size `h`.

        Args:
            eps_c_max (float): Maximum compressive strain (should be negative).
            d_eps (float): Strain step size for sampling.
            h (float): Characteristic element length for energy regularization.
            eps_c_min (float, optional): Minimum compressive strain (positive value). Defaults to 0.0.
            save_path (str, optional): Path to save the generated figure. Defaults to 'log/damage/...'.
            show (bool, optional): Whether to display the figure interactively. Defaults to True.
            close (bool, optional): Whether to close the figure after saving. Defaults to True.
            clip (bool, optional): Whether to clamp damage values to [0, 0.99]. Defaults to False.

        Returns:
            Tuple[plt.Figure, plt.Axes, plt.Axes]: The generated Matplotlib figure,
            the primary axis for stress-strain, and the secondary axis for damage evolution.
        """
        # Generate compressive strain samples (all positive for internal calculation)
        eps = np.arange(np.abs(eps_c_min), np.abs(eps_c_max), d_eps)
        
        # Compute damage evolution using Modified Mazars law
        d_vals = self.calculate_uniaxial_compression_damage(eps=eps, clip=clip)
        
        # Plot base stress-strain curve from the material model
        fig, ax1 = self.mat.plot_stress_strain_compression(eps_c_max=eps_c_max, d_eps=d_eps, save_path='', show=False, close=False)
        
        # Overlay damage evolution curve on secondary y-axis
        ax2 = self.plot_right_damage_evolution(ax=ax1, eps=eps, d_vals=d_vals)

        # Adjust layout for clarity
        fig.tight_layout()
        
        # Save the figure
        fig.savefig(save_path, dpi=300)
        
        # Show the plot if requested
        if show:
            plt.show()

        # Close the figure if requested
        if close:
            plt.close(fig)
    
        return fig, ax1, ax2
    
    def plot_uniaxial_tension_damage_evolution(self, eps_t_max: float, d_eps: float,
            eps_t_min: float = 0.0,
            save_path: str ='log/damage/MM_uniaxial_tension_damage_evolution.png',
            show: bool = True, close: bool = True, clip: bool = False) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
        """Plots uniaxial tension stress-strain and corresponding damage evolution curves.

        This method overlays the Modified Mazars damage evolution curve on top of 
        the stress-strain curve for uniaxial tension. The stress-strain response 
        is generated from the material model, while the damage curve is computed 
        using energy-based regularization with the Debuisne formulation.

        Args:
            eps_t_max (float): Maximum tensile strain (positive).
            d_eps (float): Strain step size.
            h (float): Characteristic element length for regularization.
            eps_t_min (float, optional): Minimum tensile strain. Defaults to 0.0.
            save_path (str, optional): Path to save the plot. Defaults to 'log/damage/...'.
            show (bool, optional): If True, displays the figure interactively. Defaults to True.
            close (bool, optional): If True, closes the figure after saving. Defaults to True.
            clip (bool, optional): If True, clamps damage values to [0, 0.99]. Defaults to False.

        Returns:
            Tuple[plt.Figure, plt.Axes, plt.Axes]: The Matplotlib figure, 
            primary axis for stress-strain, and secondary axis for damage.
        """
        # Generate tensile strain samples (positive range)
        eps = np.arange(np.abs(eps_t_min), np.abs(eps_t_max), d_eps)
        
        # Compute damage evolution using Modified Mazars law
        d_vals = self.calculate_uniaxial_tension_damage(eps=eps, clip=clip)
        
        # Generate base stress-strain plot from the material model
        fig, ax1 = self.mat.plot_stress_strain_tension(eps_t_max=eps_t_max, d_eps=d_eps, save_path='', show=False, close=False)
        
        # Overlay damage evolution curve on secondary y-axis
        ax2 = self.plot_right_damage_evolution(ax=ax1, eps=eps, d_vals=d_vals)

        # Adjust layout for clarity
        fig.tight_layout()
        
        # Save the figure
        fig.savefig(save_path, dpi=300)
        
        # Show the plot if requested
        if show:
            plt.show()

        # Close the figure if requested
        if close:
            plt.close(fig)
    
        return fig, ax1, ax2
    