# fem/fem/materials.py


import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Union, Iterable
from abc import ABC, abstractmethod


from fem.parameters import Concrete_Property
from fem.utils.utils import plot_template, annotate_point
from fem.utils.log_config import setup_logger


class Material(ABC):
    """A class representing a material with different constitutive models.
    
    This class allows defining a material and computing its damage variable
    using Mazars' isotropic damage model.
    
    Attributes:
        prop (Concrete_Property): A dataclass containing material properties.
    """
    def __init__(self, prop: Concrete_Property):
        """Initializes a Material instance.
        
        Args:
            prop (Concrete_Property): Material property object.
        """
        self.prop = prop
    
    def print_properties(self) -> None:
        """Prints the material properties in a formatted, readable layout.
        
        Iterates through all fields in the `prop` dataclass and prints
        each property name and value, aligned and sorted for readability.
        """
        print(f"\nMaterial Properties ({self.__class__.__name__}):")
        for key, value in vars(self.prop).items():
            print(f"{key:<20}: {value}")
    
    def calculate_D(self, E: Union[float, np.ndarray]) -> np.ndarray:
        """Computes the material stiffness (constitutive) matrix for isotropic materials.
        
        This function calculates the 6x6 constitutive matrix (stiffness matrix) based on 
        Young's modulus (E) and Poisson's ratio (ν) for an isotropic material.
        
        The constitutive matrix is derived using the isotropic elasticity relations, 
        where the material behavior follows Hooke’s Law in 3D.
        
        Reference:
            - C.3 Page 869 (presumably a reference from a book on elasticity)
            
        Args:
            E (Union[float, np.ndarray]): Young's modulus.
            - If float: a scalar value representing isotropic modulus.
            - If np.ndarray: must be of shape (6,) to represent per-direction stiffness.
        
        Returns:
            np.ndarray: A 6x6 material stiffness matrix (D), representing the isotropic 
                elasticity tensor in Voigt notation.
        
        Raises:
            ValueError: If E is not a float or a NumPy array of shape (6,).
        """
        # Retrieve Poisson's ratio from material properties
        nu = self.prop.nu # Poisson's ratio
        
        # Compute isotropic elasticity coefficient
        a = 1 / ((1 + nu) * (1 - 2 * nu))
        
        # Construct the 6x6 isotropic material stiffness matrix (C)
        C = a * np.array([
            [1-nu, nu,   nu,   0,           0,           0],
            [nu,   1-nu, nu,   0,           0,           0],
            [nu,   nu,   1-nu, 0,           0,           0],
            [0,    0,    0,    (1-2*nu)/2., 0,           0],
            [0,    0,    0,    0,           (1-2*nu)/2., 0],
            [0,    0,    0,    0,           0,           (1-2*nu)/2.]
        ])
        
        # Determine scaling mode based on type of E
        if isinstance(E, (float, int)):
            D = E * C  # Uniform stiffness
        elif isinstance(E, np.ndarray) and E.shape == (6,):
            D = C * E[np.newaxis, :]  # Element-wise scaling per direction
        else:
            raise ValueError("E must be a float or a NumPy array of shape (6,)")
        
        return D
    
    @abstractmethod
    def calculate_single_point_stress(self, eps: float) -> float:
        """Computes stress for a single strain value using the constitutive model.
        
        This method should be implemented in subclasses to define how a 
        single scalar strain is mapped to stress based on the material model.
        
        Args:
            eps: Strain value (unitless).
            
        Returns:
            Stress value (e.g., in MPa) corresponding to the input strain.
        """
        pass
    
    def calculate_stress(self, eps: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
        """Computes stress from one or multiple strain values.
        
        For scalar input, returns a scalar stress.
        For array-like input (e.g., list or NumPy array), returns stress values
        as a NumPy array. Internally dispatches to `calculate_single_point_stress`.
        
        Strain sign convention:
            - Compressive strain should be negative (ε < 0)
            - Tensile strain should be positive (ε > 0)
            
        Args:
            eps: Strain value or iterable of strain values.
            
        Returns:
            Corresponding stress value(s), scalar or NumPy array.
        
        Raises:
            ValueError: If `eps` is not a numerical value or iterable of numerical values.
        """
        # Check if `eps` is iterable (list, tuple, np.ndarray) but NOT a string
        is_iterable = isinstance(eps, (list, tuple, np.ndarray)) and not isinstance(eps, (str, bytes))
        
        # If `eps` is iterable, compute stress for each value
        if is_iterable:
            stress_results = [self.calculate_single_point_stress(eps=e) for e in eps]
            return np.array(stress_results)
        
        # If `eps` is a single float, compute stress
        if isinstance(eps, (int, float)):
            return self.calculate_single_point_stress(eps=eps)
        
        # Raise error if `eps` is not a valid type
        raise ValueError("`eps` must be a float or an iterable of floats.")
    
    def plot_stress_strain_compression(self, eps_c_max: float, d_eps: float, save_path: str,
                show: bool, close: bool) -> Tuple[plt.Figure, plt.Axes]:
        """Plots the stress-strain curve under compression.
        
        This method should be implemented in subclasses to compute and plot 
        the constitutive response in compression from 0 to `eps_c_max`.
        
        Args:
            eps_c_max: Maximum compressive strain for plotting (unitless).
            d_eps: Strain step size for discretization (unitless).
            save_path: File path to save the plot image.
            show: If True, displays the plot in an interactive window.
            close: If True, closes the figure after saving (prevents memory accumulation).
            
        Returns:
            A tuple of (Figure, Axes) objects representing the plotted curve.
        """
        pass
    
    def plot_stress_strain_tension(self, eps_t_max: float, d_eps: float, save_path: str,
                show: bool, close: bool) -> Tuple[plt.Figure, plt.Axes]:
        """Plots the stress-strain curve under tension.
        
        This method should be implemented in subclasses to compute and plot 
        the constitutive response in tension from 0 to `eps_t_max`.
        
        Args:
            eps_t_max: Maximum tensile strain for plotting (unitless).
            d_eps: Strain step size for discretization (unitless).
            save_path: File path to save the plot image.
            show: If True, displays the plot in an interactive window.
            close: If True, closes the figure after saving (prevents memory accumulation).
            
        Returns:
            A tuple of (Figure, Axes) objects representing the plotted curve.
        """
        pass
    
    def plot_stress_strain(self, eps_c_max: float, eps_t_max: float, d_eps: float, save_path: str,
                show: bool, close: bool) -> Tuple[plt.Figure, plt.Axes]:
        """Plots the full stress-strain response in both compression and tension.
        
        This method should be implemented in subclasses to compute and visualize
        the constitutive response of the material under both compressive and tensile loading.
        
        Strain sign convention:
            - Compressive strain should be negative (ε < 0)
            - Tensile strain should be positive (ε > 0)
            
        Args:
            eps_c_max: Maximum compressive strain for plotting (unitless).
            eps_t_max: Maximum tensile strain for plotting (unitless).
            d_eps: Strain step size for discretization (unitless).
            save_path: Path to save the plot image.
            show: Whether to display the figure interactively.
            close: Whether to close the figure after saving (to free memory).
            
        Returns:
            A tuple of (Figure, Axes) objects representing the combined plot.
        """
        pass
    
    def calculate_residual_tensile_eps(self) -> float:
        """Computes the residual tensile strain corresponding to the residual post-peak
        stress `omega_tu * ft`, using an explicit inverse formula from the 
        softening segment of the five-line model.
        
        Returns:
            float: Residual tensile strain ε_t,res (unitless)
            
        Raises:
            ValueError: If the softening type is not supported.
        """
        pass
    
    def calculate_residual_compressive_eps(self, eps_cu: float) -> float:
        """Computes the residual compressive strain ε_c,res corresponding to the residual
        stress `omega_cu * fc`, using the final softening segment of the five-line
        piecewise compression model.
        
        Args:
            eps_cu (float): Ultimate compressive strain (unitless).
            
        Returns:
            float: Residual compressive strain ε_c,res (unitless).
        """
        pass
    
class Quadrilinear_Concrete(Material):
    """Quadrilinear stress-strain model for concrete.
    
    This class models the stress-strain behavior of concrete using a 
    quadrilinear approximation, considering both compression and tension.
    
    Attributes:
        prop (Concrete_Property): Dataclass containing material properties.
        logger (Logger): Logger for recording model initialization and operations.
    """
    
    def __init__(self, prop: Concrete_Property) -> None:
        """Initializes a Quadrilinear_Concrete material model.
        
        Args:
            prop: The dataclass of material properties.
        """
        super().__init__(prop)
        
        self.logger = setup_logger(self.__class__.__name__, log_dir='log/fem/log')
        self.logger.info("Initialized Quadrilinear_Concrete model with properties: %s", vars(self.prop))
    
    def calculate_single_point_stress(self, eps: float) -> float:
        """Computes the uniaxial stress for a single strain value `eps`.
        
        Implements a quadrilinear stress-strain model for concrete, 
        considering both compression (ε ≤ 0) and tension (ε > 0), with 
        distinct peak, softening, and residual regions.
        
        Strain sign convention:
        - Compressive strain should be negative (ε < 0)
        - Tensile strain should be positive (ε > 0)
        
        Args:
            eps: Strain value (unitless).
            
        Returns:
            Stress value (MPa), positive in tension and negative in compression.
        """
        # Material properties
        E = self.prop.E                       # Elastic modulus (MPa)
        fc = self.prop.fc                    # Compressive strength (MPa)
        omega_ce = self.prop.omega_ce        # Compressive elastic stress ratio
        omega_cu = self.prop.omega_cu        # Compressive residual stress ratio
        eps_ce = self.prop.eps_ce            # Compressive elastic limit
        eps_cp = self.prop.eps_cp            # Compressive peak strain
        eps_cu = self.prop.eps_cu            # Compressive ultimate strain
        
        ft = self.prop.ft                    # Tensile strength (MPa)
        eps_tp = self.prop.eps_tp            # Tensile peak strain
        eps_tu = self.prop.eps_tu            # Tensile ultimate strain
        
        # Compression region (ε ≤ 0)
        if eps <= 0:
            eps_pos = abs(eps)  # Convert strain to positive for calculations
            
            if 0 <= eps_pos <= eps_ce:
                sig = E * eps_pos  # Elastic region
            elif eps_ce < eps_pos <= eps_cp:
                sig_e = omega_ce * fc  # Stress at elastic limit
                sig = sig_e + (fc - sig_e) * (eps_pos - eps_ce) / (eps_cp - eps_ce) # Hardening
            elif eps_cp < eps_pos <= eps_cu:
                sig_u = omega_cu * fc  # Ultimate stress
                sig = fc - (fc - sig_u) * (eps_pos - eps_cp) / (eps_cu - eps_cp) # Softening
            else:
                sig = omega_cu * fc  # Post-failure plateau
            
            return float(-sig) # Compression is negative
        
        # Tension region (ε > 0)
        if 0 < eps <= eps_tp:
            sig = E * eps  # Linear elastic region
        elif eps_tp < eps <= eps_tu:
            sig = ft + (eps - eps_tp) * ft / (eps_tp - eps_tu)  # Softening
        else:  # Beyond ultimate tensile strain
            sig = 0.0
        
        return float(sig)
    
    def plot_stress_strain_compression(self, eps_c_max: float, d_eps: float,
                eps_c_min: float = 0.0,
                save_path: str = 'log/fem/materials/Quadrilinear_Compression.png',
                show: bool = True, close: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """Plots the compressive stress-strain curve of the material model.
        
        This function computes stress over a strain range (compression only)
        and generates a high-resolution matplotlib plot, with optional annotations,
        saving and display options.
        
        Args:
            eps_c_max: Maximum compressive strain to plot (unitless).
            d_eps: Strain step size (unitless).
            eps_c_min: Minimum compressive strain to plot (default is 0.0).
            save_path: File path to save the plot image (default saves to 'log/materials').
            show: If True, displays the plot in an interactive window.
            close: If True, closes the figure after saving (prevents memory accumulation).
            
        Returns:
            A tuple of (Figure, Axes) containing the plot. If `close` is True, the returned
            objects are still valid but should not be shown or modified further.
        """
        # Generate strain and stress values
        eps = np.arange(np.abs(eps_c_min), np.abs(eps_c_max), d_eps)
        sig = -self.calculate_stress(eps=-eps)
        
        # Generate plot
        fig, ax = plot_template(
            [eps, sig],
            colors=['black'],
            xlabel='Compressive Strain (mm/mm)',
            ylabel='Compressive Stress (MPa)',
            xlim=(np.abs(eps_c_min), np.abs(eps_c_max)),
            ylim=(0, 60)
        )
        
        # Annotate key points on the plot
        dpx = 4
        dpy = 1
        annotate_point(self.prop.eps_ce, self.prop.f_ce, fig, ax, decimal_places_x=dpx, decimal_places_y=dpy)
        annotate_point(self.prop.eps_cp, self.prop.fc,   fig, ax, decimal_places_x=dpx, decimal_places_y=dpy, va='bottom')
        annotate_point(self.prop.eps_cu, self.prop.f_cu, fig, ax, decimal_places_x=dpx-1, decimal_places_y=dpy, va='bottom')
        
        # Save the figure
        if save_path:
            fig.savefig(save_path, dpi=300)
        
        # Show the plot if requested
        if show:
            plt.show()
            
        # Close the figure if requested
        if close:
            plt.close(fig)
        
        return fig, ax
    
    def plot_stress_strain_tension(self, eps_t_max: float, d_eps: float,
                eps_t_min: float = 0.0,
                save_path: str = 'log/fem/materials/Quadrilinear_Tension.png',
                show: bool = True, close: bool = True) -> Tuple[plt.Figure | plt.Axes]:
        """Plots the tensile stress-strain curve of the material model.
        
        This function computes tensile stress over a specified strain range 
        and generates a matplotlib plot with optional annotations, saving, and display.
        
        Args:
            eps_t_max: Maximum tensile strain to plot (unitless).
            d_eps: Strain step size (unitless).
            eps_t_min: Minimum tensile strain to plot (default is 0.0).
            save_path: File path to save the plot image.
            show: Whether to display the figure interactively.
            close: Whether to close the figure after saving (recommended in batch use).
            
        Returns:
            A tuple of (Figure, Axes) containing the plot. If `close` is True, the returned
            figure and axes are still valid but should not be shown or reused.
        """
        # Generate strain and stress values
        eps = np.arange(np.abs(eps_t_min), np.abs(eps_t_max), d_eps)
        sig = self.calculate_stress(eps=eps)
        
        # Generate plot
        fig, ax = plot_template(
            [eps, sig],
            colors=['black'],
            xlabel='Tensile Strain (mm/mm)',
            ylabel='Tensile Stress (MPa)',
            xlim=(np.abs(eps_t_min)-0.0002, np.abs(eps_t_max)+0.0002),
            ylim=(-0.1,1.5)
        )
        
        # Annotate key points on the plot
        dpx = 5
        dpy = 2
        annotate_point(self.prop.eps_tp, self.prop.ft, fig, ax, decimal_places_x=dpx, decimal_places_y=dpy, va='bottom')
        annotate_point(self.prop.eps_tu, 0.0, fig, ax, decimal_places_x=dpx, decimal_places_y=dpy, va='bottom')
        
        # Save the figure
        if save_path:
            fig.savefig(save_path, dpi=300)
        
        # Show the plot if requested
        if show:
            plt.show()
            
        # Close the figure if requested
        if close:
            plt.close(fig)
        
        return fig, ax
    
    def plot_stress_strain(self, eps_c_max: float, eps_t_max: float, d_eps: float,
                save_path: str = 'log/fem/materials/Quadrilinear.png',
                show: bool = True, close: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """Plots the full stress-strain curve (compression + tension).
        
        This function computes and plots the stress-strain response from
        `eps_c_max` (negative, compressive strain) to `eps_t_max` (positive, tensile strain),
        including key points of interest for annotation.
        
        Strain sign convention:
            - Compressive strain should be negative (ε < 0)
            - Tensile strain should be positive (ε > 0)
            
        Args:
            eps_c_max: Maximum compressive strain (should be negative).
            eps_t_max: Maximum tensile strain.
            d_eps: Strain step size.
            save_path: Path to save the plot image.
            show: Whether to display the figure interactively.
            close: Whether to close the figure after saving (to free memory).
            
        Returns:
            Tuple of (Figure, Axes) containing the generated plot.
        """
        # Generate strain and stress values
        eps = np.arange(eps_c_max, eps_t_max, d_eps)
        sig = self.calculate_stress(eps=eps)
        
        # Generate plot
        fig, ax = plot_template(
            [eps, sig],
            figsize=(13/2.54, 7/2.54),
            colors=['black'],
            xlabel='Strain (mm/mm)',
            ylabel='Stress (MPa)',
            xlim=(eps_c_max, eps_t_max),
            ylim=(-60, 7),
            major_x_interval=0.005,
            major_y_interval=20,
            xtick_precision=3,
        )
        
        # Add axes lines at origin
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
        
        # Annotate key points on the plot
        dpx = 4
        dpy = 1
        annotate_point(-self.prop.eps_ce, -self.prop.f_ce, fig, ax, decimal_places_x=dpx, decimal_places_y=dpy)
        annotate_point(-self.prop.eps_cp, -self.prop.fc,   fig, ax, decimal_places_x=dpx-1, decimal_places_y=dpy)
        annotate_point(-self.prop.eps_cu, -self.prop.f_cu, fig, ax, decimal_places_x=dpx-1, decimal_places_y=dpy, va='bottom')
        
        annotate_point(self.prop.eps_tp, self.prop.ft, fig, ax, decimal_places_x=dpx+1, decimal_places_y=dpy+1, va='bottom')
        
        ax.scatter([self.prop.eps_tu], [0.0], color=(192/255, 0, 0), s=7, zorder=3)
        ax.text(self.prop.eps_tu, 0.0-2, f" ({self.prop.eps_tu:.5f}, 0)", fontsize=8, color=(0, 47/255, 167/255), va='top')
        
        # Save the figure
        if save_path:
            fig.savefig(save_path, dpi=300)
        
        # Show the plot if requested
        if show:
            plt.show()
            
        # Close the figure if requested
        if close:
            plt.close(fig)
        
        return fig, ax
    
class Five_Line_Concrete(Material):
    """Five-line piecewise linear stress-strain model for concrete.
    
    This class models the uniaxial stress-strain behavior of concrete using 
    a five-segment linear approximation, covering both compression and tension.
    
    Attributes:
        prop (Concrete_Property): Dataclass containing material properties.
        logger (Logger): Logger for recording model initialization and operations.
    """
    
    def __init__(self, prop: Concrete_Property) -> None:
        """Initializes a Five_Line_Concrete material model.
        
        Args:
            prop: The dataclass of material properties.
        """
        super().__init__(prop)
        
        self.logger = setup_logger(self.__class__.__name__, log_dir='log/fem/log')
        self.logger.info("Initialized Five_Line_Concrete model with properties: %s",vars(self.prop))
    
    def calculate_single_point_stress(self, eps: float) -> float:
        """Computes uniaxial stress for a given strain value `eps`.
        
        This function implements a five-segment piecewise linear constitutive 
        model for both compressive (ε < 0) and tensile (ε > 0) behavior.
        
        Strain sign convention:
            - Compressive strain should be negative (ε < 0)
            - Tensile strain should be positive (ε > 0)
            
        Args:
            eps: Strain value (unitless).
            
        Returns:
            Stress σ corresponding to input strain ε (in MPa).
        """
        # Material properties
        E = self.prop.E                      # Elastic modulus (MPa)
        fc = self.prop.fc                    # Compressive strength (MPa)
        omega_ce = self.prop.omega_ce        # Compressive elastic stress ratio
        omega_cu = self.prop.omega_cu        # Compressive residual stress ratio
        eps_ce  = self.prop.eps_ce           # Compressive elastic limit
        eps_cp  = self.prop.eps_cp           # Compressive peak strain
        eps_cp2 = self.prop.eps_cp2          # End of plateau (compression)
        eps_cu  = self.prop.eps_cu           # Compressive ultimate strain
        
        ft = self.prop.ft                    # Tensile strength (MPa)
        eps_te = self.prop.eps_te            # Elastic strain limit (tension)
        eps_tp = self.prop.eps_tp            # Tensile peak strain
        eps_tu = self.prop.eps_tu            # Tensile ultimate strain
        soften = self.prop.soften
        alpha_ct = self.prop.alpha_ct
        
        # Compression region (ε ≤ 0)
        if eps <= 0:
            eps_pos = abs(eps)  # Convert strain to positive for calculations
            
            if 0 <= eps_pos <= eps_ce:
                sig = E * eps_pos  # Elastic region
            elif eps_ce < eps_pos <= eps_cp:
                sig_e = omega_ce * fc  # Stress at elastic limit
                sig = sig_e + (fc - sig_e) * (eps_pos - eps_ce) / (eps_cp - eps_ce) # Hardening
            elif eps_cp < eps_pos <= eps_cp2:
                sig = fc  # Plateau
            elif eps_cp2 < eps_pos <= eps_cu:
                sig_u = omega_cu * fc  # Ultimate stress
                sig = fc - (fc - sig_u) * (eps_pos - eps_cp2) / (eps_cu - eps_cp2) # Softening
            else:
                sig = omega_cu * fc  # Residual post-failure
            
            return float(-sig) # Compression stress is negative
        
        # Tension region (ε > 0)
        if 0 < eps <= eps_te:
            sig = E * eps  # Linear elastic region
        elif eps_te < eps <= eps_tp:
            sig_e = E * eps
            sig = sig_e + (eps - eps_te) * (ft - sig_e) / (eps_tp - eps_te)  # Hardening
        elif eps_tp < eps <= eps_tu:
            if soften.lower() in {'l', 'linear'}:
                sig = ft + (eps - eps_tp) * ft / (eps_tp - eps_tu)  # Softening
            elif soften.lower() in {'e', 'exp', 'exponential'}:
                sig = ft * np.exp(-alpha_ct * (eps - eps_tp) / eps_tp)
            else:
                raise ValueError(f"Invalid softening type '{self.prop.soften}'. Expected 'linear' or 'exponential'.")
        else:  # Beyond ultimate tensile strain
            if soften.lower() in {'l', 'linear'}:
                sig =  0.0
            elif soften.lower() in {'e', 'exp', 'exponential'}:
                sig = ft * np.exp(-alpha_ct * (eps - eps_tp) / eps_tp)
            else:
                raise ValueError(f"Invalid softening type '{self.prop.soften}'. Expected 'linear' or 'exponential'.")
        
        return float(sig)
    
    def calculate_residual_tensile_eps(self) -> float:
        """Computes the residual tensile strain corresponding to the residual post-peak
        stress `omega_tu * ft`, using an explicit inverse formula from the 
        softening segment of the five-line model.
        
        Returns:
            float: Residual tensile strain ε_t,res (unitless)
            
        Raises:
            ValueError: If the softening type is not supported.
        """
        ft = self.prop.ft
        omega_tu = self.prop.omega_tu
        eps_tp = self.prop.eps_tp
        eps_tu = self.prop.eps_tu
        soften = self.prop.soften.lower()
        alpha_ct = self.prop.alpha_ct
        
        sig_t_res = omega_tu * ft
        
        if soften in {'l', 'linear'}:
            slope = -ft / (eps_tu - eps_tp)
            eps_t_res = eps_tp + (sig_t_res - ft) / slope
            
        elif soften in {'e', 'exp', 'exponential'}:
            eps_t_res = eps_tp - (np.log(sig_t_res / ft) * eps_tp) / alpha_ct
            
        else:
            raise ValueError(f"Unsupported softening type: '{self.prop.soften}'. Use 'linear' or 'exponential'.")
        
        return float(eps_t_res)
    
    def calculate_residual_compressive_eps(self, eps_cu: float) -> float:
        """Computes the residual compressive strain ε_c,res corresponding to the residual
        stress `omega_cu * fc`, using the final softening segment of the five-line
        piecewise compression model.
        
        Args:
            eps_cu (float): Ultimate compressive strain (unitless).
            
        Returns:
            float: Residual compressive strain ε_c,res (unitless).
        """
        fc = self.prop.fc
        omega_cu = self.prop.omega_cu
        eps_cp2 = self.prop.eps_cp2
        
        # Residual compressive stress
        sig_c_res = omega_cu * fc
        
        # Linear softening slope between fc and residual stress
        slope = -(fc - sig_c_res) / (eps_cu - eps_cp2)
        
        # Residual strain at stress = sig_c_res
        eps_c_res = eps_cp2 + (sig_c_res - fc) / slope
        
        return float(eps_c_res)
    
    def plot_stress_strain_compression(self, eps_c_max: float, d_eps: float,
                eps_c_min: float = 0.0,
                save_path: str = 'log/fem/materials/Five_Line_Compression.png',
                show: bool = True, close: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """Plots the compressive stress-strain curve of the material model.
        
        This function computes stress over a strain range (compression only)
        and generates a high-resolution matplotlib plot, with optional annotations,
        saving and display options.
        
        Args:
            eps_c_max: Maximum compressive strain to plot (unitless).
            d_eps: Strain step size (unitless).
            eps_c_min: Minimum compressive strain to plot (default is 0.0).
            save_path: File path to save the plot image (default saves to 'log/materials').
            show: If True, displays the plot in an interactive window.
            close: If True, closes the figure after saving (prevents memory accumulation).
            
        Returns:
            A tuple of (Figure, Axes) containing the plot. If `close` is True, the returned
            objects are still valid but should not be shown or modified further.
        """
        # Generate strain and stress values
        eps = np.arange(np.abs(eps_c_min), np.abs(eps_c_max), d_eps)
        sig = -self.calculate_stress(eps=-eps)
        
        # Generate plot
        fig, ax = plot_template(
            [eps, sig],
            colors=['black'],
            xlabel='Compressive Strain (mm/mm)',
            ylabel='Compressive Stress (MPa)',
            xlim=(np.abs(eps_c_min), np.abs(eps_c_max)),
            ylim=(0, 60)
        )
        
        # Annotate key points on the plot
        dpx = 4
        dpy = 1
        annotate_point(self.prop.eps_ce,  self.prop.f_ce, fig, ax, decimal_places_x=dpx, decimal_places_y=dpy)
        annotate_point(self.prop.eps_cp,  self.prop.fc,   fig, ax, decimal_places_x=dpx, decimal_places_y=dpy, va='bottom')
        annotate_point(self.prop.eps_cp2, self.prop.fc,   fig, ax, decimal_places_x=dpx, decimal_places_y=dpy)
        annotate_point(self.prop.eps_cu, self.prop.f_cu,  fig, ax, decimal_places_x=dpx-1, decimal_places_y=dpy, va='bottom')
        
        # Save the figure
        if save_path:
            fig.savefig(save_path, dpi=300)
        
        # Show the plot if requested
        if show:
            plt.show()
            
        # Close the figure if requested
        if close:
            plt.close(fig)
        
        return fig, ax
    
    def plot_stress_strain_tension(self, eps_t_max: float, d_eps: float,
                eps_t_min: float = 0.0,
                save_path: str = 'log/fem/materials/Five_Line_Tension.png',
                show: bool = True, close: bool = True) -> Tuple[plt.Figure | plt.Axes]:
        """Plots the tensile stress-strain curve of the material model.
        
        This function computes tensile stress over a specified strain range 
        and generates a matplotlib plot with optional annotations, saving, and display.
        
        Args:
            eps_t_max: Maximum tensile strain to plot (unitless).
            d_eps: Strain step size (unitless).
            eps_t_min: Minimum tensile strain to plot (default is 0.0).
            save_path: File path to save the plot image.
            show: Whether to display the figure interactively.
            close: Whether to close the figure after saving (recommended in batch use).
            
        Returns:
            A tuple of (Figure, Axes) containing the plot. If `close` is True, the returned
            figure and axes are still valid but should not be shown or reused.
        """
        # Generate strain and stress values
        eps = np.arange(np.abs(eps_t_min), np.abs(eps_t_max), d_eps)
        sig = self.calculate_stress(eps=eps)
        
        # Generate plot
        fig, ax = plot_template(
            [eps, sig],
            colors=['black'],
            xlabel='Tensile Strain (mm/mm)',
            ylabel='Tensile Stress (MPa)',
            xlim=(np.abs(eps_t_min)-0.0002, np.abs(eps_t_max)+0.0002),
            ylim=(-0.1,1.5)
        )
        
        # Annotate key points on the plot
        dpx = 5
        dpy = 1
        annotate_point(self.prop.eps_te, self.prop.f_te, fig, ax, decimal_places_x=dpx, decimal_places_y=dpy, ha='left')
        annotate_point(self.prop.eps_tp, self.prop.ft, fig, ax, decimal_places_x=dpx, decimal_places_y=dpy, va='bottom', ha='left')
        annotate_point(self.prop.eps_tu, 0.0, fig, ax, decimal_places_x=dpx, decimal_places_y=dpy, va='bottom')
        
        # Save the figure
        if save_path:
            fig.savefig(save_path, dpi=300)
        
        # Show the plot if requested
        if show:
            plt.show()
            
        # Close the figure if requested
        if close:
            plt.close(fig)
        
        return fig, ax
    
    def plot_stress_strain(self, eps_c_max: float, eps_t_max: float, d_eps: float,
                save_path: str = 'log/fem/materials/Five_Line.png',
                show: bool = True, close: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """Plots the full stress-strain curve (compression + tension).
        
        This function computes and plots the stress-strain response from
        `eps_c_max` (negative, compressive strain) to `eps_t_max` (positive, tensile strain),
        including key points of interest for annotation.
        
        Strain sign convention:
            - Compressive strain should be negative (ε < 0)
            - Tensile strain should be positive (ε > 0)
            
        Args:
            eps_c_max: Maximum compressive strain (should be negative).
            eps_t_max: Maximum tensile strain.
            d_eps: Strain step size.
            save_path: Path to save the plot image.
            show: Whether to display the figure interactively.
            close: Whether to close the figure after saving (to free memory).
            
        Returns:
            Tuple of (Figure, Axes) containing the generated plot.
        """
        # Generate strain and stress values
        eps = np.arange(eps_c_max, eps_t_max, d_eps)
        sig = self.calculate_stress(eps=eps)
        
        # Generate plot
        fig, ax = plot_template(
            [eps, sig],
            figsize=(13/2.54, 7/2.54),
            colors=['black'],
            xlabel='Strain (mm/mm)',
            ylabel='Stress (MPa)',
            xlim=(eps_c_max, eps_t_max),
            ylim=(-60, 7),
            major_x_interval=0.005,
            major_y_interval=20,
            xtick_precision=3,
        )
        
        # Add axes lines at origin
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
        
        # Annotate key points on the plot
        dpx = 4
        dpy = 1
        annotate_point(-self.prop.eps_ce,  -self.prop.f_ce, fig, ax, decimal_places_x=dpx, decimal_places_y=dpy)
        annotate_point(-self.prop.eps_cp,  -self.prop.fc,   fig, ax, decimal_places_x=dpx, decimal_places_y=dpy)
        annotate_point(-self.prop.eps_cp2, -self.prop.fc,   fig, ax, decimal_places_x=dpx, decimal_places_y=dpy, ha='right')
        annotate_point(-self.prop.eps_cu,  -self.prop.f_cu, fig, ax, decimal_places_x=dpx-1, decimal_places_y=dpy, va='bottom')
        
        annotate_point(self.prop.eps_te, self.prop.f_te, fig, ax, decimal_places_x=dpx+1, decimal_places_y=dpy, va='bottom', ha='right')
        annotate_point(self.prop.eps_tp, self.prop.ft, fig, ax, decimal_places_x=dpx+1, decimal_places_y=dpy, va='bottom')
        
        ax.scatter([self.prop.eps_tu], [0.0], color=(192/255, 0, 0), s=7, zorder=3)
        ax.text(self.prop.eps_tu, 0.0-2, f" ({self.prop.eps_tu:.5f}, 0)", fontsize=8, color=(0, 47/255, 167/255), va='top')
        
        # Save the figure
        if save_path:
            fig.savefig(save_path, dpi=300)
        
        # Show the plot if requested
        if show:
            plt.show()
            
        # Close the figure if requested
        if close:
            plt.close(fig)
        
        return fig, ax
