# pc_fem/config_loader.py


"""Configuration loader for FEM material parameters.

This module loads concrete, damage, geometry, and contact parameter sets from a YAML file
and returns them as strongly typed data class instances.
"""

import yaml

from typing import Tuple


from pc_fem.parameters import Concrete_Property, Damage_Parameter, Geometry_Parameter, Mesh_Parameter, Contact_Parameter


def load_config(path: str = "configs/config.yaml") -> Tuple[
    Concrete_Property, Damage_Parameter, Geometry_Parameter, Contact_Parameter
]:
    """Loads FEM material parameters from a YAML config file.

    Args:
        path: Path to the YAML configuration file. Default is "config.yaml".

    Returns:
        A tuple of:
            - Concrete_Property: Material parameters for concrete
            - Damage_Parameter: Parameters for concrete damage modeling
            - Geometry_Parameter: Geometry parameters for 3D printed concrete 
            - Mesh_Parameter: Mesh parameters for 3D printed concrete
            - Contact_Parameter: Parameters for contact interaction
    """
    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    concrete = Concrete_Property(**config["concrete"])
    damage = Damage_Parameter(**config["damage"])
    geometry = Geometry_Parameter(**config["geometry"])
    mesh = Mesh_Parameter(**config["mesh"])
    contact = Contact_Parameter(**config["contact"])

    return concrete, damage, geometry, mesh, contact
