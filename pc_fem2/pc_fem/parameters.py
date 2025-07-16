# pc_fem/parameters.py


import numpy as np

from typing import Optional, List
from dataclasses import dataclass


@dataclass
class Concrete_Property:
    E: float
    E_pinn: float
    nu: float
    
    fc: float
    f_ce: Optional[float]
    f_cu: Optional[float]
    omega_ce: float
    omega_cu: float
    eps_ce: Optional[float]
    eps_0c: Optional[float]
    eps_cp: float
    eps_cp2: float
    eps_cu: float

    ft: Optional[float]
    f_te: float
    omega_te: float
    omega_tu: float
    eps_te: Optional[float]
    eps_tp: Optional[float]
    eps_tu: Optional[float]
    Ets: float
    soften: str
    alpha_ct: float
    
    def __post_init__(self):
        if self.f_ce is None:
            self.f_ce = self.omega_ce * self.fc
        if self.f_cu is None:
            self.f_cu = self.omega_cu * self.fc
        if self.eps_ce is None:
            self.eps_ce = self.omega_ce * self.fc / self.E
        if self.eps_0c is None:
            self.eps_0c = self.eps_ce
        
        if self.ft is None:
            self.ft = 0.6228 * np.sqrt(self.fc)
        if self.f_te is None:
            self.f_te = self.omega_te * self.ft
        if self.eps_te is None:
            self.eps_te = self.omega_te * self.ft / self.E
        if self.eps_tp is None:
            self.eps_tp = self.ft / self.E
        if self.eps_tu is None:
            self.eps_tu = self.eps_tp + self.ft / (np.abs(self.Ets) * self.E)
    
@dataclass
class Damage_Parameter:
    At: float
    Bt: float
    Ac: float
    Bc: float
    ib: float
    
    control: str
    
    k: float
    
    G_ft: float
    G_fc: float
    model_type: str

@dataclass
class Contact_Parameter:
    epsilon: List[float]
    mu: float

@dataclass
class Geometry_Parameter:
    lx: float
    ly: float
    lz: float
    n_layers: int
    ndof: int

@dataclass
class Mesh_Parameter:
    mesh_size: List[float]