# test/pc_fem/test_utils.py

import numpy as np

from pc_fem.utils.utils import calculate_equivalent_strain_from_voigt


def test_equivalent_strain() -> None:
    eps = np.array([0.000751, 0.000751, -0.00312, 0.00047, -0.00121, -0.00121])
    eq_function = calculate_equivalent_strain_from_voigt(vec=eps)
    
    eps_tensor = np.array([
        [eps[0], 0.5*eps[3], 0.5*eps[5]],
        [0.5*eps[3], eps[1], 0.5*eps[4]],
        [0.5*eps[5], 0.5*eps[4], eps[2]]
    ])
    
    eps_p, _ = np.linalg.eigh(eps_tensor)
    print('Principal strains eps_p =', eps_p)
    
    eps_p_pos = np.maximum(eps_p, 0.0)
    print('Tensile principal strain eps_p_pos =', eps_p_pos)
    
    eps_eq = np.linalg.norm(eps_p_pos)
    print('Equivalent strain eps_eq =', eps_eq)
    
    print('Equivalent strain eps_eq =', eq_function, 'using utils function')

def calculate_eps_eq() -> None:
    eps = np.loadtxt('log/kfu/elastic_eps_voigt.csv', delimiter=',')
    eps_eq = [calculate_equivalent_strain_from_voigt(vec=ep) for ep in eps]
    np.savetxt('log/kfu/elastic_eps_eq.csv', np.array(eps_eq), delimiter=',')


if __name__ == '__main__':
    test_equivalent_strain()
    calculate_eps_eq()
    pass