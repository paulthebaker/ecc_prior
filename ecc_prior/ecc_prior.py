"""
compute the eccentric bursts prior
"""
__author__ = "Paul T. Baker"
#__copyright__ = ""
#__license__ = ""
__version__ = "2018-05-03"
#__maintainer__ = ""
#__email__ = ""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np

from .ecc_burst import EccBurst

_GMsun = 1.32712440018e20  # m^3/s^2
_c = 299792458 # m/s

_Rsun = _GMsun / _c**2
_Tsun = _GMsun / _c**3

def bigauss(x, x0=np.array([0,0]), cov=np.diag([1,1])):
    """bivariate Gaussian
    """
    x = np.asarray(x)
    x0 = np.asarray(x0)
    cov = np.asarray(cov)
    icov = np.linalg.inv(cov)
    dx = x-x0
    
    #norm = 1/np.sqrt(2*np.pi * np.linalg.det(cov))
    arg = -0.5 * np.einsum('i,ij,j', dx, icov, dx)
    return np.exp(arg)  # don't normalize
    #return norm * np.exp(arg)
