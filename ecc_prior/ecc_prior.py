"""
compute the eccentric bursts prior
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Paul T. Baker"
#__copyright__ = ""
#__license__ = ""
__version__ = "2018-08-13"

import numpy as np
from scipy.special import logsumexp

from .ecc_burst import EccBurst

_GMsun = 1.32712440018e20  # m^3/s^2
_c = 299792458 # m/s
_Tsun = _GMsun / _c**3

def _q_from_Mc(Mc, Mtot, convention='neg'):
    """compute mass ratio from chirp mass and total mass
    units don't matter as long as Mc and Mtot have same units

    :param Mc: chirp mass
    :param Mtot: total mass
    :param convention: string flag: 'neg' or 'pos'
        which root to use, defines q as less than 1 or greater than 1.

        negative root: 0 < q <= 1
        positive root: q >= 1
    """
    eta = (Mc/Mtot)**(5/3)
    if 'neg':
        q = ((1-2*eta) - np.sqrt(1-4*eta))/(2*eta)
    elif 'pos':
        q = ((1-2*eta) + np.sqrt(1-4*eta))/(2*eta)
    else:
        msg = "convention='{}', must be 'neg' or 'pos'"
        raise ValueError(msg.format(convention))
    return q

def _rp_kepler(de, f, M):
    """Compute pericenter separation rp for a keplerian orbit corresponding
    to a burst in units of total mass.

    :param de: instantaneous eccentricity of current burst (de = 1 - e)
    :param f: instantaneous GW frequency of burst
    :param M: total mass of system
    """
    return ((2-de)/(2*np.pi*f*M*_Tsun)**2)**(1/3)


class Prior(object):
    """An object to calculate the eccentric binary prior
    """
    # Mc <= (1/4)^(3/5) * Mtot (max eta==1/4)
    _maxfac = 0.25**(3/5)

    def __init__(self, tmin=-2, tmax=2):
        """initialize a Prior object
        
        :param tmin: min time to search for bursts [sec]
        :param tmax: max time to search for bursts [sec]
        """
        # initialize EccBurst with dummy q
        self._eb = EccBurst(q=1)
        self._tmin = tmin
        self._tmax = tmax

    def get_logprior(self, tf_from_BW, Mtot, Mc, destar, tstar, fstar):
        """Log of prior probability for wavelets in BayesWave model
        given astrophysically motivated meta-parameters. This is not
        normalized correctly for variable number of wavelets.

        :param tf_from_BW: list of tuples; t,f for each wavelet
        :param Mtot: total mass of binary [Msun]
        :param Mc: chirp mass of binary [Msun]
        :param destar: 1-e of reference burst
        :param tstar: central time of reference burst [sec]
        :param fstar: central frequency of reference burst [Hz]
        :return: float, relative log probability
        """
        # enforce priors on meta-params
        if (Mc <= 0.0 or Mtot <= 0.0 or Mc > self._maxfac*Mtot or
            destar < self._eb._min_de or destar > self._eb._max_de or
            tstar < self._tmin or tstar > self._tmax or
            fstar <= 0):
            return -np.inf
        
        this_q = _q_from_Mc(Mc, Mtot)
        if this_q <= 0 or this_q > 1:
            return -np.inf

        this_rp = _rp_kepler(destar, fstar, Mtot)
        if this_rp <= 2:
            return -np.inf

        self._eb.q = this_q

        # convert ts and fstar to units of total mass
        M2sec = Mtot * _Tsun
        tstar /= M2sec
        fstar *= M2sec
        tmin = self._tmin / M2sec
        tmax = self._tmax / M2sec

        # get bursts convert to SI w/ this Mtot
        tf_prior_M = self._eb.get_all_bursts(tstar, fstar, destar,
                                             tmin=tmin, tmax=tmax)
        tf_prior_SI = np.array([[t*M2sec, f/M2sec] for t,f in tf_prior_M])

        #TODO compute these from uncert!!!
        sigTs, sigFs = np.array([[1/f, f/6] for t,f in tf_prior_SI]).T
        # fix rho to zero for now
        rhos = np.hstack([np.ones(len(tf_prior_SI))*0.0, [0]])
        covs = [[[dT**2, dT*dF*rho], [dT*dF*rho, dF**2]]
                  for dT,dF,rho in zip(sigTs, sigFs, rhos)]

        # All have same norm because the f's cancel out in the determinate
        icovs = [np.linalg.inv(c) for c in covs]
        norms = [1/(2*np.pi * np.sqrt(np.linalg.det(c))) for c in covs]

        norm_N = -np.log(len(tf_prior_SI)) # normalize for number of blobs
        log_prob = 0
        for this_tf in tf_from_BW:
            rs = this_tf - tf_prior_SI
            args = [-0.5 * np.einsum('i,ij,j', r, ic, r)
                    for r,ic in zip(rs, icovs)]
            log_prob += norm_N + logsumexp(a=args)#, b=norms)
        return log_prob


    def get_prior(self, tf_from_BW, Mtot, Mc, destar, tstar, fstar):
        """Prior probability for wavelets in BayesWave model given
        astrophysically motivated meta-parameters. This is not 
        normalized correctly for variable number of wavelets.

        :param tf_from_BW: list of tuples; t,f for each wavelet
        :param Mtot: total mass of binary [Msun]
        :param Mc: chirp mass of binary [Msun]
        :param destar: 1-e of reference burst
        :param tstar: central time of reference burst [sec]
        :param fstar: central frequency of reference burst [Hz]
        :return: float, relative probability
        """
        logP = self.get_logprior(tf_from_BW, Mtot, Mc, destar, tstar, fstar)
        return np.exp(logP)
