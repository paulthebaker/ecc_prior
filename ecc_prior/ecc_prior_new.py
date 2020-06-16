"""
compute the eccentric bursts prior
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Paul T. Baker, Belinda D. Cheeseboro"
#__copyright__ = ""
#__license__ = ""
__version__ = "2019-04-24"

import numpy as np
from scipy.special import logsumexp

from ecc_burst_new import EccBurstNew

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


class NewPrior(object):
    """An object to calculate the eccentric binary prior
    """
    # Mc <= (1/4)^(3/5) * Mtot (max eta==1/4)
    _maxfac = 0.25**(3/5)

    def __init__(self, tmin, tmax):
        """initialize a Prior object

        :param tmin: min time to search for bursts [sec]
        :param tmax: max time to search for bursts [sec]
        """
        # initialize EccBurst with dummy q
        self._eb = EccBurstNew(q=1)
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
        if (Mc <= 0.0):
            return -np.inf
        if (Mc > self._maxfac*Mtot):
            return -np.inf
        if (Mtot <= 0.0):
            return -np.inf
        #if (destar < self._eb._min_de):
        #    return -np.inf
        #if (destar > self._eb._max_de):
        #    return -np.inf
        if (tstar < self._tmin):
            return -np.inf
        if (tstar > 0.0):
            return -np.inf
        if (fstar <= 0):
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
        tstarM = tstar/M2sec
        fstarM = fstar*M2sec
        tminM = self._tmin / M2sec
        tmaxM = self._tmax / M2sec
        
        
        # get bursts convert to SI w/ this Mtot
        tf_prior_M = self._eb.get_all_bursts(tstarM, fstarM, destar,
                                             tmin=tminM, tmax=tmaxM)
        tf_prior = np.array([[t*M2sec, f/M2sec, de] for t,f, de in tf_prior_M])
        ts, fs, des = tf_prior.T
        tf_prior_SI = np.array([[t, f] for t,f in zip(ts, fs)])

        #Create anchor covariance matrix
        dt0 = 1/fstar
        df0 = fstar/(2*np.pi)
        dde0 = destar/10
        anchor_cov = [[dt0**2,0,0],[0,df0**2,0], [0,0,dde0**2]]
        
 
        #Calculate inverse covariance matrices for bursts
        covs, icovs = self.get_all_icovs(tstarM, fstarM, destar, tminM, tmaxM, anchor_cov, Mtot, Mc)
        norms = [np.sqrt(np.linalg.det(c)) / (2*np.pi) for c in icovs] #normalization factor for each burst
        
        norm_N = -np.log(len(tf_prior_SI)) # normalize for number of blobs
        log_prob = 0
        for ii,this_tf in enumerate(tf_from_BW):
            rs = this_tf - tf_prior_SI
            args = [-0.5 * np.einsum('i,ij,j', r, ic, r)
                    for r,ic in zip(rs, icovs)]
            log_prob += norm_N + logsumexp(a=args, b=norms)
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
    
    def get_jfor(self, f0, de0, Mchirp):
        '''get_jfor returns the jacobian matrix for the forward moving equations.
        These quantities are calculated in SI units.
        Mc: chirp mass
        f0: previous burst frequency
        de0: previous burst eccentricity'''
        
        Mc = Mchirp*_Tsun #chirp mass in seconds
        #t components
        tt = 1
        tf = -(np.sqrt(((2 - de0)/de0**3))/f0**2)
        tde = (-3 + de0)/(np.sqrt(-(-2 + de0)*de0**5)*f0)
    
        #f components
        PfMc53 = (np.pi*f0*Mc)**(5/3)
        ft = 0
        ff = 1 + ((4140 + 7547*de0)*np.pi*PfMc53)/(540*2**(2/3)) + (3725*np.pi**2*PfMc53**2)/(144*2**(1/3))
        fde = (7547*f0*np.pi*PfMc53)/(540*2**(2/3))
    
        #de components
        de_t = 0
        de_f = (17*(225+121*de0)*Mc*(f0*Mc)**(2/3)*np.pi**(8/3))/(81*2**(2/3))
        de_de = 1 + (2057*PfMc53*np.pi)/(135*2**(2/3))
        
        J_for = np.array([[tt, tf, tde],[ft, ff, fde], [de_t, de_f, de_de]])
        return J_for

    def get_jback(self, f1, de1, Mchirp):
        '''get_jback returns the jacobian matrix for the backward moving equations.
        These quantities are calculated in SI units.
        Mc: chirp mass
        f1: current burst frequency
        de1: current burst eccentricity
        '''
        Mc = Mchirp*_Tsun #chirp mass in seconds
        
        #t components
        tt = 1
        tf = (np.sqrt(((2 - de1)/de1**3))/f1**2)
        tde = (3-de1)/(np.sqrt(-(-2+de1)*de1**5)*f1)
        
        #f components
        PfMc53 = (np.pi*f1*Mc)**(5/3)
        ft = 0
        ff = 1-(1/405)*((4140 + 7547*de1)*np.pi*PfMc53)+(3725*np.pi**2*PfMc53**2)/(144*2**(1/3))
        fde = -(7547*f1*np.pi*PfMc53)/(540*2**(2/3))
    
        #de components
        de_t = 0
        de_f = -((425*Mc*(f1*Mc)**(2/3)*np.pi**(8/3))/(9*2**(2/3)))
        de_de = 1 + (2057*PfMc53*np.pi)/(135*2**(2/3))
        
        J_back = np.array([[tt, tf, tde],[ft, ff, fde], [de_t, de_f, de_de]])
        return J_back
    
    def get_all_icovs(self, tstar, fstar, destar, tmin, tmax, anchor_cov, Mtot, Mchirp):
        """gets all the inverse covariance matrices for the corresponding bursts.
           These quantities are in SI units.
       
           tstar: anchor burst time in total mass units
           fstar: anchor burst central frequency in total mass units
           destar: anchor burst eccentricity
           tmin: starting time window in mass units
           tmax: ending time window in mass units
           anchor_cov: covariance matrix for anchor burst
           Mtot: total mass in solar masses
           Mchirp: chirp mass of the binary
        """ 
        
        M2sec = Mtot * _Tsun
        covs = [anchor_cov.copy()]
        rpstar =  ((2-destar)/(2*np.pi*fstar)**2)**(1/3)
        
        # get forward icovs
        t, f, rp, de = tstar, fstar, rpstar, destar
        jj = 0

        while rp > 3 and t < tmax:
            t, f, de = self._eb.tfe_forward(t, f, de)
            rp = ((2-de)/(2*np.pi*f)**2)**(1/3)
            if(t<tmax and de < 1):
                t, f = t*M2sec, f/M2sec #convert to SI units
                jfor = self.get_jfor(f,de, Mchirp)#jacobian matrix for the forward direction
                cov = jfor@covs[jj]@jfor.T#calculating the covariance matrix
                covs.append(cov)#append to covs array
                t, f = t/M2sec, f*M2sec #convert to total mass units
                jj+=1

        # get backward icovs
        t, f, de = tstar, fstar, destar
        while t > tmin:
            t, f, de = self._eb.tfe_backward(t, f, de)
            if(t > tmin and de > 0):
                t, f = t*M2sec, f/M2sec #convert to SI units
                jback = self.get_jback(f,de, Mchirp) #jacobian matrix for the backward direction
                cov = jback@covs[0]@jback.T #calculating the covariance matrix
                covs.insert(0, cov) #prepend to covs array
                t, f = t/M2sec, f*M2sec #convert to total mass units
                
    
        #Find the inverse covariance matrices
        icovs = np.linalg.pinv(covs)
    
        #Convert to 2x2 matrices
        icovs = np.array(icovs)[:,0:2,0:2]
    
        #return np.array(covs), icovs
        return np.array(covs), icovs
