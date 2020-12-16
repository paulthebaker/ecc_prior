"""
compute centroids of repeated eccentric bursts

Forward evolution from Loutrel & Yunes 2017. Algebraic inversions
for backward evolution by BC.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from numpy.linalg import inv

__author__ = "Belinda Cheeseboro, Paul T. Baker"
#__copyright__ = ""
#__license__ = ""
__version__ = "2020-08-24"

import numpy as np

class EccBurstNew(object):
    _min_de = 1.0e-3
    _max_de = 0.9
    

    def __init__(self, q):
        """calculate t,f of eccentric bursts
        Always work in units of total mass!! (Mtot=1)

        :param q: mass ratio of bursts
        """
        self._q = q
        self._Mchirp = q**(3/5)/(1+q)**(6/5)

        # Define constants
        self._A = 59/24 * np.pi*np.sqrt(2) * (self._Mchirp)**(5/3)
        self._B = 121/236
        self._C = 85/12 * np.pi*np.sqrt(2) * (self._Mchirp)**(5/3)
        self._D = 1718/1800

    @property
    def Mchirp(self):
        """binary chirp mass in units of total mass"""
        return self._Mchirp

    @property
    def q(self):
        """binary mass ratio"""
        return self._q

    @q.setter
    def q(self, value):
        if value <= 0 or value > 1.0:
            raise ValueError("q = {}.  Mass ratio defined as: 0 < q <= 1".format(value))
            
        self._q = value
        q = self._q
        self._Mchirp = q**(3/5)/(1+q)**(6/5)
        old_Mc = self._Mchirp
        self._Mchirp = (self._q/(1+self._q)**2)**(3/5)

        # recompute A,C constants
        self._A = 59/24 * np.pi*np.sqrt(2) * (self._Mchirp)**(5/3)
        self._C = 85/12 * np.pi*np.sqrt(2) * (self._Mchirp)**(5/3)
        
        self._A *= (self._Mchirp/old_Mc)**(5/3)
        self._C *= (self._Mchirp/old_Mc)**(5/3)

    def _rp_kepler(self, de, f):
        """Compute pericenter separation rp for a keplerian orbit corresponding
        to a burst in units of total mass.

        :param de: instantaneous eccentricity of current burst (de = 1 - e)
        :param f: instantaneous GW frequency of burst
        """
        return ((2-de)/(2*np.pi*f)**2)**(1/3)

    def re_valid(self, r, de):
        """check that r and de are in the region of validity for assumptions
        M/r << 1 and de << 1
        :param r: periastron distance
        :param de: eccentricity (de = 1 - e)
        """
        return 1/r < 0.5 and de < 0.5

    def tfe_forward(self, t0, f0, de0):
        """calculate the time, freq, and ecc of next burst
       
        directly calculate using new 1st order equations, bypassing rp used
        by Loutrel & Yunes.

        :param t0: time of current burst
        :param f0: freq of current burst
        :param de0: eccentricity of current burst (de = 1-e)
        :return: tuple (t1, f1, de1), params for next burst
        """
        Po223 = np.pi/2**(2/3)
        PfM53 = (np.pi * f0 * self._Mchirp)**(5/3)

        t1 = t0 + np.sqrt((2-de0) / de0**3) / f0
        f1 = f0 * (1 + 23/3 * Po223*PfM53 *
                   (1 + 7547/4140*de0 + 3725/552*Po223*PfM53))
        de1 = de0 + 85/3 * Po223*PfM53 * (1 - 121/225*de0)

        return (t1, f1, de1)

    def tfe_backward(self, t1, f1, de1):
        """calculate the time, freq, and ecc of previous burst
       
        directly calculate using new 1st order equations, bypassing rp used
        by Loutrel & Yunes.

        :param t1: time of current burst
        :param f1: freq of current burst
        :param de1: eccentricity of current burst (de = 1-e)
        :return: tuple (t0, f0, de0), params for previous burst
        """
        Po223 = np.pi/2**(2/3)
        PfM53 = (np.pi * f1 * self._Mchirp)**(5/3)

        t0 = t1 - np.sqrt((2-de1) / de1**3) / f1
        f0 = f1 * (1 - 23/3 * Po223*PfM53
                   * (1 + 7547/4140*de1 + 5875/368*Po223*PfM53))
        de0 = de1 - (85/3 * Po223*PfM53 
                     * (1 - 121/225*de1 - 295/12*Po223*PfM53))

        return (t0, f0, de0)

    def get_all_bursts(self, tstar, fstar, destar, tmin, tmax):
        """get all bursts in time window from start to ISCO
        include one post ISCO burst if it fits in time window
        never include bursts outside of time window
        """
        min_fM = 4.92549094830932e-05
        bursts = [[tstar, fstar, destar]]
        rpstar = self._rp_kepler(destar, fstar)

        # get forward bursts
        t, f, de, rp = tstar, fstar, destar, rpstar
        while rp > 3 and t < tmax:
            t, f, de = self.tfe_forward(t, f, de)
            rp = self._rp_kepler(de, f)
            if(t<tmax) and de<1: bursts.append([t, f, de])

        # get backward busrts
        t, f, rp, de = tstar, fstar, rpstar, destar
        while t > tmin and de > 0:
            t, f, de = self.tfe_backward(t, f, de)
            rp = self._rp_kepler(de, f)
            if(t>tmin) and de>0 and (f>0): bursts.insert(0, [t, f, de]) # prepend
        return bursts
