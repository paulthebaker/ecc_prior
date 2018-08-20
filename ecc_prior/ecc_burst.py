"""
compute centroids of repeated eccentric bursts

Forward evolution from Loutrel & Yunes 2017. Algebraic inversions
for backward evolution by BC.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Belinda Cheeseboro, Paul T. Baker"
#__copyright__ = ""
#__license__ = ""
__version__ = "2018-08-15"

import numpy as np

class EccBurst(object):
    _min_de = 1.0e-3
    _max_de = 0.9

    def __init__(self, q):
        """calculate t,f of eccentric bursts
        Always work in units of total mass!! (Mtot=1)

        :param q: mass ratio of bursts
        """
        self.q = q
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
            raise ValueError("q = {}.  Mass ratio defined as: 0 < q <= 1"
                             .format(value))
        self._q = value
        old_Mc = self._Mchirp
        self._Mchirp = (self._q/(1+self._q)**2)**(3/5)

        # recompute A,C constants
        self._A *= (self._Mchirp/old_Mc)**(5/3)
        self._C *= (self._Mchirp/old_Mc)**(5/3)

    def re_forward(self, r0, de0):
        """calculate r and de of next burst
        from Loutrel & Yunes 2017
        :param r0:
            periastron distance of current burst
        :param de0:
            instantaneous eccentricity of current burst
            de = 1 - e
        :return: tuple (r1, de1), next burst r and de
        """
        A, B, C, D = self._A, self._B, self._C, self._D

        r1 = r0*(1 - A * (1/r0)**(5/2) * (1 + B*de0))
        de1 = de0 + C * (1/r0)**(5/2) * (1 - D*de0)

        if de1 > self._max_de:
            #print("de WARNING: de = {:.3e}, setting de = 1"
            #      .format(de1))
            de1 = self._max_de

        return r1, de1

    def re_backward(self, r1, de1):
        """calculate r and de of previous burst
        from Cheeseboro
        :param r1:
            periastron distance of current burst
        :param de1:
            instantaneous eccentricity of current burst
            de = 1 - e
        :return: tuple (r0, de0), previous burst r and de
        """
        A, B, C, D = self._A, self._B, self._C, self._D

        r0 = r1 * (1 + A*(1/r1)**(5/2) * (1 + B*de1))
        de0 = de1 * (1 + C*D * (1/r1)**(5/2)) - C*(1/r1)**(5/2)

        if de0 < self._min_de:
            #print("de WARNING: de = {:.3e}, setting de = {:.3e}"
            #      .format(de0, self._min_de))
            de0 = self._min_de

        return r0, de0

    def re_valid(self, r, de):
        """check that r and de are in the region of validity for assumptions
        M/r << 1 and de << 1
        :param r: periastron distance
        :param de: eccentricity (de = 1 - e)
        """
        return 1/r < 0.5 and de < 0.5

    def tf_forward(self, t0, f0, r0, de0, re=True):
        """determine the time and freq next burst using the time and freq
        of the current burst
        :param t0: time of current burst
        :param f0: freq of current burst
        :param r0: periastron distance of current burst
        :param de0: eccentricity of current burst (de = 1 - e)
        :param re: flag. if True, returns r1 and de1 too
        """
        r1, de1 = self.re_forward(r0, de0)

        t1 = t0 + (1/f0) * np.sqrt((2 - de0)/de0**3)
        f1 = f0 * np.sqrt((2 - de1)/(2 - de0) * (r0/r1)**3)
        f1 = np.sqrt((2-de1)/r1**3) / (2*np.pi)

        if re:
            return t1, f1, r1, de1
        else:
            return t1, f1

    def tf_backward(self, t1, f1, r1, de1, re=True):
        """determine the time and freq next burst using the time and freq
        of the current burst
        :param t1: time of current burst
        :param f1: freq of current burst
        :param r1: periastron distance of current burst
        :param de1: eccentricity of current burst (de = 1 - e)
        :param re: flag. if True, returns r0 and de0 too
        """
        r0, de0 = self.re_backward(r1, de1)

        t0 = t1 - (1/f1) * np.sqrt((2 - de1)/de1**3)
        f0 = f1 * np.sqrt((2 - de0)/(2 - de1) * (r1/r0)**3)
        f0 = np.sqrt((2-de0)/r0**3) / (2*np.pi)

        if re:
            return t0, f0, r0, de0
        else:
            return t0, f0

    def get_all_bursts(self, tstar, fstar, destar, tmin, tmax):
        """get all bursts in time window from start to ISCO
        include one post ISCO burst if it fits in time window
        never include bursts outside of time window
        """
        bursts = [[tstar, fstar]]
        rpstar =  ((2-destar)/(2*np.pi*fstar)**2)**(1/3)

        # get forward bursts
        t, f, rp, de = tstar, fstar, rpstar, destar
        while rp > 3 and t < tmax:
            t, f, rp, de = self.tf_forward(t, f, rp, de)
            if(t<tmax): bursts.append([t, f])

        # get backward busrts
        t, f, rp, de = tstar, fstar, rpstar, destar
        while t > tmin:
            t, f, rp, de = self.tf_backward(t, f, rp, de)
            if(t>tmin): bursts.insert(0, [t, f]) # prepend

        return bursts

    def get_cov(f, rho=0):
        """get covariance matrix for ellipsoid in t-f plane

        The Fourier uncertainty principle says that dt*df <= 1/(4pi).
        Each burst is highly localized in time and covers a large
        bandwidth. The GWs are emitted primarily at pericenter with a
        characteristic timescale of T ~ 1/(2pi f).  The bandwidth is
        then ~f/2.

        :param f: GW freq of this burst
        :param rho: correlation coefficient
        """
        dF = f / 2
        dT = 1 / (2*np.pi*f)
        cov = np.array([[dT**2, dT*dF*rho], [dT*dF*rho, dF**2]])
