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
__version__ = "2018-01-17"

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

    def de_forward(self, f0, de0):
        """calculate r and de of next burst
        from Loutrel & Yunes 2017
        :param r0:
            periastron distance of current burst
        :param de0:
            instantaneous eccentricity of current burst
            de = 1 - e
        :return: tuple (r1, de1), next burst r and de
        """
        Mc = self._Mchirp
        de1 = de0 + ((85*np.pi)/(2**(2/3)*3))*(np.pi*f0*Mc)**(5/3)*(1 + (121/225)*de0)

        if de1 > self._max_de:
            #print("de WARNING: de = {:.3e}, setting de = 1"
            #      .format(de1))
            de1 = self._max_de

        return de1

    def de_backward(self, f1, de1):
        """calculate r and de of previous burst
        from Cheeseboro
        :param r1:
            periastron distance of current burst
        :param de1:
            instantaneous eccentricity of current burst
            de = 1 - e
        :return: tuple (r0, de0), previous burst r and de
        """
        Mc = self._Mchirp
        de0 = de1 - ((85*np.pi*(np.pi*f1*Mc)**(5/3))/(3*2**(2/3))) + (2057*de1*(f1*Mc)**(5/3)*np.pi**(8/3))/(135*2**(2/3))

        if de0 < self._min_de:
            #print("de WARNING: de = {:.3e}, setting de = {:.3e}"
            #      .format(de0, self._min_de))
            de0 = self._min_de

        return de0

    def re_valid(self, r, de):
        """check that r and de are in the region of validity for assumptions
        M/r << 1 and de << 1
        :param r: periastron distance
        :param de: eccentricity (de = 1 - e)
        """
        return 1/r < 0.5 and de < 0.5

    def tf_forward(self, t0, f0, de0, re=True):
        """determine the time and freq next burst using the time and freq
        of the current burst
        :param t0: time of current burst
        :param f0: freq of current burst
        :param r0: periastron distance of current burst
        :param de0: eccentricity of current burst (de = 1 - e)
        :param re: flag. if True, returns r1 and de1 too
        """
        de1 = self.de_forward(f0, de0)
        Mc = self._Mchirp
        
        t1 = t0 + (1/f0*np.sqrt((2 - de0)/de0**3))*(1 - (85*np.pi/(2**(2/3)*2)*(np.pi*f0*Mc)**(5/3)/de0*(1 + 59/170)*de0))
        f1 = f0*(1 + ((23*np.pi)/(2**(2/3)*3))*(np.pi*f0*Mc)**(5/3)*(1 + (7547/4140)*de0))

        if re:
            return t1, f1, de1
        else:
            return t1, f1

    def tf_backward(self, t1, f1, de1, re=True):
        """determine the time and freq next burst using the time and freq
        of the current burst
        :param t1: time of current burst
        :param f1: freq of current burst
        :param r1: periastron distance of current burst
        :param de1: eccentricity of current burst (de = 1 - e)
        :param re: flag. if True, returns r0 and de0 too
        """
        de0 = self.de_backward(f1, de1)
        Mc = self._Mchirp

        t0 = t1 - (1/f1)*np.sqrt((2-de1)/de1**3)
        f0 = f1*(1-((np.pi*(np.pi*f1*Mc)**(5/3))/(3*2**(2/3)))*(23+de1*(7547/180)))

        if re:
            return t0, f0, de0
        else:
            return t0, f0

    def get_all_bursts(self, tstar, fstar, destar, tmin, tmax):
        """get all bursts in time window from start to ISCO
        include one post ISCO burst if it fits in time window
        never include bursts outside of time window
        """
        
        bursts = [[tstar, fstar, destar]]
        rpstar =  ((2-destar)/(2*np.pi*fstar)**2)**(1/3)
        
        # get forward bursts
        t, f, rp, de = tstar, fstar, rpstar, destar
        while rp > 3 and t < tmax:
            t, f, de = self.tf_forward(t, f, de)
            rp = ((2-de)/(2*np.pi*f)**2)**(1/3)
            if(t<tmax):
                bursts.append([t, f, de]) #append to bursts list

        # get backward busrts
        t, f, rp, de = tstar, fstar, rpstar, destar
        while t > tmin:
            t, f, de = self.tf_backward(t, f, de)
            if(t > tmin):
                bursts.insert(0, [t, f, de]) # prepend to bursts list
                
        return bursts
