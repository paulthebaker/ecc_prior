#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Belinda D. Cheeseboro"
__version__ = "2020-11-11"
'''This code generates a set of bursts for these  given parameters:
	Mtot: total mass of the binary [solar masses]
	Mchirp: chirp mass of the binary [solar masses]
	q: mass ratio
	destar: eccentricity of the anchor burst
	tstar: central time of the anchor burst [sec]
	fstar: central frequency of the anchor burst [sec^-1]
	tmin: start of time window [sec]
	tmax: end of time window [sec]'''

import sys
import numpy as np
import matplotlib.pyplot as plt
import kombine as kb
import arviz as az
from corner import corner
from ecc_burst_new import EccBurstNew
from ecc_prior_new import NewPrior
import json
import os
import argparse

#####
##  ARGUMENT PARSER
#####
parser = argparse.ArgumentParser(description='run an MCMC on three bursts using wavelets and eccprior')

parser.add_argument('-v', '--verbose',
                    action='store_true', required=False,
                    help='print verbose output')
parser.add_argument('-o', '--outdir',
                    action='store', required=False, default='.',
                    help='directory for output')
parser.add_argument('-m', '--mtot',
                    action='store', required=True, type=float,
                    help='total mass of the source (solar masses)')
parser.add_argument('-q', '--mass_ratio',
                    action='store', required=True, type=float,
                    help='mass ratio')
parser.add_argument('--destar',
                    action='store', required=True, default=1e-3, type=float,
                    help='1-e where e is eccentricity')
parser.add_argument('--tstar',
                    action='store', required=True, default=0.1, type=float,
                    help='central time of the anchor burst (sec)')

parser.add_argument('--tmin',
                    action='store', required=True, default=-5., type=float,
                    help='start of time window (sec)')
parser.add_argument('--tmax',
                    action='store', required=True, default=0., type=float,
                    help='end of time window (sec)')


args = parser.parse_args()

VERBOSE = args.verbose

def get_f_ISCO(Mtot):
    '''calculates the ISCO frequency for a given mass'''
    c = 299792458
    GMsun = 1.32712440018e20
    f_isco = c**3/(np.pi*6**(3/2)*GMsun*Mtot)
    
    return f_isco

#Establish ecc_burst_new class
Mtot = args.mtot #total mass
q = args.mass_ratio #mass ratio
Mchirp = q**(3/5)/(1+q)**(6/5)*Mtot #chirp mass
eb = EccBurstNew(q) #establishing eccburst class

#Conversion factors
GMsun = 1.32712440018e20  # m^3/s^2
c = 299792458 # m/s
Tsun = GMsun / c**3
M2sec = Tsun*Mtot

#Convert meta params to mass units to be passed to eccburst
tstar = args.tstar #central time of anchor burst
tstarM = tstar/M2sec
fstar = 0.5*get_f_ISCO(Mtot) #central frequency of anchor burst
fstarM = fstar*M2sec
destar = args.destar #eccentricity of anchor burst
tmin = args.tmin #beginning of time window
tmax = args.tmax #end of time window
tminM = tmin/M2sec
tmaxM = tmax/M2sec
tmin, tmax = tminM*M2sec, tmaxM*M2sec

print('generating bursts')
#get bursts from eccburst
bursts = eb.get_all_bursts(tstarM, fstarM, destar, tminM, tmaxM)
tf_bursts_SI = np.array([[t*M2sec, f/M2sec, de] for t,f, de in bursts]) #converting to SI units

print('bursts generated')
#create output directory
outdir = args.outdir

if not os.path.exists(os.path.dirname(outdir)):
    try:
        os.makedirs(os.path.dirname(outdir))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
            
#save bursts to a file
print('saving bursts to file')
np.savetxt(outdir+'bursts.dat', tf_bursts_SI)
