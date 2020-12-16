#!/usr/bin/env python
# coding: utf-8

''' This code is a toy model of BayesWave and uses three Morlet-Gabor wavelets to represent three
bursts in a series of bursts for a particular eccentric gravitational wave signal. It runs a simple MCMC analysis
on a select part of the signal using wavelets and eccprior, a prior that uses a Newtownian approximation to
determine the centroids of bursts in an eccentric signal.
'''

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Belinda D. Cheeseboro"
__version__ = "2020-11-23"


import sys
import numpy as np
import matplotlib.pyplot as plt
import kombine as kb
import arviz as az
from corner import corner
from ecc_prior_new import NewPrior
from ecc_burst_new import EccBurstNew
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
parser.add_argument('-m', '--mtot',
                    action='store', required=True, type=float,
                    help='total mass of the source (solar masses)')
parser.add_argument('-q', '--mass_ratio',
                    action='store', required=True, type=float,
                    help='mass ratio')
parser.add_argument('-A', '--amp',
                    action='store', required=True, type=float,
                    help='amplitude [arb]')
parser.add_argument('--anchor_idx',
                    action='store', required=True, type=int,
                    help='anchor burst index.')
parser.add_argument('--tmin',
                    action='store', required=True, type=float,
                    help='time window lower bound')
parser.add_argument('--tmax',
                    action='store', required=True, type=float,
                    help='time window upper bound')
parser.add_argument('--burst_file',
                    action='store', required=True, default='.',
                    help='burst file location')
parser.add_argument('--runs', action='store', required=False, default=1000, type=int,
		    help='number of iterations for the MCMC')
parser.add_argument('--walkers', action='store', required=False, default=500, type=int,
		    help='number of iterations for the MCMC')

args = parser.parse_args()

VERBOSE = args.verbose

def time_wavelet(times,  ctime, cfreq, amp, Q, phi0):
    "Time-domain Morlet-Gabor wavelet based on equation 4 from the BW paper"
    
    tau = Q/(2*np.pi*cfreq)
    dt = times-ctime
    
    psi = amp*np.exp(-dt**2/tau**2)*np.cos(2*np.pi*cfreq*dt+phi0)
    return psi

def get_snr(amp, cfreq, Q):
    '''prior for Morlet-Gabor wavelet given by equation 9 from BW paper
    Assuming SNR_{*} = 1 and S_{n} = 1
    '''
    SNR = ((amp*np.sqrt(Q))/np.sqrt(2*np.sqrt(np.pi*2)*cfreq))
    return SNR


def get_logL(wavelet_model, wavelet_data):
    '''Calculates the log likelihood for given dataset with wavelets'''
    res = wavelet_data-wavelet_model
    log_prob = -0.5*(np.sum(np.square(res)))
    return log_prob

def get_signal_logprior(amp, cfreq, Q):
    '''amplitude prior from BW paper (equation 12)'''
    SNR = get_snr(amp, cfreq, Q)
    amp_prior = (3*SNR)/(4*(1+(SNR/4)))
    return np.log(amp_prior)

def get_logP_uniform(wavelet_params, times):
    '''checks if wavelet parameters and eccprior meta parameters are within the bounds'''

    #assign values to the appropriate parameters
    ts = [wavelet_params[0], wavelet_params[5], wavelet_params[10]]
    fs = [wavelet_params[1], wavelet_params[6], wavelet_params[11]]
    amps = [wavelet_params[2], wavelet_params[7], wavelet_params[12]]
    Qs = [wavelet_params[3], wavelet_params[8], wavelet_params[13]]
    phis = [wavelet_params[4], wavelet_params[9], wavelet_params[14]]
    meta_params = np.array([wavelet_params[15], wavelet_params[16], wavelet_params[17], wavelet_params[18], wavelet_params[19]])
    
    dt_inv = 1/(times[1]-times[0])
    
    for i,t in enumerate(ts):
        if t < times[0] or t > times[-1]:
            return -np.inf
    for i,f in enumerate(fs):    
        if f < abs(1/times[0]) or f > (dt_inv/2):
            return -np.inf
    for i,q in enumerate(Qs):
        if q < 1.5 or q > 15:
            return -np.inf
    for i,a in enumerate(amps):
        if a < 0:
            return -np.inf
    for i,phi in enumerate(phis):
        if phi < -np.pi or phi > np.pi:
            return -np.inf
    lp = ep.get_logprior([[ts[0],fs[0]],[ts[1],fs[1]], [ts[2],fs[2]]], *meta_params)
    if np.isinf(lp):
        return lp
    else:
        return 0 

def get_logP(wavelet_params, times, wavelet_data):
    "calculates the log probability for a given sent of wavelets"
    lp = get_logP_uniform(wavelet_params, times)
    if np.isinf(lp):
        return lp
    else:
        #assign values to the appropriate parameters
        ts = [wavelet_params[0], wavelet_params[5], wavelet_params[10]]
        fs = [wavelet_params[1], wavelet_params[6], wavelet_params[11]]
        amps = [wavelet_params[2], wavelet_params[7], wavelet_params[12]]
        Qs = [wavelet_params[3], wavelet_params[8], wavelet_params[13]]
        phis = [wavelet_params[4], wavelet_params[9], wavelet_params[14]]
        meta_params = np.array([wavelet_params[15], wavelet_params[16], wavelet_params[17], wavelet_params[18], wavelet_params[19]])
        
        #calculate prior probabilities
        log_ecc = ep.get_logprior([[ts[0],fs[0]],[ts[1],fs[1]], [ts[2],fs[2]]], *meta_params)
        wavelet_model = 0
        log_sig_prior = 0
        for i,t in enumerate(ts):
            wavelet_model += time_wavelet(times, t, fs[i], amps[i], Qs[i], phis[i])
            log_sig_prior += get_signal_logprior(amps[i], fs[i], Qs[i])
        #calculate likelihood probability
        log_like = get_logL(wavelet_model, wavelet_data)
        #calculate total probability
        log_prob = log_like + log_sig_prior + log_ecc
        return log_prob

def start_loc(wavelet_params, nwalkers, dim, times, wavelet_data):
    '''Picks starting locations for the walkers using the given meta-parameters for a given waveform'''
    x_sigma = [0.01, 1, 1, 0.1, 0.2, 0.01, 1, 1, 0.1, 0.2, 0.01, 1, 1, 0.1, 0.2, 1, 0.5, 0.01, 0.01, 1]
    x0 = np.array([wavelet_params + x_sigma*np.random.randn(dim)
                   for i in range(nwalkers)])

    for i,x in enumerate(x0):
        lp = get_logP(x, times, wavelet_data)
        while np.isinf(lp):
            x = np.array([wavelet_params + x_sigma*np.random.randn(dim)])
            x = x.ravel()
            lp = get_logP(x, times, wavelet_data)
            x0[i] = x
    return x0

#read in bursts file
bursts = np.loadtxt(args.burst_file, delimiter=' ')
anchor_idx = args.anchor_idx #anchor burst index
t_SI, f_SI, de = bursts.T

#Establish meta parameters
Mtot = args.mtot
q = args.mass_ratio
Mchirp = q**(3/5)/(1+q)**(6/5)*Mtot
destar = de[anchor_idx]
tstar = t_SI[anchor_idx]
fstar = f_SI[anchor_idx]

eb = EccBurstNew(q)

#Conversion factors
GMsun = 1.32712440018e20  # m^3/s^2
c = 299792458 # m/s
Tsun = GMsun / c**3
M2sec = Tsun*Mtot

#convert meta params to mass units
tstarM = tstar/M2sec
fstarM = fstar*M2sec
tmin = args.tmin
tminM = tmin/M2sec
tmaxM = args.tmax

meta_params = np.array([Mtot, Mchirp, destar, tstar, fstar])

#generate bursts for anchor
bursts = eb.get_all_bursts(tstarM, fstarM, destar, tminM, tmaxM)
ts, fs, des = np.array([[t*M2sec, f/M2sec, de] for t,f, de in bursts]).T

new_anchor_idx = np.where(ts==tstar)[0][0]

if new_anchor_idx == 0:
    t_low = ts[new_anchor_idx]
    t_mid = ts[new_anchor_idx+1]
    t_high = ts[new_anchor_idx+2]

    f_low = fs[new_anchor_idx]
    f_mid = fs[new_anchor_idx+1]
    f_high = fs[new_anchor_idx+2]
    
elif new_anchor_idx >= 1 and new_anchor_idx < len(ts):
    
    t_low = ts[new_anchor_idx-1]
    t_mid = ts[new_anchor_idx]
    t_high = ts[new_anchor_idx+1]

    f_low = fs[new_anchor_idx-1]
    f_mid = fs[new_anchor_idx]
    f_high = fs[new_anchor_idx+1]

else:
    t_low = ts[new_anchor_idx-2]
    t_mid = ts[new_anchor_idx-1]
    t_high = ts[new_anchor_idx]

    f_low = fs[new_anchor_idx-2]
    f_mid = fs[new_anchor_idx-1]
    f_high = fs[new_anchor_idx]

#new time window for this set
new_tmin = t_low+((t_low-t_mid)/2)
new_tmax = t_high+((t_high-t_mid)/2)

#make wavelet dataset
Amp = args.amp
Q = 2
phi = 0


time = np.linspace(new_tmin, new_tmax, 1000)
ep = NewPrior(new_tmin, new_tmax)
wavelet_1 = time_wavelet(time, t_low, f_low, Amp, Q, phi)
wavelet_2 = time_wavelet(time, t_mid, f_mid, Amp, Q, phi)
wavelet_3 = time_wavelet(time, t_high, f_high, Amp, Q, phi)

wavelets = wavelet_1 + wavelet_2 + wavelet_3
wave_data_set = np.array([time, wavelets])
np.savetxt('data_set.txt', wave_data_set)

#plot the wavelets
wavelet_fig = plt.figure(figsize=(10,8))

ax = wavelet_fig.add_subplot(1,1,1)
ax.plot(time, wavelets)
ax.set_xlabel('time (s)')
ax.set_ylabel('amplitude')
plt.savefig('wavelets')

#setup the sampler
#Initialize MCMC
nwalkers = args.walkers
dim = 20
runs = args.runs

print('setting up sampler')
#setting up kombine sampler
sampler = kb.Sampler(nwalkers, dim, get_logP, args=[time, wavelets])

print('setting up walkers starting positions')
#setup starting locations for walkers
#parameters for wavelet 1
tstart_1 = t_low
fstart_1 = f_low

#parameters for wavelet 2
tstart_2 = t_mid
fstart_2 = f_mid

#parameters for wavelet 3
tstart_3 = t_high
fstart_3 = f_high

wavelet_params = [tstart_1, fstart_1, Amp, Q, phi, tstart_2, fstart_2, Amp, Q, phi, 
                  tstart_3, fstart_3, Amp, Q, phi, Mtot, Mchirp, destar, tstar, fstar]
x0 = start_loc(wavelet_params, nwalkers, dim, time, wavelets)

metap_dict = {'M':Mtot, 'Mc':Mchirp, 'destar':destar, 'tstar':tstar, 'fstar':fstar}
wavelets_dict = {'t_1':tstart_1, 
                 'f_1':fstart_1,
                 'A_1':Amp,
                 'Q_1':Q,
                 'phi_1':phi,
                 't_2':tstart_2, 
                 'f_2':fstart_2,
                 'A_2':Amp,
                 'Q_2':Q,
                 'phi_2':phi,
                 't_3':tstart_3, 
                 'f_3':fstart_3,
                 'A_3':Amp,
                 'Q_3':Q,
                 'phi_3':phi}

meta_json = json.dumps(metap_dict)
wavelet_json = json.dumps(wavelets_dict)

print('saving meta parameters and wavelet parameters to json files.')
meta_file = open("meta_params.json","w")
wave_file = open("wavelet_params.json",'w')
meta_file.write(meta_json)
wave_file.write(wavelet_json)

meta_file.close()
wave_file.close()

print('running burnin')
sampler.burnin(p0=x0, test_steps=30, max_steps=int(runs/2));

print('running MCMC')
#run MCMC
state = sampler.run_mcmc(runs, update_interval=20)

chains = sampler.chain
logpost = sampler.lnpost


samples = sampler.get_samples()
print("Nsamp = ", len(samples))

print('saving chains, samples, and log posterior values to a file.')
np.save('chains.npy', chains)
np.savetxt('samples.txt', samples)
np.savetxt('logpost.txt', logpost)
