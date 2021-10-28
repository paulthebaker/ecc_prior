#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import arviz as az
import json
import os
from ecc_prior_new import NewPrior

def prior_mapping(tmin, tmax, Mtot, Mc, destar, tstar, fstar):
    '''Creates a 2-D histogram of the prior surface for a given set of meta parameters
       Returns the ts, fs, and surface probabilities'''
    Nf = 500
    Nt = 500

    Fs = np.linspace((fstar-20), (fstar+20), Nf)
    Ts = np.linspace(tmin, tmax, Nt)

    prior_map = np.zeros([Nf, Nt])

    for ii,tt in enumerate(Ts):
        for jj,ff in enumerate(Fs):
            prior_map[jj,ii] = ep.get_prior([[tt,ff]], Mtot, Mc, destar, tstar, fstar)
    
    return Ts, Fs, prior_map

print('loading wavelet data')
#load the wavelet dataset
wavelet_data = np.loadtxt('data_set.txt').T

print('loading meta parameters')
#Load meta parameters from the json file
with open('meta_params.json') as meta_file:
    meta_params = json.load(meta_file)

#Assign meta parameter values to an array
meta_p = np.fromiter(meta_params.values(), dtype=float)

#Establish the eccprior class
ep = NewPrior(wavelet_data[0,0], wavelet_data[-1,0])

print('generating blob map')
#Create the prior map
ts, fs, blob_map = prior_mapping(wavelet_data[0,0], wavelet_data[-1,0], *meta_p)

#Plot the prior map
prior_map_fig = plt.figure(figsize=(12,10))
ax = prior_map_fig.add_subplot(111)
ax.pcolormesh(ts, fs, blob_map, cmap='viridis')
ax.set_xlabel('$t$ (sec)', fontsize=20);
ax.set_ylabel('$f$ (Hz)', fontsize=20);
ax.tick_params(labelsize=16)

plt.savefig('prior_map.png')

print('loading log posterior chains')
#load the log posterior probabilities
log_post = np.loadtxt('logpost.txt')

#Plot logpost chains
walkers_idx = np.arange(0,500,50)
for ii in walkers_idx:
    plt.plot(log_post[:,ii])
plt.xlabel('iterations')
plt.ylabel('log_post_prob')
plt.savefig('log_post.png')

print('loading chains')
#load chains
burnin = 600
ecc_chains_run1 = np.load('chains.npy')
ecc_chains_burned = ecc_chains_run1[burnin:,:,:]

print('loading samples')
#Load samples from eccprior runs
ecc_samples_run1 = np.loadtxt('samples.txt')

#get the rows, cols, and dim from the shape of chains
rows, cols, dims = np.shape(ecc_chains_burned)

#Take a subset of the samples from chains
sample_subset =  np.random.randint(0, rows, size=5000)
ecc_sub_samples_run1 = ecc_samples_run1[sample_subset,:]

length, width = np.shape(ecc_sub_samples_run1)

if width == 20:
    param_list = [r'$t_{1}$',r'$f_{1}$',r'$A_{1}$',r'$Q_{1}$',r'$\phi_{1}$',
              r'$t_{2}$',r'$f_{2}$',r'$A_{2}$',r'$Q_{2}$',r'$\phi_{2}$',
              r'$t_{3}$',r'$f_{3}$',r'$A_{3}$',r'$Q_{3}$',r'$\phi_{3}$',
              r'$M$', r'$M_{c}$', r'$\delta e_{*}$',r'$t_{*}$', r'f_{*}$']
    print('generating end-start state plot')
    #end state distributions for each parameter
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(12,16))
    for samps, ax, pp in zip(ecc_chains_run1[-1].T, axes.flatten(), param_list):
        ax.hist(samps, histtype='step', density=True, bins=20, label=pp)
        ax.legend()
    #start state distributions for each parameter
    for samps, ax, pp in zip(ecc_chains_run1[0].T, axes.flatten(), param_list):
        ax.hist(samps, histtype='step', density=True, bins=20, label=pp)
        ax.legend()

    plt.savefig('start_end_states_dist.png')


    print('generating parameter post distribution histograms')
    #Create parameter dictionary for arviz to plot the posterior distribution for each parameter
    param_dict = {}
    for i,par in enumerate(param_list):
        param_dict[par] = ecc_sub_samples_run1[:,i]
    az_wave_chains = az.convert_to_inference_data(param_dict)

    #Plot posterior distributions for each parameter
    az.plot_posterior(az_wave_chains, var_names=param_list, bins=50, kind='hist', backend='matplotlib');
    plt.savefig('kombine_hist_eccprior.png')

    print('generating corner plots')
    #Separate samples for each wavelet and the meta parameters
    wave_samples = ecc_sub_samples_run1[:,[0,1,5,6,10,11]]
    meta_samples = ecc_sub_samples_run1[:,[15,16,17,18,19]]

    #Parameter names for eache wavelet and meta parameters
    wave_param_list = [r'$t_{1}$',r'$f_{1}$',r'$t_{2}$',r'$f_{2}$',r'$t_{3}$',r'$f_{3}$']
    meta_param_list = [r'$M$', r'$M_{c}$', r'$\delta e_{*}$',r'$t_{*}$', r'f_{*}$']

    #Create dictionaries for each wavelet and the set of meta parameters
    wave_param_dict = {}
    meta_param_dict = {}
    for i,par in enumerate(wave_param_list):
        wave_param_dict[par] = wave_samples[:,i]
    for i,par in enumerate(meta_param_list):
        meta_param_dict[par] = meta_samples[:,i]
    
    #Convert chains for each wavelet and the set of meta parameters ot inference data for arviz plots
    wave_chains = az.convert_to_inference_data(wave_param_dict)
    meta_chains = az.convert_to_inference_data(meta_param_dict)

    #Create corner plots for eache wavelet and the set of meta parameters
    az.plot_pair(wave_chains,var_names=wave_param_list, kind='scatter', textsize=22, backend='matplotlib');
    plt.savefig('tf_pair_plot.png')

    az.plot_pair(meta_chains,var_names=meta_param_list, kind='scatter', textsize=22, backend='matplotlib');
    plt.savefig('meta_param_pair_plot.png')

    print('done!')

else:
    param_list = [r'$t_{1}$',r'$f_{1}$',r'$A_{1}$',r'$Q_{1}$',r'$\phi_{1}$',
              r'$t_{2}$',r'$f_{2}$',r'$A_{2}$',r'$Q_{2}$',r'$\phi_{2}$',
              r'$t_{3}$',r'$f_{3}$',r'$A_{3}$',r'$Q_{3}$',r'$\phi_{3}$']
    
    print('generating end-start state plot')
    #end state distributions for each parameter
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(12,16))
    for samps, ax, pp in zip(ecc_chains_run1[-1].T, axes.flatten(), param_list):
        ax.hist(samps, histtype='step', density=True, bins=20, label=pp)
        ax.legend()
    #start state distributions for each parameter
    for samps, ax, pp in zip(ecc_chains_run1[0].T, axes.flatten(), param_list):
        ax.hist(samps, histtype='step', density=True, bins=20, label=pp)
        ax.legend()

    plt.savefig('start_end_states_dist.png')

    print('generating parameter post distribution histograms')
    #Create parameter dictionary for arviz to plot the posterior distribution for each parameter
    param_dict = {}
    for i,par in enumerate(param_list):
        param_dict[par] = ecc_sub_samples_run1[:,i]
    az_wave_chains = az.convert_to_inference_data(param_dict)

    #Plot posterior distributions for each parameter
    az.plot_posterior(az_wave_chains, var_names=param_list, bins=50, kind='hist', backend='matplotlib');
    plt.savefig('kombine_hist_eccprior.png')

    print('generating corner plots')
    #Separate samples for each wavelet and the meta parameters
    wave_samples = ecc_sub_samples_run1[:,[0,1,5,6,10,11]]


    #Parameter names for eache wavelet and meta parameters
    wave_param_list = [r'$t_{1}$',r'$f_{1}$',r'$t_{2}$',r'$f_{2}$',r'$t_{3}$',r'$f_{3}$']

    #Create dictionaries for each wavelet and the set of meta parameters
    wave_param_dict = {}

    for i,par in enumerate(wave_param_list):
        wave_param_dict[par] = wave_samples[:,i]

    #Convert chains for each wavelet and the set of meta parameters ot inference data for arviz plots
    wave_chains = az.convert_to_inference_data(wave_param_dict)

    #Create corner plots for eache wavelet and the set of meta parameters
    az.plot_pair(wave_chains,var_names=wave_param_list, kind='scatter', textsize=22, backend='matplotlib');
    plt.savefig('tf_pair_plot.png')

    print('done!')
