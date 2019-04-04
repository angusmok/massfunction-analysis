# !/usr/bin/env python

#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------

# Declarations + Constants - v.20180314

#######################################################
# Declares Libraries
#######################################################
### Standard Libaries ###
import string
import math
import sys
import os
### Additional Libraries ###
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.path as mplpath
import matplotlib.ticker
import pylab as py
import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize
import scipy.integrate as integrate
import astropy.wcs as wcs
import astropy.io as io
import aplpy
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import pymc3 as pm
#######################################################
# Set Parameters
#######################################################
### Set Seed Variable ###
np.random.seed(146)
### Matplotlib Parameters ###
A = plt.rcParams.keys()
plt.rc('font', family = 'serif', size = 26)
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['axes.labelsize'] = 32
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['lines.markersize'] = 8
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.labelsize'] = 42
plt.rcParams['ytick.labelsize'] = 42
plt.rcParams['xtick.major.size'] = 35
plt.rcParams['xtick.minor.size'] = 22
plt.rcParams['ytick.major.size'] = 35
plt.rcParams['ytick.minor.size'] = 22
plt.rcParams['xtick.major.width'] = 4
plt.rcParams['xtick.minor.width'] = 2
plt.rcParams['ytick.major.width'] = 4
plt.rcParams['ytick.minor.width'] = 2
plt.rcParams['legend.numpoints'] = 1
### Bins ###
Bins_Hist_Mass = np.power(10, np.linspace(2, 9, num = 29))
Bins_Hist_Mass_Norm = np.power(10, np.linspace(-6, 2, num = 29))
Bins_Hist_Age = np.power(10, np.linspace(4, 11, num = 29))
### Analysis Parameters ###
mcmcnumsample = 2500
ncmcnumtune = 500
mcmcsigma = 2.
optimizesigma = 2.
bootstrapnumsample = 1000
plotstretchfactor = 1.00
masslimtestfactor = 3

#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------

###
# Declares Math Functions
###

# Function: Define simple power law function
def simplepowerlaw(M, M0, Gamma):

	return (np.power((M / M0), Gamma))

# Function: Define truncated power law function
def truncatedpowerlaw(M, N0, M0, Gamma):

	return N0 * (np.power((M / M0), Gamma) - 1)

# Function: Define truncated power law function with a slope of -1
def truncatedpowerlaw_1(M, N0, M0):

	return N0 * (np.power((M / M0), -1.) - 1)

# Function: Define Schechter Function
def schechter(M, phi, M0, Gamma):

	return phi * (np.power((M / M0), Gamma)) * np.exp(-(M / M0))

# Function: Define Schechter Function with a slope of -1
def schechter_1(M, phi, M0):

	return phi * (np.power((M / M0), -1.)) * np.exp(-(M / M0))

# Function: Define Schechter Function with a slope of -2
def schechter_2(M, phi, M0):

	return phi * (np.power((M / M0), -2.)) * np.exp(-(M / M0))

# Function: Define simple power law function (log)
def simplepowerlaw_log(logM, M0, Gamma):

	return (Gamma * logM) - (Gamma * np.log10(M0))

# Function: Define truncated power law function (log)
def truncatedpowerlaw_log(logM, N0, M0, Gamma):

	return np.log10(N0) + np.log10(np.power((np.power(10, logM) / M0), Gamma) - 1)

# Function: Define truncated power law function (log) with a slope of -1
def truncatedpowerlaw_1_log(logM, N0, M0):

	return np.log10(N0) + np.log10(np.power((np.power(10, logM) / M0), -1.) - 1)

# Function: Define truncated power law function (log) with a slope of -2
def truncatedpowerlaw_2_log(logM, N0, M0):

	return np.log10(N0) + np.log10(np.power((np.power(10, logM) / M0), -2.) - 1)

# Function: Define Schechter Function (log)
def schechter_log(logM, phi, M0, Gamma):

	return  np.log10(phi) + (Gamma * logM) - (Gamma * np.log10(M0)) + np.log10(np.exp(-(np.power(10, logM) / M0)))

# Function: Define Schechter Function (log) with a slope of -1
def schechter_1_log(logM, phi, M0):

	return  np.log10(phi) + (1. * logM) - (1. * np.log10(M0)) + np.log10(np.exp(-(np.power(10, logM) / M0)))

# Function: Define Schechter Function (log) with a slope of -2
def schechter_2_log(logM, phi, M0):

	return  np.log10(phi) + (2. * logM) - (2. * np.log10(M0)) + np.log10(np.exp(-(np.power(10, logM) / M0)))

# Function: Return the mean of the trace
def trace_mean(x):
	
	return pd.Series(np.mean(x, 0), name = 'mean')

# Function: Return the standard deviation of the trace
def trace_sd(x):
	
	return pd.Series(np.std(x, 0), name = 'sd')