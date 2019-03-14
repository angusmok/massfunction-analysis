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