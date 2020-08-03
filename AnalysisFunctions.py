# !/usr/bin/env python

#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------

# Declarations + Constants - v.20190603 (20190409)

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
import scipy.interpolate as interpolate
import astropy.wcs as wcs
import astropy.io as io
import aplpy
# import pymc3 as pm
import inspect as inspect
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
os.environ['MKL_THREADING_LAYER'] = 'GNU'
mcmcnumsample = 2500
ncmcnumtune = 500
mcmcsigma = 2.
optimizesigma = 2.
bootstrapnumsample = 1000
plotstretchfactor = 1.00
masslimtestfactor = 3
num_interp = 50
### Star Formation Rates ###
Ant_SFR = 20. # Chandar+17 (HA)
LMC_SFR = 0.25 # Chandar+17 (HA)
M33_SFR = 0.56 # Javadi+16
M51_SFR = 3.20 # Chandar+17 (HA)
M83_SFR = 2.65 # Chandar+17 (HA)
M100_SFR = 2.6 # Wilson+09
NGC300_SFR = 0.3 # Kang+
NGC3256_SFR = 50 # Chandar+17 (HA)
NGC4214_SFR = 0.11 # Chandar+17 (HA)
NGC4449_SFR = 0.35 # Chandar+17 (HA)
NGC4526_SFR = 0.03 # Amblard+14
NGC4826_SFR = 0.19 # Braun+94
NGC6946_SFR = 3.24 # Leroy+08
SMC_SFR = 0.06 # Chandar+17 (HA)
### Star Formation Rates (Corrected) ###
LMC_SFRC = np.array([0.25, 0.32, 0.22]) * 0.70
SMC_SFRC = np.array([0.06, 0.16, 0.14]) * 0.90
NGC4214_SFRC = np.array([0.11, 0.07, 0.06]) * 1.00
NGC4449_SFRC = np.array([0.35, 0.49, 0.41]) * 0.85
M83_SFRC = np.array([2.65, 3.12, 2.35]) * 0.60
M51_SFRC = np.array([3.20, 3.54, 2.45]) * 0.90
Ant_SFRC = np.array([20, 11, 11]) * 1.00
NGC3256_SFRC = np.array([50, 47, 48]) * 1.00
###
M51_LEG_SFRC = np.array([1.437, 1.636, 1.636])
### Distances ###
Ant_dist = 22.*1E6
LMC_dist = 0.05*1E6
M33_dist = 840000.
M51_dist = 8.2*1E6
M83_dist = 4.5*1E6
NGC3256_dist = 36*1E6
NGC4214_dist = 3.1*1E6
NGC4449_dist = 3.8*1E6
NGC3627_dist = 10100000.
NGC6744_dist = 7300000.
SMC_dist = 0.06*1E6
### Number of Clusters ###
Ant_Num_age3 = 990
LMC_Num_age3 = 121
M51_Num_age3 = 633
M83_Num_age3 = 245
DSS_folder = '/Users/Angus/Documents/SD/DSS/'
HST_folder = '/Users/Angus/Documents/SD/HST/'
### Cluster Completeness Limits ###
Test_SC_complimits = [np.power(10, 4.0), np.power(10, 4.0), np.power(10, 4.0)]
Dwarf_SC_complimits_old = [np.power(10, 3.7), np.power(10, 3.7), np.power(10, 3.7)]
Dwarf_SC_complimits = [np.power(10, 3.4), np.power(10, 3.7), np.power(10, 3.7)]
###
Ant_SC_complimits = [np.power(10, 4.0), np.power(10, 4.25), np.power(10, 4.5)]
LMC_SC_complimits = [np.power(10, 2.5), np.power(10, 3.2), np.power(10, 3.25)]
M31_SC_complimits = [np.power(10, 3.5), np.power(10, 3.5), np.power(10, 3.5)]
M33_SC_complimits = [np.power(10, 2.5), np.power(10, 2.8), np.power(10, 3.2)]
M51_SC_complimits = [np.power(10, 3.5), np.power(10, 3.9), np.power(10, 4.0)]
M83_SC_complimits = [np.power(10, 3.3), np.power(10, 3.7), np.power(10, 4.0)]
NGC3256_SC_complimits = [np.power(10, 5.2), np.power(10, 5.2), np.power(10, 5.5)]
NGC3627_SC_complimits = [np.power(10, 3.5), np.power(10, 4.2), np.power(10, 4.5)]
NGC4214_SC_complimits = [np.power(10, 2.5), np.power(10, 3.0), np.power(10, 3.0)]
NGC4449_SC_complimits = [np.power(10, 3.4), np.power(10, 4.0), np.power(10, 4.0)]
SMC_SC_complimits = [np.power(10, 2.5), np.power(10, 3.2), np.power(10, 3.25)]
###
NGC0045_SC_complimits = [np.power(10, 2.5), np.power(10, 3.0), np.power(10, 3.3)]
NGC0628_SC_complimits = [np.power(10, 3.7), np.power(10, 3.7), np.power(10, 3.7)]
NGC1433_SC_complimits = [np.power(10, 2.5), np.power(10, 3.0), np.power(10, 3.6)]
NGC1566_SC_complimits = [np.power(10, 3.2), np.power(10, 3.8), np.power(10, 4.3)]
NGC1705_SC_complimits = [np.power(10, 2.5), np.power(10, 3.0), np.power(10, 3.5)]
NGC3344_SC_complimits = [np.power(10, 2.5), np.power(10, 3.2), np.power(10, 3.6)]
NGC3351_SC_complimits = [np.power(10, 2.7), np.power(10, 3.3), np.power(10, 4.0)]
NGC3738_SC_complimits = [np.power(10, 2.5), np.power(10, 3.2), np.power(10, 3.5)]
NGC4242_SC_complimits = [np.power(10, 2.5), np.power(10, 3.0), np.power(10, 3.3)]
NGC4395_SC_complimits = [np.power(10, 2.5), np.power(10, 3.0), np.power(10, 3.5)]
NGC4656_SC_complimits = [np.power(10, 2.5), np.power(10, 3.0), np.power(10, 3.5)]
NGC5238_SC_complimits = [np.power(10, 2.5), np.power(10, 3.0), np.power(10, 3.25)]
NGC5253_SC_complimits = [np.power(10, 2.5), np.power(10, 3.0), np.power(10, 3.25)]
NGC5457_SC_complimits = [np.power(10, 2.5), np.power(10, 3.0), np.power(10, 3.7)]
NGC5474_SC_complimits = [np.power(10, 2.5), np.power(10, 3.0), np.power(10, 3.3)]
NGC5477_SC_complimits = [np.power(10, 2.5), np.power(10, 3.0), np.power(10, 3.25)]
NGC6503_SC_complimits = [np.power(10, 2.5), np.power(10, 3.0), np.power(10, 3.3)]
NGC7793_SC_complimits = [np.power(10, 2.5), np.power(10, 3.2), np.power(10, 3.5)]
### GMC Completeness Limits ###
Ant_Z14_GMC_complimits = np.power(10, 6.7)
LMC_W11_GMC_complimits = 3E4 # log = 4.47
M33_R07_GMC_complimits = 1.3E5 # log = 5.11
M33_C17_GMC_complimits = 6.3E4 # log = 4.8
M51_C14_GMC_complimits = 3.6E5 # log = 5.56
M83_F17_GMC_complimits = 5E5 # log = 5.7
M100_P17_GMC_complimits = 8.8E6 # log = 6.95
NGC300_F16_GMC_complimits = 1.7E4 # log = 4.23
NGC3256_N_complimits = np.power(10, 7.1)
NGC3627_N_complimits = 1E6
NGC4526_U15_GMC_complimits = np.power(10, 5.7)
NGC4826_D13_GMC_complimits = 1E6
NGC6744_N_complimits = 1E1
NGC6946_W17_GMC_complimits = 2E5
###
Ant_Z14_GMC_complimits_mf = np.power(10, 7.0) # 5% = 7.5
LMC_W11_GMC_complimits_mf = np.power(10, 4.5) # 5% = 4.3
M51_C14_GMC_complimits_mf = np.power(10, 6.1) # 5% = 6.1
M83_F17_GMC_complimits_mf = np.power(10, 6.1) # 5% = 6.1
NGC3256_N_complimits_mf = np.power(10, 7.6) # 5% = 7.6
NGC3627_N_complimits_mf = np.power(10, 6.5) # 5% = 6.6
### Correction Factors ###
Ant_PowerLaw_F = [0.589, 0.536, 0.486]
Ant_Schechter5_F = [0.286, 0.207, 0.135]
LMC_PowerLaw_F = [0.898, 0.754, 0.744]
LMC_Schechter5_F = [0.819, 0.566, 0.549]
M51_PowerLaw_F = [0.693, 0.610, 0.589]
M51_Schechter5_F = [0.460, 0.319, 0.286]
M83_PowerLaw_F = [0.734, 0.651, 0.589]
M83_Schechter5_F = [0.531, 0.389, 0.286]
NGC3256_PowerLaw_F = [0.340, 0.340, 0.280]
NGC3256_Schechter5_F = [0.011, 0.011, 0.001]
NGC4214_PowerLaw_F = [0.898, 0.795, 0.795]
NGC4214_Schechter5_F = [0.819, 0.639, 0.639]
NGC4449_PowerLaw_F = [0.713, 0.589, 0.589]
NGC4449_Schechter5_F = [0.495, 0.286, 0.286]
SMC_PowerLaw_F = [0.898, 0.754, 0.744]
SMC_Schechter5_F = [0.819, 0.566, 0.549]
Test_PowerLaw_F = [0.589, 0.589, 0.589]
Test_Schechter5_F = [0.286, 0.286, 0.286]
Test37_PowerLaw_F = [0.651, 0.651, 0.651]
Test37_Schechter5_F = [0.389, 0.389, 0.389]

#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------

###
# Check Folders
###
if not os.path.exists('./ClusterTemp/'):
	if not os.path.exists('./FiguresGMC/'):
		os.makedirs('./FiguresGMC')
	if not os.path.exists('./FiguresSC/'):
		os.makedirs('./FiguresSC')
	if not os.path.exists('./FiguresXB/'):
		os.makedirs('./FiguresXB')
	if not os.path.exists('./FiguresSimSC/'):
		os.makedirs('./FiguresSimSC')
	if not os.path.exists('./FiguresSummary/Radius/'):
		os.makedirs('./FiguresSummary/Radius')
	if not os.path.exists('./FiguresSummary/'):
		os.makedirs('./FiguresSummary')
	if not os.path.exists('./FiguresSCGMC/'):
		os.makedirs('./FiguresSCGMC')
	if not os.path.exists('./FiguresSCCorr/'):
		os.makedirs('./FiguresSCCorr')
	if not os.path.exists('./GridLike/'):
		os.makedirs('./GridLike')
	if not os.path.exists('./Logs/'):
		os.makedirs('./Logs')
	if not os.path.exists('./Output/'):
		os.makedirs('./Output')
	
print('Import - AnalysisFunctions - v2020.08')

#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------

###
# (1) Math Functions
###

def simplepowerlaw(M, M0, Gamma):

	'''
	Function: Define simple power law function
	'''

	return (np.power((M / M0), Gamma))

def truncatedpowerlaw(M, N0, M0, Gamma):

	'''
	Function: Define truncated power law function
	'''

	return N0 * (np.power((M / M0), Gamma) - 1)

def truncatedpowerlaw_1(M, N0, M0):

	'''
	Function: Define truncated power law function with a slope of -1
	'''

	return N0 * (np.power((M / M0), -1.) - 1)

def schechter(M, phi, M0, Gamma):

	'''
	Function: Define Schechter Function
	'''

	return phi * (np.power((M / M0), Gamma)) * np.exp(-(M / M0))

def schechter_1(M, phi, M0):

	'''
	Function: Define Schechter Function with a slope of -1
	'''

	return phi * (np.power((M / M0), -1.)) * np.exp(-(M / M0))

def schechter_2(M, phi, M0):

	'''
	Function: Define Schechter Function with a slope of -2
	'''

	return phi * (np.power((M / M0), -2.)) * np.exp(-(M / M0))

def simplepowerlaw_log(logM, M0, Gamma):

	'''
	Function: Define simple power law function (log)
	'''

	return (Gamma * logM) - (Gamma * np.log10(M0))

def truncatedpowerlaw_log(logM, N0, M0, Gamma):

	'''
	Function: Define truncated power law function (log)
	'''

	return np.log10(N0) + np.log10(np.power((np.power(10, logM) / M0), Gamma) - 1)

def truncatedpowerlaw_1_log(logM, N0, M0):

	'''
	Function: Define truncated power law function (log) with a slope of -1
	'''

	return np.log10(N0) + np.log10(np.power((np.power(10, logM) / M0), -1.) - 1)

def truncatedpowerlaw_2_log(logM, N0, M0):

	'''
	Function: Define truncated power law function (log) with a slope of -2
	'''

	return np.log10(N0) + np.log10(np.power((np.power(10, logM) / M0), -2.) - 1)

def schechter_log(logM, phi, M0, Gamma):

	'''
	Function: Define Schechter Function (log)
	'''

	return  np.log10(phi) + (Gamma * logM) - (Gamma * np.log10(M0)) + np.log10(np.exp(-(np.power(10, logM) / M0)))

def schechter_1_log(logM, phi, M0):

	'''
	Function: Define Schechter Function (log) with a slope of -1
	'''

	return  np.log10(phi) + (1. * logM) - (1. * np.log10(M0)) + np.log10(np.exp(-(np.power(10, logM) / M0)))

def schechter_2_log(logM, phi, M0):

	'''
	Function: Define Schechter Function (log) with a slope of -2
	'''

	return  np.log10(phi) + (2. * logM) - (2. * np.log10(M0)) + np.log10(np.exp(-(np.power(10, logM) / M0)))

def trace_mean(x):

	'''
	Function: Return the mean of the trace
	'''
	
	return pd.Series(np.mean(x, 0), name = 'mean')


def trace_sd(x):

	'''
	Function: Return the standard deviation of the trace
	'''
	
	return pd.Series(np.std(x, 0), name = 'sd')

def linefunction(x, m, b):

	'''
	Function: Define simple linear function
	'''

	return (m * x) + b

###
# (2) Printing Function
###

def printarraynumbering(array):

	'''
	Function: Print array with numbering
	'''

	string_out = ''
	for i in range(0, len(array)):
		string_out = string_out + '{}[{}], '.format(array[i], i)

	print(string_out)

	return 0

###
# (3) Declare Simple Functions
###

def rsquared(x, y):

	'''
	Function: Determine r-square value
	'''

	slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

	return r_value**2

def reducedchisq(ydata, ymod, dof, sd): 

	'''
	Function: Returned chi-squared value
	'''
 
	chisq = np.sum(((ydata - ymod) / sd) ** 2) 
			
	nu = len(ydata) - 1 - dof
	reducedchisq_val = chisq / nu

	return reducedchisq_val       

def log10_labels(x, pos):

	'''
	Function: Label with log instead of linear value
	log10_labels_format = plt.FuncFormatter(log10_labels)
	'''

	return '%1i' % (np.log10(x))
	
def find_nearest2(array, value):
	
	'''
	Function: Search for nearest value in ascending/descending values
	'''

	idx1, idx2 = 0, 0

	# Go from start to end
	for i in range(0, len(array)):

		# Check if array value is less than true value (passes contour)
		if array[i] < value and array[i] > 0.1:

			# If value is at start, then just accept it
			if i == 0:
				idx1 = i
			# Check if the previous array value was actually closer
			elif abs(array[i - 1] - value) < abs(array[i] - value):
				idx1 = i - 1

			# If not, then accept it
			else:
				idx1 = i
			break

	# Go from end to start
	for i in range(len(array) - 1, 0, -1):
		if array[i] < value and array[i] > 0.1:
			if i == len(array) - 1:
				idx2 = i
			elif abs(array[i + 1] - value) < abs(array[i] - value):
				idx2 = i + 1
			else:
				idx2 = i
			break

	return idx1, idx2

def find_nearest2guided(array, value, org_idx):
	
	'''
	Function: Search for nearest value in guided
	'''


	idx1, idx2 = 0, len(array) - 1

	# For lower bound, start at original index, then go to zero
	for i in range(org_idx, 0, -1):

		# Check if array value is greater than the true value (passes contour)
		if array[i] > value:

			# If value is at start, then just accept it
			# Check if the previous array value was actually closer
			if abs(array[i + 1] - value) < abs(array[i] - value):
				idx1 = i + 1

			# If not, then accept it
			else:
				idx1 = i

			break

	# For upper bound, start at orginal index, then go to end of array
	for i in range(org_idx, len(array) - 1):


		if array[i] > value:
		
			# If value is at start, then just accept it
			# Check if the previous array value was actually closer
			if abs(array[i - 1] - value) < abs(array[i] - value):
				idx2 = i - 1
			# If not, then accept it
			else:
				idx2 = i
			break


	return idx1, idx2