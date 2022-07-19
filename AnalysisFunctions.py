# !/usr/bin/env python

#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------

# Declarations + Constants - v.2022_v2

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
LMC_SFR = 0.25 # Chandar+17
SMC_SFR = 0.06 # Chandar+17
NGC4214_SFR = 0.11 # Chandar+17
NGC4449_SFR = 0.35 # Chandar+17
M83_SFR = 2.65 # Chandar+17
M51_SFR = 3.20 # Chandar+17
Ant_SFR = 20. # Chandar+17
NGC3256_SFR = 50. # Chandar+17
###
M33_SFR = 0.56 # Javadi+16
M100_SFR = 2.6 # Wilson+09
NGC300_SFR = 0.3 # Kang+
NGC4526_SFR = 0.03 # Amblard+14
NGC4826_SFR = 0.19 # Braun+94
NGC6946_SFR = 3.24 # Leroy+08
### PHANGS
NGC1433_SFR = 0.56
NGC1559_SFR = 3.76
NGC1566_SFR = 3.21
NGC1672_SFR = 6.6
NGC1792_SFR = 3.23
NGC2775_SFR = 0.87
NGC3351_SFR = 0.87
NGC3627_SFR = 3.31
NGC4303_SFR = 4.25
NGC4321_SFR = 2.43
NGC4535_SFR = 1.23
NGC4548_SFR = 0.34
NGC4571_SFR = 0.26
NGC4654_SFR = 3.06
NGC4826_SFR = 0.17
NGC5248_SFR = 1.66
### Star Formation Rates (Corrected) ###
LMC_SFRC1 = LMC_SFR * 0.70
SMC_SFRC1 = SMC_SFR * 0.90
NGC4214_SFRC1 = NGC4214_SFR * 1.00
NGC4449_SFRC1 = NGC4449_SFR * 0.85
M83_SFRC1 = M83_SFR * 0.60
M51_SFRC1 = M51_SFR * 0.90
Ant_SFRC1 = Ant_SFR * 1.00
NGC3256_SFRC1 = NGC3256_SFR * 1.00
###
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
### Folder Locations ###
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
# NGC1433_SC_complimits = [np.power(10, 2.5), np.power(10, 3.0), np.power(10, 3.6)]
# NGC1566_SC_complimits = [np.power(10, 3.2), np.power(10, 3.8), np.power(10, 4.3)]
NGC1705_SC_complimits = [np.power(10, 2.5), np.power(10, 3.0), np.power(10, 3.5)]
NGC3344_SC_complimits = [np.power(10, 2.5), np.power(10, 3.2), np.power(10, 3.6)]
# NGC3351_SC_complimits = [np.power(10, 2.7), np.power(10, 3.3), np.power(10, 4.0)]
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
### PHANGS
NGC1433_SC_complimits = [np.power(10, 3.0), np.power(10, 3.3), np.power(10, 3.7)]
NGC1559_SC_complimits = [np.power(10, 4.2), np.power(10, 4.5), np.power(10, 4.8)]
NGC1566_SC_complimits = [np.power(10, 4.0), np.power(10, 4.2), np.power(10, 4.5)]
NGC1672_SC_complimits = [np.power(10, 4.0), np.power(10, 4.2), np.power(10, 4.5)]
NGC1792_SC_complimits = [np.power(10, 4.3), np.power(10, 4.5), np.power(10, 4.8)]
NGC2775_SC_complimits = [np.power(10, 4.3), np.power(10, 4.5), np.power(10, 4.7)]
NGC3351_SC_complimits = [np.power(10, 3.2), np.power(10, 3.5), np.power(10, 3.8)]
NGC3627_SC_complimits = [np.power(10, 3.9), np.power(10, 4.2), np.power(10, 4.5)]
NGC4303_SC_complimits = [np.power(10, 4.4), np.power(10, 4.4), np.power(10, 4.7)]
NGC4321_SC_complimits = [np.power(10, 4.2), np.power(10, 4.4), np.power(10, 4.7)]
NGC4535_SC_complimits = [np.power(10, 3.7), np.power(10, 3.9), np.power(10, 4.2)]
NGC4548_SC_complimits = [np.power(10, 3.4), np.power(10, 3.5), np.power(10, 3.7)]
NGC4571_SC_complimits = [np.power(10, 3.2), np.power(10, 3.5), np.power(10, 3.7)]
NGC4654_SC_complimits = [np.power(10, 4.2), np.power(10, 4.3), np.power(10, 4.4)]
NGC4826_SC_complimits = [np.power(10, 3.2), np.power(10, 3.3), np.power(10, 4.0)]
NGC5248_SC_complimits = [np.power(10, 3.7), np.power(10, 4.0), np.power(10, 4.2)]
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

#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
###
# (3) Simple Functions
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
	'''

	return '%1i' % (np.log10(x))

log10_labels_format = plt.FuncFormatter(log10_labels)
	
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

#------------------------------------------------------------------------------
###
# (4) Code Snippets (General)
###

def converttonumpy(array, arrayname, output = True):

	'''
	Function: Output numpy array from original array
	'''

	# Convert to numpy array, check shape
	np_array = np.ndarray((len(array), len(array[0])), dtype = object)
	for i in range(0, len(array)):
		np_array[i] = tuple(array[i])

	if output == True:
		print('- {}'.format(arrayname))
		print('Original File: {}'.format(len(array)))
		print('Array Size: {} x {}'.format(len(np_array), len(np_array[0])))
		print('Max Value: {:.2f}'.format(np.log10(np.nanmax(np_array[:,4]))))

	return np_array

def outputselectedcatalog(gal_array, outname, complimits, outputcatstofile = False):

	'''
	Function: Output selected clusters
	'''

	# Create classic 3 age bins
	gal_array_age1 = gal_array[gal_array[:,6] <= 10.01*1E6]
	gal_array_age1_masslimit = gal_array_age1[gal_array_age1[:,4] > complimits[0]]
	gal_array_age2_tmp = gal_array[gal_array[:,6] > 10.01*1E6]
	gal_array_age2 = gal_array_age2_tmp[gal_array_age2_tmp[:,6] <= 100.01*1E6]
	gal_array_age2_masslimit = gal_array_age2[gal_array_age2[:,4] > complimits[1]]
	gal_array_age3_tmp = gal_array[gal_array[:,6] > 100.01*1E6]
	gal_array_age3 = gal_array_age3_tmp[gal_array_age3_tmp[:,6] <= 400.01*1E6]
	gal_array_age3_masslimit = gal_array_age3[gal_array_age3[:,4] > complimits[2]]
	print('- Out: {}'.format(outname))
	# print('Length of Input Array: {}'.format(len(gal_array)))
	print('Input Completeness Limits: {:.1f}, {:.1f}, {:.1f}'.format(np.log10(complimits[0]), np.log10(complimits[1]), np.log10(complimits[2])))
	print('Number in Bin Above Completeness Limits: {}, {}, {}'.format(len(gal_array_age1_masslimit), len(gal_array_age2_masslimit), len(gal_array_age3_masslimit)))

	if outputcatstofile == True:
		f1 = open('./Logs/ZZOutput_' + outname +  '_Cut.txt', 'w')
		f2 = open('./Logs/ZZOutput_' + outname +  '_Full.txt', 'w')
		for i in range(0, len(gal_array)):
			if (gal_array[i][6] <= 10.01*1E6 and gal_array[i][4] > complimits[0]) or (gal_array[i][6] > 10.01*1E6 and gal_array[i][6] <= 100.01*1E6 and gal_array[i][4] > complimits[1]) or (gal_array[i][6] > 100.01*1E6 and gal_array[i][6] <= 400.01*1E6 and gal_array[i][4] > complimits[2]):
				f1.write('{}, {:.1f}, {:.2f}\n'.format(gal_array[i][1], gal_array[i][4], np.log10(gal_array[i][6])))
			f2.write('{}, {:.1f}, {:.2f}\n'.format(gal_array[i][1], gal_array[i][4], np.log10(gal_array[i][6])))
		f1.close()
		f2.close()

	return 0

def ageflagconvert(agename):

	'''
	Function: Return age name from flag
	'''

	# Set age names
	if agename == '_A1':
		ageout = r'$\tau \leq$ 10 Myr'
	elif agename == '_A2':
		ageout = r'10 < $\tau \leq$ 100 Myr'
	elif agename == '_A3':
		ageout = r'100 < $\tau \leq$ 400 Myr'
	elif agename == '_A3d':
		ageout = r'$\tau \leq$ 200 Myr'
	elif agename == '_A5':
		ageout = r'1 <= $\tau \leq$ 200 Myr'
	elif agename == '_A6':
		ageout = r'100 < $\tau \leq$ 200 Myr'
	elif agename == '_A7':
		ageout = r'$\tau \leq$ 400 Myr'
	elif agename == '_A7a':
		ageout = r'30 <= $\tau \leq$ 400 Myr'
	elif agename == '_A7b':
		ageout = r'50 <= $\tau \leq$ 400 Myr'
	elif agename == '_A7c':
		ageout = r'80 <= $\tau \leq$ 400 Myr'
	elif agename == '_A8':
		ageout = r'100 <= $\tau \leq$ 200 Myr'
	elif agename == '_A9':
		ageout = r'$\tau \leq$ 100 Myr'
	elif agename == '_AL':
		ageout = r'$\tau \leq$ 400 Myr'
	elif agename == '_T1':
		ageout = r'$\tau \leq$ 5 Myr'
	elif agename == '_T2':
		ageout = r'5 < $\tau \leq$ 10 Myr'
	else:
		ageout = agename

	return ageout

#------------------------------------------------------------------------------
###
# (5) Code Snippets (outputmassfunction)
###

def original(Beta, M0, M):

	'''
	Function: Flag 1 - Schechter function
	'''

	output = (1 / M0) * np.power(M / M0, -Beta) * np.exp(-M / M0)
	
	return output

def integrateconvolve(sigma, Beta, M0, M):

	'''
	Function: Flag 2 - Schechter function + log-normal distribution with sigma width
	'''

	sigma_e = np.log(10) * sigma

	normfun = lambda x: np.power(x, -Beta) * np.exp(- x - (np.power(np.log(M / M0) - np.log(x), 2) / (2 * sigma_e * sigma_e)))
	
	norm, normerr = integrate.quad(normfun, 0, np.inf, limit = 10000)
	output = (1. / (np.sqrt(2 * math.pi) * sigma_e * M)) * norm
	
	return output

def convolvemag(sigma, Beta, M0, M):

	'''
	Function: Flag 3 - Schechter function + Gaussian function with sigma width + mag units
	'''

	sigma_mag = sigma

	normfun = lambda x: np.power(np.power(10, 0.4 * (M0 - x)), -Beta + 1) * np.exp(-np.power(10, 0.4 * (M0 - x))) * np.exp(-np.power(x - M, 2) / (2 * sigma * sigma))
	norm, normerr = integrate.quad(normfun, -10, 10, limit = 10000)
	output = (1. / (np.sqrt(2 * math.pi) * sigma_mag)) * (0.4 * np.log(10)) * norm

	return output

def convolvemagnoerr(Beta, M0, M):

	'''
	Function: Flag 4 - Schechter function + mag units
	'''
	
	output = (0.4 * np.log(10)) * np.power(np.power(10, 0.4 * (M0 - M)), -Beta + 1) * np.exp(-np.power(10, 0.4 * (M0 - M)))

	return output

def original2(Beta, tau0, tau):

	'''
	Function: Flag 5 - Schechter function + log form
	'''

	output = np.power(np.power(10, tau - tau0), -Beta) * np.exp(-np.power(10, tau - tau0))
	
	return output

def integrateconvolve2(sigma, Beta, tau0, tau):

	'''
	Function: Flag 6 - Schechter function + Gaussian function with sigma width + log form
	'''

	sigma_e = np.log(10) * sigma
	normfun = lambda x: np.power(np.power(10, x - tau0), -Beta) * np.exp(-np.power(10, x - tau0)) * np.log(10) * np.exp(-np.power(x - tau, 2) / (2 * sigma * sigma))
	norm, normerr = integrate.quad(normfun, tau - (5 * sigma), tau + (5 * sigma), limit = 10000)
	output = ((1. / (np.sqrt(2 * math.pi) * sigma)) * norm)
	
	return output

def integratestep(width, Beta, M0, M):

	'''
	Function: Flag 7 step function (standard form)
	'''

	width_val = (width / 2) * M
	
	normfun = lambda x: (1 / M0) * np.power(x / M0, -Beta) * np.exp(-x / M0) * (0.5 / (2 * width_val)) * (np.sign((x - M) + width_val) + np.sign(width_val - (x - M)))
	
	if  (M - (5 * width_val)) > 1:
		lowerlim =  M - (5 * width_val)
	else:
		lowerlim = 1
	
	norm, normerr = integrate.quad(normfun, lowerlim, M + (5 * width_val), limit = 10000)
	if M == 1E5:
		print('For integratestep - M0 = {:.2e} and width = {:.2f} * M = {:.2e}, M({:.2e}), no error = {:.2e} vs error = {:.2e}'.format(M0, width, width_val, M, (1 / M0) * np.power(M / M0, -Beta) * np.exp(-M / M0), norm))
	if normerr > 1E5:
		print ('Integration Error is {:2e} in integratestep'.format(normerr))
	output = norm
	
	return output

def integratestep2(width, Beta, M0, M):

	'''
	Function: Flag 8 step function (fixed sigma value)
	'''

	width_val = width / 2

	if  (M - (5 * width_val)) > 1:
		lowerlim =  M - (5 * width_val)
	else:
		lowerlim = 1

	normfun = lambda x: (1 / M0) * np.power(x / M0, -Beta) * np.exp(-x / M0) * (0.5 / (2 * width_val)) * (np.sign((x - M) + width_val) + np.sign(width_val - (x - M)))
	norm, normerr = integrate.quad(normfun, lowerlim, M + (5 * width_val), limit = 10000)

	if M == 1E5:
		print('For integratestep2 - M0 = {:.2e} and width = {:.2e}, M({:.2e}), no error = {:.2e} vs error = {:.2e}'.format(M0, width_val, M, (1 / M0) * np.power(M / M0, -Beta) * np.exp(-M / M0), norm))
	if normerr > 1E5:
		print ('Integration Error is {:2e} in integratestep2'.format(normerr))
	output = norm
	
	return output

def integratestep3(sigma, Beta, M0, M):

	'''
	Function: Flag 9 step function (log error)
	'''

	width_val_high = (M * np.power(10, sigma) - M)
	width_val_low = (M - M / np.power(10, sigma))
	width_val_total = width_val_high + width_val_low
	normfun = lambda x: (1 / M0) * np.power(x / M0, -Beta) * np.exp(-x / M0) * (0.5 / (2 * width_val_total)) * (np.sign((x - M) + width_val_low) + np.sign(width_val_high - (x - M)))
	norm, normerr = integrate.quad(normfun, M / np.power(10, sigma) * 0.5, M * np.power(10, sigma) * 2, limit = 10000)

	if M == 1E5:
		print(' --- ')
		print('Sigma = {} for M {:.2e}: ({:.2e} to {:.2e})'.format(sigma, M, M / np.power(10, sigma), M * np.power(10, sigma)))
		print('Width (low) = {:.2e}, Width (high) = {:.2e}, Total Width = {:.2e}'.format(width_val_low, width_val_high, width_val_total))
		print('For integratestep3 - M0 = {:.2e} and width = {:.2e}, M({:.2e}), no error = {:.2e} vs error = {:.2e}'.format(M0, width_val_total, M, (1 / M0) * np.power(M / M0, -Beta) * np.exp(-M / M0), norm))
	if normerr > 1E5:
		print ('Integration Error is {:2e} in integratestep2'.format(normerr))
	output = norm
	
	return output

def integrateguass(sigma, Beta, M0, M):

	'''
	Function: Flag 10
	'''

	normfun = lambda x: (1 / (np.sqrt(2) * math.pi * sigma)) * (1 / M0) * np.power(x / M0, -Beta) * np.exp(-x / M0) * np.exp(- np.power(x - M0) / (2 * sigma * sigma))

	norm, normerr = integrate.quad(normfun, 0, np.inf, limit = 10000)
	output = norm

	return output

def outputmassfunction11(Beta, M0, M):

	'''
	Function: Flag 11 - 
	'''

	output = np.power((M / M0), -Beta)

	return output

def outputmassfunction12(sigma, Beta, M0, M):

	'''
	Function: Flag 12 - 
	'''

	if False:
		sigma_e = np.log(10) * sigma

		normfun = lambda x: np.power(x, -Beta) * np.exp(- x - (np.power(np.log(M / M0) - np.log(x), 2) / (2 * sigma_e * sigma_e)))
		
		norm, normerr = integrate.quad(normfun, 0, np.inf, limit = 10000)
		output = (1. / (np.sqrt(2 * math.pi) * sigma_e * M)) * norm


	sigma_e = np.log(10) * sigma
	normfun = lambda x: np.power(x, -Beta) * np.exp(- x - (np.power(- np.log(x), 2) / (2 * sigma_e * sigma_e)))
	norm, normerr = integrate.quad(normfun, 0, np.inf, limit = 10000)
	output = (1. / (np.sqrt(2 * math.pi) * sigma_e * M)) * norm
	
	return output

def outputmassfunction(sigma, Beta, M0, M_plot, flag_type, flag_log):

	'''
	Function: Output mass functions, based on values of M_plot
	'''

	output_array = []

	# Function: Flag 1 - Schechter function
	if flag_type == 1:
		if flag_log == 0:
			for i in range(0, len(M_plot)):
				output_array.append(original(Beta, M0, M_plot[i]))
		else:
			for i in range(0, len(M_plot)):
				output_array.append(np.log10(original(Beta, M0, M_plot[i])))

	# Function: Flag 2 - Schechter function + log-normal distribution with sigma width
	if flag_type == 2:
		if flag_log == 0:
			for i in range(0, len(M_plot)):
				output_array.append(integrateconvolve(sigma, Beta, M0, M_plot[i]))
		else:
			for i in range(0, len(M_plot)):
				output_array.append(np.log10(integrateconvolve(sigma, Beta, M0, M_plot[i])))

	# Function: Flag 3 - Schechter function + Gaussian function with sigma width + mag units
	if flag_type == 3:
		if flag_log == 0:
			for i in range(0, len(M_plot)):
				output_array.append(convolvemag(sigma, Beta, M0, M_plot[i]))
		else:
			for i in range(0, len(M_plot)):
				output_array.append(np.log10(convolvemag(sigma, Beta, M0, M_plot[i])))

	# Function: Flag 4 - Schechter function + mag units
	if flag_type == 4:
		if flag_log == 0:
			for i in range(0, len(M_plot)):
				output_array.append(convolvemagnoerr(Beta, M0, M_plot[i]))
		else:
			for i in range(0, len(M_plot)):
				output_array.append(np.log10(convolvemagnoerr(Beta, M0, M_plot[i])))

	# Function: Flag 5 - Schechter function + log form
	if flag_type == 5:
		if flag_log == 0:
			for i in range(0, len(M_plot)):
				output_array.append(original2(Beta, np.log10(M0), np.log10(M_plot[i])))
		else:
			for i in range(0, len(M_plot)):
				output_array.append(np.log10(original2(Beta, np.log10(M0), np.log10(M_plot[i]))))

	# Function: Flag 6 - Schechter function + Gaussian function with sigma width + log form
	if flag_type == 6:
		if flag_log == 0:
			for i in range(0, len(M_plot)):
				output_array.append(integrateconvolve2(sigma, Beta, np.log10(M0), np.log10(M_plot[i])))
		else:
			for i in range(0, len(M_plot)):
				output_array.append(np.log10(integrateconvolve2(sigma, Beta, np.log10(M0), np.log10(M_plot[i]))))

	# Function: Flag 7 - Step Function
	if flag_type == 7:
		if flag_log == 0:
			for i in range(0, len(M_plot)):
				output_array.append(integratestep(sigma, Beta, M0, M_plot[i]))
		else:
			for i in range(0, len(M_plot)):
				output_array.append(np.log10(integratestep(sigma, Beta, M0, M_plot[i])))

	# Function: Flag 8 - Step Function (fixed)
	if flag_type == 8:
		if flag_log == 0:
			for i in range(0, len(M_plot)):
				output_array.append(integratestep2(sigma, Beta, M0, M_plot[i]))
		else:
			for i in range(0, len(M_plot)):
				output_array.append(np.log10(integratestep2(sigma, Beta, M0, M_plot[i])))

	# Function: Flag 9 - Step Function (log)
	if flag_type == 9:
		if flag_log == 0:
			for i in range(0, len(M_plot)):
				output_array.append(integratestep3(sigma, Beta, M0, M_plot[i]))
		else:
			for i in range(0, len(M_plot)):
				output_array.append(np.log10(integratestep3(sigma, Beta, M0, M_plot[i])))

	# Function: Flag 10 - Test
	if flag_type == 10:
		if flag_log == 0:
			for i in range(0, len(M_plot)):
				output_array.append(integrateguass(sigma, Beta, M0, M_plot[i]))
		else:
			for i in range(0, len(M_plot)):
				output_array.append(np.log10(integrateguass(sigma, Beta, M0, M_plot[i])))

	# Function: Flag 11 - Test
	if flag_type == 11:
		if flag_log == 0:
			for i in range(0, len(M_plot)):
				output_array.append(outputmassfunction11(Beta, M0, M_plot[i]))
		else:
			for i in range(0, len(M_plot)):
				output_array.append(np.log10(outputmassfunction11(Beta, M0, M_plot[i])))

	# Function: Flag 12 - Test
	if flag_type == 12:
		if flag_log == 0:
			for i in range(0, len(M_plot)):
				output_array.append(outputmassfunction12(sigma, Beta, M0, M_plot[i]))
		else:
			for i in range(0, len(M_plot)):
				output_array.append(np.log10(outputmassfunction12(sigma, Beta, M0, M_plot[i])))


	return np.array(output_array)

#------------------------------------------------------------------------------
###
# (6) Code Snippets (Clumps)
###

def outputplotsymbol(flag):

	'''
	Function: Define Plot Symbols
	'''

	if flag == 1:
		plotsym = 'g^'
		plotcolour = 'g'
		plotsymbol = '^'
	else:
		plotsym = 'g^'
		plotcolour = 'g'
		plotsymbol = '^'
	return plotsym, plotcolour, plotsymbol

def plotclumps_histogram_b(array1, complimits_val, linetype, labelshape, labelcolour, textlabel, norm_val = 0, sortindex = -1, massindex = 4, equal = False):

	'''
	Function: 
	'''

	if sortindex > 0:
		arraytemp = array1[array1[:,sortindex] == 1]
		array1 = arraytemp

	mass_bins_log = np.power(10, np.linspace(1, 6, num = 21))
	array_plot = np.power(10, np.linspace(np.log10(np.min(mass_bins_log)), np.log10(np.max(mass_bins_log)) + 1))

	if equal == False:
		n, bins, bins_width, bins_centre, n_fit, bins_fit, n_dM, n_fit_dM, n_dlogM, n_fit_dlogM, ncum, ncum_fit, n_fit_err, n_fit_dM_err, n_fit_dlogM_err = makearrayhist(array1, mass_bins_log, 1E1, massindex = massindex)
		plt.errorbar(bins_fit, n_fit_dM * np.power(10, norm_val), marker = labelshape, linestyle = 'None', color = labelcolour, markeredgecolor = labelcolour, markerfacecolor = labelcolour, yerr = n_fit_dM_err * np.power(10, norm_val), markersize = 20, label = textlabel)

	else:
		n, bins, bins_width, bins_centre, n_fit, bins_fit, n_dM, n_fit_dM, n_dlogM, n_fit_dlogM, ncum, ncum_fit, n_fit_err, n_fit_dM_err, n_fit_dlogM_err = makearrayhistequal(array1, 1E1, -1, massindex = massindex, numgal_bin_in = 3)
		plt.errorbar(bins_fit, n_fit_dM * np.power(10, norm_val), marker = labelshape, linestyle = 'None', color = labelcolour, markeredgecolor = labelcolour, markerfacecolor = labelcolour, yerr = n_fit_dM_err * np.power(10, norm_val), markersize = 20, label = textlabel, alpha = 0.5)

	popt_truncatedpowerlaw, pcov_truncatedpowerlaw, popt_simplepowerlaw, pcov_simplepowerlaw, powerlawchisq, popt_schechter, pcov_schechter, schechterchisq = curve_fit3(bins_fit, n_fit_dM * np.power(10, norm_val), [1E2, 1E8], complimits_val, 1, 0, n_fit_dM_err * np.power(10, norm_val))
	plt.plot(array_plot, simplepowerlaw(array_plot, *popt_simplepowerlaw), linetype, label = r'$\beta$ = {:.2f} $\pm$ {:.2f}'.format(popt_simplepowerlaw[1], np.sqrt(abs(pcov_simplepowerlaw[1][1]))))
	
	plt.plot([np.nanmax(complimits_val), np.nanmax(complimits_val)], [1E-10, 1E10], 'k--')
	plt.legend()

	return 0

def plotclumps_histogram_blike(array1, complimits_val, beta, linetype, labelshape, labelcolour, textlabel, norm_val = 0, sortindex = -1, massindex = 4):

	'''
	Function: 
	'''

	if sortindex > 0:
		arraytemp = array1[array1[:,sortindex] == 1]
		array1 = arraytemp

	mass_bins_log = np.power(10, np.linspace(1, 6, num = 21))
	array_plot = np.power(10, np.linspace(np.log10(np.min(mass_bins_log)), np.log10(np.max(mass_bins_log)) + 1))

	n, bins, bins_width, bins_centre, n_fit, bins_fit, n_dM, n_fit_dM, n_dlogM, n_fit_dlogM, ncum, ncum_fit, n_fit_err, n_fit_dM_err, n_fit_dlogM_err = makearrayhist(array1, mass_bins_log, 1E1, massindex = massindex)
	plt.errorbar(bins_fit, n_fit_dM * np.power(10, norm_val), marker = labelshape, linestyle = 'None', color = labelcolour, markeredgecolor = labelcolour, markerfacecolor = labelcolour, yerr = n_fit_dM_err * np.power(10, norm_val), markersize = 20, label = textlabel)

	def simplepowerlawwithbeta(M, M0):

		return (np.power((M / M0), beta))

	def simplepowerlaw_log_beta(logM, M0):

		return (beta * logM) - (beta * np.log10(M0))

	array1_tofit = bins_fit[bins_fit > complimits_val]
	array2_tofit = n_fit_dM[bins_fit > complimits_val] * np.power(10, norm_val)
	array2_tofit_err = n_fit_dM_err[bins_fit > complimits_val] * np.power(10, norm_val)

	popt_simplepowerlawwithbeta, pcov_simplepowerlawwithbeta = optimize.curve_fit(simplepowerlaw_log_beta, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 100000, absolute_sigma = True, sigma = 0.434 * (array2_tofit_err / array2_tofit))
	print(beta, popt_simplepowerlawwithbeta)

	plt.plot(array_plot, simplepowerlawwithbeta(array_plot, *popt_simplepowerlawwithbeta), linetype)
	
	plt.plot([np.nanmax(complimits_val), np.nanmax(complimits_val)], [1E-10, 1E10], 'k--')
	plt.legend()

	return 0

def plotclumps_histogram2(array1, complimits_val):

	'''
	Function: 
	'''

	array2 = array1[array1[:,8] == 1]
	array3 = array1[array1[:,9] == 2]

	mass_bins_log = np.power(10, np.linspace(1, 6, num = 21))
	array_plot = np.power(10, np.linspace(np.log10(np.min(mass_bins_log)), np.log10(np.max(mass_bins_log)) + 1))

	n, bins, bins_width, bins_centre, n_fit, bins_fit, n_dM, n_fit_dM, n_dlogM, n_fit_dlogM, ncum, ncum_fit, n_fit_err, n_fit_dM_err, n_fit_dlogM_err = makearrayhist(array1, mass_bins_log, 1E1)
	n2, bins2, bins2_width, bins2_centre, n2_fit, bins2_fit, n2_dM, n2_fit_dM, n2_dlogM, n2_fit_dlogM, n2cum, n2cum_fit, n2_fit_err, n2_fit_dM_err, n2_fit_dlogM_err = makearrayhist(array2, mass_bins_log, 1E1)
	n3, bins3, bins3_width, bins3_centre, n3_fit, bins3_fit, n3_dM, n3_fit_dM, n3_dlogM, n3_fit_dlogM, n3cum, n3cum_fit, n3_fit_err, n3_fit_dM_err, n3_fit_dlogM_err = makearrayhist(array3, mass_bins_log, 1E1)

	plt.errorbar(bins2_fit, n2_fit_dM, marker = '^', linestyle = 'None', markeredgecolor = 'green', markerfacecolor = 'green', yerr = n2_fit_dM_err, markersize = 24, label = r'Leaves')
	plt.errorbar(bins_fit, n_fit_dM, marker = 's', linestyle = 'None', markeredgecolor = 'brown', markerfacecolor = 'brown', yerr = n_fit_dM_err, markersize = 20, label = r'All Structures')
	plt.errorbar(bins3_fit, n3_fit_dM, marker = '^', linestyle = 'None', markeredgecolor = 'green', markerfacecolor = 'blue', yerr = n3_fit_dM_err, markersize = 34, label = r'Branches')
	
	popt_truncatedpowerlaw, pcov_truncatedpowerlaw, popt_simplepowerlaw, pcov_simplepowerlaw, powerlawchisq, popt_schechter, pcov_schechter, schechterchisq = curve_fit3(bins_fit, n_fit_dM, [1E2, 1E8], complimits_val, 1, 0, n_fit_dM_err)
	popt_truncatedpowerlaw2, pcov_truncatedpowerlaw2, popt_simplepowerlaw2, pcov_simplepowerlaw2, powerlawchisq2, popt_schechter2, pcov_schechter2, schechterchisq2 = curve_fit3(bins2_fit, n2_fit_dM, [1E2, 1E8], complimits_val, 1, 0, n2_fit_dM_err)
	popt_truncatedpowerlaw3, pcov_truncatedpowerlaw3, popt_simplepowerlaw3, pcov_simplepowerlaw3, powerlawchisq3, popt_schechter3, pcov_schechter3, schechterchisq3 = curve_fit3(bins3_fit, n3_fit_dM, [1E2, 1E8], complimits_val, 1, 0, n3_fit_dM_err)

	plt.plot(array_plot, simplepowerlaw(array_plot, *popt_simplepowerlaw2), 'g:', label = r'$\beta$ = {:.2f} $\pm$ {:.2f}'.format(popt_simplepowerlaw2[1], np.sqrt(abs(pcov_simplepowerlaw2[1][1]))))
	plt.plot(array_plot, simplepowerlaw(array_plot, *popt_simplepowerlaw), 'r--', label = r'$\beta$ = {:.2f} $\pm$ {:.2f}'.format(popt_simplepowerlaw[1], np.sqrt(abs(pcov_simplepowerlaw[1][1]))))
	
	plt.plot([np.nanmax(complimits_val), np.nanmax(complimits_val)], [1E-10, 1E10], 'k--')

	plt.legend()

	return 0

def plotclumps_cdf(array1, array2, complimits_val):

	'''
	Function: 
	'''

	def makecdf(array, complimits_val):
		
		sorted_data = np.sort(array[:,4])
		xdata = np.concatenate([sorted_data[::-1]])
		ydata = np.arange(sorted_data.size) + 1
		
		# Filter array to those > complimit, output results
		filt_array = np.where(sorted_data >= complimits_val)
		data_lim = sorted_data[filt_array]
		sorted_data_lim = np.sort(data_lim)

		# Output results
		xdata_lim = np.concatenate([sorted_data_lim[::-1]])
		ydata_lim = np.arange(sorted_data_lim.size) + 1
		xdata_lim_float = xdata_lim.astype(float)
		ydata_lim_float = ydata_lim.astype(float)
		
		print('[{}, {}]'.format(np.nanmin(xdata_lim_float), np.nanmax(xdata_lim_float)))
		print('[{}, {}]'.format(np.nanmax(ydata_lim_float), np.nanmin(ydata_lim_float)))

		# Print result to screen
		print('CDF Function Results {:.2e}:'.format(complimits_val))
		print('[Len: {}, Min: {:.2e}, Max:{:.2e}]'.format(len(xdata), np.nanmin(xdata), np.nanmax(xdata)))
		print('[Len: {}, Min: {:.2e}, Max:{:.2e}]'.format(len(xdata_lim_float), np.nanmin(xdata_lim_float), np.nanmax(xdata_lim_float)))

		return xdata, ydata

	xdata1, ydata1 = makecdf(array1, complimits_val)
	xdata2, ydata2 = makecdf(array2, complimits_val)
		
	plt.step(xdata1, ydata1, label = 'All')
	plt.step(xdata2, ydata2, color = 'brown', linestyle = '-.', label = 'Leaves')
	plt.plot([np.nanmax(complimits_val), np.nanmax(complimits_val)], [1E-10, 1E10], 'k--')

	return 0

def plotclumps_mr_clouds(axes, np_Full_array, np_array, np_NL_array, radiuslimit, plotcolour, plotsymbol, plotsym, legendname, masslimit = 0):

	'''
	Function: 
	'''	

	np_L_array = np_array[np_array[:,8] == 1]
	np_Full_L_array = np_Full_array[np_Full_array[:,8] == 1]

	axes.axhline(y = radiuslimit, color = 'k', linestyle = '--')
	if masslimit != 0:

		np_L_AL_array = np_L_array[np_L_array[:,4] > masslimit]
		np_NL_AL_array = np_NL_array[np_NL_array[:,4] > masslimit]
		np_AL_array = np_array[np_array[:,4] > masslimit]

		axes.axvline(x = masslimit, color = 'k', linestyle = '--')
		axes.plot(np_Full_L_array[:,4], np_Full_L_array[:,6], linestyle = 'None', markerfacecolor = 'None', markeredgecolor = plotcolour, marker = plotsymbol, markersize = 14, alpha = 0.9)
		axes.errorbar(np_L_AL_array[:,4], np_L_AL_array[:,6], xerr = np_L_AL_array[:,5], yerr = np_L_AL_array[:,7], linestyle = 'None', color = 'green', marker = plotsymbol, markersize = 14, alpha = 0.9)
		axes.plot(np_NL_array[:,4], np_NL_array[:,6], linestyle = 'None', markerfacecolor = 'None', markeredgecolor = 'brown', marker = 's', markersize = 10, alpha = 0.9)
		axes.errorbar(np_NL_AL_array[:,4], np_NL_AL_array[:,6], xerr = np_NL_AL_array[:,5], yerr = np_NL_AL_array[:,7], linestyle = 'None', color = 'brown', markerfacecolor = 'brown', markeredgecolor = 'brown', marker = 's', markersize = 10, alpha = 0.5)
		curvefit_powerlaw_clumps(np_AL_array[:,4], np_AL_array[:,6], axes, legendname = legendname)
	else:
		axes.plot(np_Full_L_array[:,4], np_Full_L_array[:,6], linestyle = 'None', markerfacecolor = 'None', markeredgecolor = plotcolour, marker = plotsymbol, markersize = 14, alpha = 0.9)
		axes.errorbar(np_L_array[:,4], np_L_array[:,6], xerr = np_L_array[:,5], yerr = np_L_array[:,7], linestyle = 'None', color = 'green', marker = plotsymbol, markersize = 14, alpha = 0.9)
		axes.errorbar(np_NL_array[:,4], np_NL_array[:,6], xerr = np_NL_array[:,5], yerr = np_NL_array[:,7], linestyle = 'None', color = 'brown', markerfacecolor = 'brown', markeredgecolor = 'brown', marker = 's', markersize = 10, alpha = 0.5)
		curvefit_powerlaw_clumps(np_array[:,4], np_array[:,6], axes, legendname = legendname)

def plotclumps_mr_leaves(axes, np_Full_array, np_array, radiuslimit, plotcolour, plotsymbol, plotsym, legendname, masslimit = 0, radiuscut = 1):

	'''
	Function: 
	'''	

	np_L_array = np_array[np_array[:,8] == 1]
	np_Full_L_array = np_Full_array[np_Full_array[:,8] == 1]

	if radiuscut == 1:
		axes.axhline(y = radiuslimit, color = 'k', linestyle = '--')
	
		if masslimit != 0:

			np_L_AL_array = np_L_array[np_L_array[:,4] > masslimit]
			np_AL_array = np_array[np_array[:,4] > masslimit]

			axes.axvline(x = masslimit, color = 'k', linestyle = '--')
			axes.plot(np_Full_L_array[:,4], np_Full_L_array[:,6], linestyle = 'None', markerfacecolor = 'None', markeredgecolor = plotcolour, marker = plotsymbol, markersize = 14, alpha = 0.9)
			axes.errorbar(np_L_AL_array[:,4], np_L_AL_array[:,6], xerr = np_L_AL_array[:,5], yerr = np_L_AL_array[:,7], linestyle = 'None', color = 'green', marker = plotsymbol, markersize = 14, alpha = 0.9)
			curvefit_powerlaw_clumps(np_L_AL_array[:,4], np_L_AL_array[:,6], axes, legendname = legendname)
		else:
			axes.plot(np_Full_L_array[:,4], np_Full_L_array[:,6], linestyle = 'None', markerfacecolor = 'None', markeredgecolor = plotcolour, marker = plotsymbol, markersize = 14, alpha = 0.9)
			axes.errorbar(np_L_array[:,4], np_L_array[:,6], xerr = np_L_array[:,5], yerr = np_L_array[:,7], linestyle = 'None', color = 'green', marker = plotsymbol, markersize = 14, alpha = 0.9)
			curvefit_powerlaw_clumps(np_L_array[:,4], np_L_array[:,6], axes, legendname = legendname)
	else:
		axes.errorbar(np_Full_L_array[:,4], np_Full_L_array[:,6], xerr = np_Full_L_array[:,5], yerr = np_Full_L_array[:,7], linestyle = 'None', color = 'green', marker = plotsymbol, markersize = 14, alpha = 0.9)
		curvefit_powerlaw_clumps(np_Full_L_array[:,4], np_Full_L_array[:,6], axes, legendname = legendname)

def curvefit_powerlaw_clumps(array1, array2, plotaxes, flag_mr = 1, legendname = ''):

	'''
	Function: Runs simplified fitting routine for power law functions (see clump analysis)
	'''

	array_plot = np.power(10, np.linspace(-10, 10))
	array1_tofit = np.log10(array1.astype('float'))
	array2_tofit = np.log10(array2.astype('float'))
	if flag_mr == 1:
		popt_simplepowerlaw, pcov_simplepowerlaw = optimize.curve_fit(simplepowerlaw_log, array1_tofit, array2_tofit, maxfev = 100000, p0 = [100, 0.25])
	else:
		popt_simplepowerlaw, pcov_simplepowerlaw = optimize.curve_fit(simplepowerlaw_log, array1_tofit, array2_tofit, maxfev = 100000, p0 = [1, 1])
	plotaxes.plot(array_plot, simplepowerlaw(array_plot, *popt_simplepowerlaw), 'k--', label = r'$\alpha$ = {:.2f} $\pm$ {:.2f}'.format(popt_simplepowerlaw[1], np.sqrt(abs(pcov_simplepowerlaw[1][1]))))
	if legendname == '':
		plotaxes.legend(fontsize = 32)
	else:
		plotaxes.legend(title = legendname, fontsize = 32)

	return popt_simplepowerlaw, pcov_simplepowerlaw

#------------------------------------------------------------------------------
###
# (7) Code Snippets (Extreme)
###

def extremegalarrays(galarray, galname):

	'''
	Function: Print maximum mass object by normal age range (with additional at 410 Myr)
	'''

	galarray1 = galarray[galarray[:,6] <= 1E7]
	galarray2_temp = galarray[galarray[:,6] > 1E7]
	galarray2 = galarray2_temp[galarray2_temp[:,6] <= 1E8]
	galarray3_temp = galarray[galarray[:,6] > 1E8]
	galarray3 = galarray3_temp[galarray3_temp[:,6] <= 4 * 1E8]
	galarray4_temp = galarray[galarray[:,6] > 1E8]
	galarray4 = galarray4_temp[galarray4_temp[:,6] <= 4.1 * 1E8]

	string1 = galname + ' (< 10 Myr)'
	string2 = galname + ' (10 - 100 Myr)'
	string3 = galname + ' (100 - 400 Myr)'
	string4 = galname + ' (100 - 401 Myr)'

	print('In {}, {} in {}, {} in {}, {} in {}, and {} in {}'.format(galname, len(galarray1), string1, len(galarray2), string2, len(galarray3), string3, len(galarray4), string4))

	return galarray1, galarray2, galarray3, galarray4, string1, string2, string3, string4

def extremegalarrays_alt(galarray, galname):

	'''
	Function: Print maximum mass object by age range (with additional at 410 Myr)
	'''

	galarray1_temp = galarray[galarray[:,6] > 1E8]
	galarray1 = galarray1_temp[galarray1_temp[:,6] <= 1E9]
	galarray2_temp = galarray[galarray[:,6] > np.power(10, 8.25)]
	galarray2 = galarray2_temp[galarray2_temp[:,6] <= np.power(10, 8.7)]
	galarray3_temp = galarray[galarray[:,6] > 1E8]
	galarray3 = galarray3_temp[galarray3_temp[:,6] <= 4 * 1E8]
	galarray4_temp = galarray[galarray[:,6] > 1E8]
	galarray4 = galarray4_temp[galarray4_temp[:,6] <= 4.1 * 1E8]

	string1 = galname + ' (0.1 - 1 Gyr)'
	string2 = galname + ' (0.18 - 0.50 Gyr)'
	string3 = galname + ' (0.1 - 0.4 Gyr)'
	string4 = galname + ' (0.1 - 0.41 Gyr)'

	print('In {}, {} in {}, {} in {}, {} in {}, and {} in {}'.format(galname, len(galarray1), string1, len(galarray2), string2, len(galarray3), string3, len(galarray4), string4))

	return galarray1, galarray2, galarray3, galarray4, string1, string2, string3, string4

def phangsgalarrays(galarray, galname):

	'''
	Function: Print maximum mass object by age range
	'''

	galarray1_temp = galarray[galarray[:,6] > 1E8]
	galarray1 = galarray1_temp[galarray1_temp[:,6] <= 1E9]
	galarray2_temp = galarray[galarray[:,6] > np.power(10, 8.25)]
	galarray2 = galarray2_temp[galarray2_temp[:,6] <= np.power(10, 8.7)]
	galarray3_temp = galarray[galarray[:,6] > 1E8]
	galarray3 = galarray3_temp[galarray3_temp[:,6] <= 4 * 1E8]

	string1 = galname + ' (0.1 - 1 Gyr)'
	string2 = galname + ' (0.18 - 0.50 Gyr)'
	string3 = galname + ' (0.1 - 0.4 Gyr)'

	print('In {}, {} in {}, {} in {}, and {} in {}'.format(galname, len(galarray1), string1, len(galarray2), string2, len(galarray3), string3))
	if '1559' in galname:
		galarray3_max = galarray3[np.argmax(galarray3[:,4])]
		print('Note: Max in {}, array: {}'.format(string3, galarray3_max))

	return galarray1, galarray2, galarray3, string1, string2, string3

def sumupvmag(galarray, galname, galindex, galdistmnod):

	'''
	Function: 
	'''

	vmag = galarray[:,galindex] - galdistmnod
	vlum = np.power(10, 0.4 * -vmag)
	vlum_sum = np.sum(vlum)
	print('{} - {:.2e} ({:.2f})'.format(galname, vlum_sum, np.log10(vlum_sum)))

def makemassagebox(x1, y1, x2, y2):

	'''
	Function: 
	'''

	plt.plot([x1, x1], [y1, 1E10], 'k--')
	plt.plot([x1, x2], [y1, y2], 'k--')
	plt.plot([x2, x2], [y2, 1E10], 'k--')

#------------------------------------------------------------------------------
###
# (8) Code Snippets (Curvefit)
###
 
def curvefit_bootstrap(array1, age1, age2, mass_lim, numclusters):

	'''
	Function: Perform Bootstrap Routine (with errors)
	'''

	# Set basic parameters
	mass_bins_log = np.power(10, np.linspace(2, 9, num = 29))

	# Run bootstrap method
	print('Run Bootstrap Method')
	for i in range(0, bootstrapnumsample):
		np.random.shuffle(array1)
		requiredcriteria = np.cumsum(np.where((array1[:,6] > age1) & (array1[:,6] <= age2) & (array1[:,4] > mass_lim), 1, 0))
		requiredcriteriaindex = np.nanmin(np.where(requiredcriteria == numclusters)[0])
		inputarray = array1[:(requiredcriteriaindex + 1)]

		# Create cumulative function
		sorted_data_lim = np.sort(filter(lambda x: x >= mass_lim, inputarray[:,4]))
		xdata_lim = np.concatenate([sorted_data_lim[::-1]])
		ydata_lim = np.arange(sorted_data_lim.size) + 1
		xdata_lim_float = xdata_lim.astype(float)
		ydata_lim_float = ydata_lim.astype(float)
		popt_truncatedpowerlaw, pcov_truncatedpowerlaw, popt_simplepowerlaw, pcov_simplepowerlaw, popt_schechter, pcov_schechter = curve_fit3(xdata_lim_float, ydata_lim_float, [1E1, 1E9], 1E4, -2, 0)

		# Add 2 errors from a Guassian function
		inputarray_log = np.log10(inputarray[:,4].astype('float64'))
		###
		error1 = np.random.normal(loc = 0, scale = 0.30, size = len(inputarray_log))
		inputarray_error1 = np.power(10, inputarray_log + error1)
		sorted_data_lim_error1 = np.sort(filter(lambda x: x >= mass_lim, inputarray_error1))
		xdata_lim_error1 = np.concatenate([sorted_data_lim_error1[::-1]])
		ydata_lim_error1 = np.arange(sorted_data_lim_error1.size) + 1
		xdata_lim_error1_float = xdata_lim_error1.astype(float)
		ydata_lim_error1_float = ydata_lim_error1.astype(float)
		###
		error2 = np.random.normal(loc = 0, scale = 0.60, size = len(inputarray_log))
		inputarray_error2 = np.power(10, inputarray_log + error2)
		sorted_data_lim_error2 = np.sort(filter(lambda x: x >= mass_lim, inputarray_error2))
		xdata_lim_error2 = np.concatenate([sorted_data_lim_error2[::-1]])
		ydata_lim_error2 = np.arange(sorted_data_lim_error2.size) + 1
		xdata_lim_error2_float = xdata_lim_error2.astype(float)
		ydata_lim_error2_float = ydata_lim_error2.astype(float)
		###
		popt_truncatedpowerlaw_error1, pcov_truncatedpowerlaw_error1, popt_simplepowerlaw_error1, pcov_simplepowerlaw_error1, popt_schechter_error1, pcov_schechter_error1 = curve_fit3(xdata_lim_error1_float, ydata_lim_error1_float, [1E1, 1E9], 1E4, -2, 0)
		popt_truncatedpowerlaw_error2, pcov_truncatedpowerlaw_error2, popt_simplepowerlaw_error2, pcov_simplepowerlaw_error2, popt_schechter_error2, pcov_schechter_error2 = curve_fit3(xdata_lim_error2_float, ydata_lim_error2_float, [1E1, 1E9], 1E4, -2, 0)

		# Output data to array
		if i == 0:
			popt_full = np.asarray(popt_truncatedpowerlaw)
			lenarray_full = np.asarray(len(inputarray))
			popt_full_error1 = np.asarray(popt_truncatedpowerlaw_error1)
			popt_full_error2 = np.asarray(popt_truncatedpowerlaw_error2)
		else:
			popt_full = np.vstack([popt_full, np.asarray(popt_truncatedpowerlaw)])
			lenarray_full = np.vstack([lenarray_full, np.asarray(len(inputarray))])
			popt_full_error1 = np.vstack([popt_full_error1, np.asarray(popt_truncatedpowerlaw_error1)])
			popt_full_error2 = np.vstack([popt_full_error2, np.asarray(popt_truncatedpowerlaw_error2)])

		if i % (bootstrapnumsample / 25) == 0:
			print('{} done out of {}'.format(i, bootstrapnumsample))

	# Output results to console
	print('Length of Sample: {}'.format(len(popt_full)))
	print('Average Lengths: {} +/- {:.2f} [{} to {}]'.format(np.mean(lenarray_full), np.std(lenarray_full), np.nanmin(lenarray_full), np.nanmax(lenarray_full)))
	print('Average Values: {:.2f} +\- {:.2f}, {:.2e} +\- {:.2e}, {:.2f} +\- {:.2f}'.format(np.mean(popt_full[:,0]), np.std(popt_full[:,0]), np.mean(popt_full[:,1]), np.std(popt_full[:,1]), np.mean(popt_full[:,2]), np.std(popt_full[:,2])))
	print('Average Values log: {:.2f} +\- {:.2f}, {:.2f} +\- {:.2f}'.format(np.log10(np.mean(popt_full[:,0])), 0.434 * np.std(popt_full[:,0]) / np.mean(popt_full[:,0]), np.log10(np.mean(popt_full[:,1])), 0.434 * np.std(popt_full[:,1]) / np.mean(popt_full[:,1])))
	print(popt_full[:,0])
	print('Min and Max for N: {:.2e} to {:.2e}'.format(np.nanmin(popt_full[:,0]), np.nanmax(popt_full[:,0])))
	print('Min and Max for M: {:.2e} to {:.2e}'.format(np.nanmin(popt_full[:,1]), np.nanmax(popt_full[:,1])))
	print('Percentage above 1 for N: {} out of {}'.format((popt_full[:,0] > 1).sum(), len(popt_full[:,0])))
	print('Percentage above 5 for N: {} out of {}'.format((popt_full[:,0] > 5).sum(), len(popt_full[:,0])))
	print('Percentage above 10 for N: {} out of {}'.format((popt_full[:,0] > 10).sum(), len(popt_full[:,0])))

	return popt_full, popt_full_error1, popt_full_error2

def curvefit_bootstrap2(array1, age1, age2, mass_lim, numclusters, flag_idl):

	'''
	Function: Perform bootstrap procedure on array with selection effects
	'''

	# Set basic parameters
	bootstrapnumsample = 100
	mass_bins_log = np.power(10, np.linspace(2, 9, num = 29))

	# Start bootstrap method
	print('Run Bootstrap Method')
	for i in range(0, bootstrapnumsample):

		# Shuffle arrays
		np.random.shuffle(array1)
		requiredcriteriafilter = np.where((array1[:,6] > age1) & (array1[:,6] <= age2) & (array1[:,4] > mass_lim), 1, 0)
		requiredcriteria = np.cumsum(requiredcriteriafilter)
		inputarray_tmp = array1[requiredcriteriafilter == 1]
		inputarray = inputarray_tmp[:(numclusters)]

		# Check length to match original
		if len(inputarray) != numclusters:
			print('Length mismatch')
			print(len(inputarray), numclusters)

		# Method 1: Run python fitting routine
		sorted_data_lim = np.sort(filter(lambda x: x >= mass_lim, inputarray[:,4]))
		xdata_lim = np.concatenate([sorted_data_lim[::-1]])
		ydata_lim = np.arange(sorted_data_lim.size) + 1
		xdata_lim_float = xdata_lim.astype(float)
		ydata_lim_float = ydata_lim.astype(float)
		try:
			if np.nanmax(xdata_lim_float) > 1E8:
				popt_simplepowerlaw, pcov_simplepowerlaw = optimize.curve_fit(simplepowerlaw_log, np.log10(xdata_lim_float), np.log10(ydata_lim_float), maxfev = 100000, bounds = [[1E2, -5], [1E12, 0]], absolute_sigma = True, sigma = 0.434 / np.sqrt(ydata_lim_float), p0 = [1E8, -2])
			else:
				popt_simplepowerlaw, pcov_simplepowerlaw = optimize.curve_fit(simplepowerlaw_log, np.log10(xdata_lim_float), np.log10(ydata_lim_float), maxfev = 100000, bounds = [[1E2, -5], [1E12, 0]], absolute_sigma = True, sigma = 0.434 / np.sqrt(ydata_lim_float), p0 = [1E5, -2])
		except ValueError:
			popt_simplepowerlaw, pcov_simplepowerlaw = optimize.curve_fit(simplepowerlaw_log, np.log10(xdata_lim_float), np.log10(ydata_lim_float), maxfev = 100000, bounds = [[1E2, -5], [1E12, 0]], absolute_sigma = True, sigma = 0.434 / np.sqrt(ydata_lim_float))

		try:
			popt_truncatedpowerlaw, pcov_truncatedpowerlaw = optimize.curve_fit(truncatedpowerlaw_log, np.log10(xdata_lim_float), np.log10(ydata_lim_float), maxfev = 100000, bounds = [[0, 0, popt_simplepowerlaw[1] - 0.5], [1E5, 1E12, popt_simplepowerlaw[1] + 0.5]], absolute_sigma = True, sigma = 0.434 / np.sqrt(ydata_lim_float), p0 = [1., 0.5*np.nanmax(xdata_lim_float), popt_simplepowerlaw[1]])
		except ValueError:
			try:
				popt_truncatedpowerlaw, pcov_truncatedpowerlaw = optimize.curve_fit(truncatedpowerlaw_log, np.log10(xdata_lim_float), np.log10(ydata_lim_float), maxfev = 100000, bounds = [[0, 0, popt_simplepowerlaw[1] - 0.5], [1E5, 1E12, popt_simplepowerlaw[1] + 0.5]], absolute_sigma = True, sigma = 0.434 / np.sqrt(ydata_lim_float), p0 = [0.1, 1E5, popt_simplepowerlaw[1]])
			except ValueError:
				popt_truncatedpowerlaw, pcov_truncatedpowerlaw = optimize.curve_fit(truncatedpowerlaw_log, np.log10(xdata_lim_float), np.log10(ydata_lim_float), maxfev = 1000000, bounds = [[0, 0, popt_simplepowerlaw[1] - 0.5], [1E5, 1E12, popt_simplepowerlaw[1] + 0.5]], absolute_sigma = True, sigma = 0.434 / np.sqrt(ydata_lim_float))

		# Method 2: Run IDL fitting routine
		if flag_idl == 2:

			from idlpy import IDL
			fit = IDL.mspecfit(np.array(xdata_lim_float, dtype = np.float32), 1E-6*np.array(xdata_lim_float, dtype = np.float32))
		else:
			fit = [-1, -1, -1]

		# Method 3: Run updated python fitting routine
		nequal, binsequal, binsequal_width, binsequal_centre, nequal_fit, binsequal_fit, nequal_dM, nequal_fit_dM, nequal_dlogM, nequal_fit_dlogM, nequalcum, nequalcum_fit, nequal_fit_err, nequal_fit_dM_err, nequal_fit_dlogM_err = makearrayhistequal(inputarray, mass_lim, 0)
		try:
			if np.nanmax(xdata_lim_float) > 1E8:
				popt_simplepowerlaw, pcov_simplepowerlaw = optimize.curve_fit(simplepowerlaw_log, np.log10(binsequal_fit), np.log10(nequal_fit_dM), maxfev = 100000, bounds = [[1E2, -5], [1E12, 0]], absolute_sigma = True, sigma = 0.434 * (nequal_fit_dM_err / nequal_fit_dM), p0 = [1E8, -2])
			else:
				popt_simplepowerlaw, pcov_simplepowerlaw = optimize.curve_fit(simplepowerlaw_log, np.log10(binsequal_fit), np.log10(nequal_fit_dM), maxfev = 100000, bounds = [[1E2, -5], [1E12, 0]], absolute_sigma = True, sigma = 0.434 * (nequal_fit_dM_err / nequal_fit_dM), p0 = [1E5, -2])
		except ValueError:
			popt_simplepowerlaw, pcov_simplepowerlaw = optimize.curve_fit(simplepowerlaw_log, np.log10(binsequal_fit), np.log10(nequal_fit_dM), maxfev = 100000, bounds = [[1E2, -5], [1E12, 0]], absolute_sigma = True, sigma = 0.434 * (nequal_fit_dM_err / nequal_fit_dM))

		try:
			popt_schechter, pcov_schechter = optimize.curve_fit(schechter_log, np.log10(binsequal_fit), np.log10(nequal_fit_dM), maxfev = 100000, bounds = [[0, 1E2, -5], [1E12, 1E12, 0]], absolute_sigma = True, sigma = 0.434 * (nequal_fit_dM_err / nequal_fit_dM), p0 = [1, 0.5 * np.nanmax(binsequal_fit), popt_simplepowerlaw[1]])
		except ValueError:
			try:
				popt_schechter, pcov_schechter = optimize.curve_fit(schechter_log, np.log10(binsequal_fit), np.log10(nequal_fit_dM), maxfev = 100000, bounds = [[0, 1E2, -5], [1E12, 1E12, 0]], absolute_sigma = True, sigma = 0.434 * (nequal_fit_dM_err / nequal_fit_dM), p0 = [1, 1E5, popt_simplepowerlaw[1]])
			except ValueError:
				popt_schechter, pcov_schechter = optimize.curve_fit(schechter_log, np.log10(binsequal_fit), np.log10(nequal_fit_dM), maxfev = 100000, bounds = [[0, 1E2, -5], [1E12, 1E12, 0]], absolute_sigma = True, sigma = 0.434 * (nequal_fit_dM_err / nequal_fit_dM))

		simplepowerlaw_out = simplepowerlaw(binsequal_fit, *popt_simplepowerlaw)
		schechter_out = schechter(binsequal_fit, *popt_schechter)

		# Output chi2 parameters
		powerlawchisq = reducedchisq(simplepowerlaw_out, nequal_fit_dM, 2, sd = nequal_fit_dM_err)
		schechterchisq = reducedchisq(schechter_out, nequal_fit_dM, 3, sd = nequal_fit_dM_err)
		chi = [powerlawchisq, schechterchisq]
		print(fit)
		print(chi)
		print(popt_truncatedpowerlaw)

		# Output data to array
		if i == 0:
			popt_full = np.asarray(popt_truncatedpowerlaw)
			popt2_full = np.asarray(fit)
			chi_full = np.asarray(chi)
			lenarray_full = np.asarray(len(inputarray))
		else:
			popt_full = np.vstack([popt_full, np.asarray(popt_truncatedpowerlaw)])
			popt2_full = np.vstack([popt2_full, np.asarray(fit)])
			chi_full = np.vstack([chi_full, np.asarray(chi)])
			lenarray_full = np.vstack([lenarray_full, np.asarray(len(inputarray))])

	return popt_full, popt2_full, chi_full

def curve_fit3(array1, array2, array3, mass_lim, flag, flag_plot, array2_err):

	'''
	Function: Master fitting + plotting function
	>>> Flag
	-2 = no output, cumuluative function
	-1 = no output
	1 = output
	2 = output, cumuluative function
	>>> Flag Plot
	2 = yes, no legend, only linear
	1 = yes
	0 = no
	-1 = no, and output alternative legend
	'''

	flag_plot_simple = 1
	# Do basic fitting routine
	array1_plot = np.power(10, np.linspace(np.log10(np.min(array3)), np.log10(np.max(array3))))
	array1_tofit = array1[array1 > mass_lim]
	array2_tofit = array2[array1 > mass_lim]
	array2_tofit_err = array2_err[array1 > mass_lim]
	array1_tofit_masslim = array1_tofit[array1_tofit > mass_lim * masslimtestfactor]
	array2_tofit_masslim = array2_tofit[array1_tofit > mass_lim * masslimtestfactor]
	popt_truncatedpowerlaw = -1
	pcov_truncatedpowerlaw = -1
	popt_schechter = -1
	pcov_schechter = -1

	# Run optimization routines for the simple power law case
	try:
		if np.nanmax(array1_tofit) > 1E8:
			popt_simplepowerlaw, pcov_simplepowerlaw = optimize.curve_fit(simplepowerlaw_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 100000, bounds = [[1E2, -5], [1E12, 0]], absolute_sigma = True, sigma = 0.434 * (array2_tofit_err / array2_tofit), p0 = [1E8, -2])
		else:
			popt_simplepowerlaw, pcov_simplepowerlaw = optimize.curve_fit(simplepowerlaw_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 100000, bounds = [[1E2, -5], [1E12, 0]], absolute_sigma = True, sigma = 0.434 * (array2_tofit_err / array2_tofit), p0 = [1E5, -2])
	except ValueError:
		popt_simplepowerlaw, pcov_simplepowerlaw = optimize.curve_fit(simplepowerlaw_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 100000, bounds = [[1E2, -5], [1E12, 0]], absolute_sigma = True, sigma = 0.434 * (array2_tofit_err / array2_tofit))
	
	# For most functions, run Schechter case
	if abs(flag) == 1:

		# Run optimization routines for Schechter function case
		try:
			popt_schechter, pcov_schechter = optimize.curve_fit(schechter_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 100000, bounds = [[0, 1E2, -5], [1E12, 1E12, 0]], absolute_sigma = True, sigma = 0.434 * (array2_tofit_err / array2_tofit), p0 = [1, 0.5 * np.nanmax(array1_tofit), popt_simplepowerlaw[1]])
		except ValueError:
			try:
				popt_schechter, pcov_schechter = optimize.curve_fit(schechter_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 100000, bounds = [[0, 1E2, -5], [1E12, 1E12, 0]], absolute_sigma = True, sigma =  0.434 * (array2_tofit_err / array2_tofit), p0 = [1, 1E5, popt_simplepowerlaw[1]])
			except ValueError:
				popt_schechter, pcov_schechter = optimize.curve_fit(schechter_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 100000, bounds = [[0, 1E2, -5], [1E12, 1E12, 0]], absolute_sigma = True, sigma =  0.434 * (array2_tofit_err / array2_tofit))

	# For cumulative functions, run truncated power law case
	if abs(flag) == 2:

		# Run optimization routines on the truncated power law function
		if np.nanmax(array1_tofit) > 1E8:
			popt_simplepowerlaw_noweight, pcov_simplepowerlaw_noweight = optimize.curve_fit(simplepowerlaw_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 100000, bounds = [[1E2, -5], [1E12, 0]], absolute_sigma = True, sigma = optimizesigma * np.ones(len(array2_tofit)), p0 = [1E8, -2])
		else:
			popt_simplepowerlaw_noweight, pcov_simplepowerlaw_noweight = optimize.curve_fit(simplepowerlaw_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 100000, bounds = [[1E2, -5], [1E12, 0]], absolute_sigma = True, sigma = optimizesigma * np.ones(len(array2_tofit)), p0 = [1E5, -2])
		try:
			popt_truncatedpowerlaw, pcov_truncatedpowerlaw = optimize.curve_fit(truncatedpowerlaw_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 1000000, bounds = [[0, 0, popt_simplepowerlaw_noweight[1] - 0.5], [1E5, 1E12, popt_simplepowerlaw_noweight[1] + 0.5]], absolute_sigma = True, sigma = optimizesigma * np.ones(len(array2_tofit)), p0 = [1., 0.5 * np.nanmax(array1_tofit), popt_simplepowerlaw_noweight[1]])
		except ValueError:
			try:
				popt_truncatedpowerlaw, pcov_truncatedpowerlaw = optimize.curve_fit(truncatedpowerlaw_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 1000000, bounds = [[0, 0, popt_simplepowerlaw_noweight[1] - 0.5], [1E5, 1E12, popt_simplepowerlaw_noweight[1] + 0.5]], absolute_sigma = True, sigma = optimizesigma * np.ones(len(array2_tofit)), p0 = [1., 1E5, popt_simplepowerlaw_noweight[1]])
			except ValueError:
				try:
					popt_truncatedpowerlaw, pcov_truncatedpowerlaw = optimize.curve_fit(truncatedpowerlaw_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 1000000, bounds = [[0, 0, popt_simplepowerlaw_noweight[1] - 0.5], [1E5, 1E12, popt_simplepowerlaw_noweight[1] + 0.5]], absolute_sigma = True, sigma = optimizesigma * np.ones(len(array2_tofit)))
				except ValueError:
					try:
						popt_truncatedpowerlaw, pcov_truncatedpowerlaw = optimize.curve_fit(truncatedpowerlaw_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 1000000, bounds = [[0, 0, popt_simplepowerlaw_noweight[1] - 1.0], [1E5, 1E12, popt_simplepowerlaw_noweight[1] + 1.0]], absolute_sigma = True, sigma = optimizesigma * np.ones(len(array2_tofit)))
					except ValueError:
						popt_truncatedpowerlaw, pcov_truncatedpowerlaw = optimize.curve_fit(truncatedpowerlaw_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 1000000, absolute_sigma = True, sigma = optimizesigma * np.ones(len(array2_tofit)))

		# Run optimization routines on the truncated power law function with srt(N) weighting
		try:
			popt_truncatedpowerlaw_1, pcov_truncatedpowerlaw_1 = optimize.curve_fit(truncatedpowerlaw_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 100000, bounds = [[0, 0, popt_simplepowerlaw[1] - 0.5], [1E5, 1E12, popt_simplepowerlaw[1] + 0.5]], absolute_sigma = True, sigma = 0.434 * (array2_tofit_err / array2_tofit), p0 = [1., 0.5*np.nanmax(array1_tofit), popt_simplepowerlaw[1]])
		except ValueError:
			try:
				popt_truncatedpowerlaw_1, pcov_truncatedpowerlaw_1 = optimize.curve_fit(truncatedpowerlaw_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 100000, bounds = [[0, 0, popt_simplepowerlaw[1] - 0.5], [1E5, 1E12, popt_simplepowerlaw[1] + 0.5]], absolute_sigma = True, sigma = 0.434 * (array2_tofit_err / array2_tofit), p0 = [0.1, 1E5, popt_simplepowerlaw[1]])
			except ValueError:
				try:
					popt_truncatedpowerlaw_1, pcov_truncatedpowerlaw_1 = optimize.curve_fit(truncatedpowerlaw_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 1000000, bounds = [[0, 0, popt_simplepowerlaw[1] - 0.5], [1E5, 1E12, popt_simplepowerlaw[1] + 0.5]], absolute_sigma = True, sigma = 0.434 * (array2_tofit_err / array2_tofit))
				except ValueError:
					popt_truncatedpowerlaw_1, pcov_truncatedpowerlaw_1 = optimize.curve_fit(truncatedpowerlaw_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 2500000, absolute_sigma = True, sigma = 0.434 * (array2_tofit_err / array2_tofit))

	# Perform KS Test on the resulting functions
	from pynverse import inversefunc
	powerlawchisq = 0
	schechterchisq = 0

	if abs(flag) == 2:

		# Run KS test on power law function
		simplepowerlaw_out = (lambda x: np.power((x / popt_simplepowerlaw[0]), popt_simplepowerlaw[1]))
		invsimplepowerlaw_out = inversefunc(simplepowerlaw_out, y_values = array2_tofit)
		kstest_result_simplepowerlaw = stats.ks_2samp(invsimplepowerlaw_out, array1_tofit)
		print('PL v2 (KS p = {:.0e})'.format(kstest_result_simplepowerlaw[1]))

		# Run KS test on truncated power law function
		truncatedpowerlaw_out = (lambda x: popt_truncatedpowerlaw[0] * (np.power((x / popt_truncatedpowerlaw[1]), popt_truncatedpowerlaw[2]) - 1))
		invtruncatedpowerlaw_out = inversefunc(truncatedpowerlaw_out, y_values = array2_tofit)
		kstest_result_truncatedpowerlaw = stats.ks_2samp(invtruncatedpowerlaw_out, array1_tofit)
		print('Truncated PL (KS p = {:.0e})'.format(kstest_result_truncatedpowerlaw[1]))

		# Run KS test on truncated power law (weighted) function
		truncatedpowerlaw_1_out = (lambda x: popt_truncatedpowerlaw_1[0] * (np.power((x / popt_truncatedpowerlaw_1[1]), popt_truncatedpowerlaw_1[2]) - 1))
		invtruncatedpowerlaw_1_out = inversefunc(truncatedpowerlaw_1_out, y_values = array2_tofit)
		kstest_result_truncatedpowerlaw_1 = stats.ks_2samp(invtruncatedpowerlaw_1_out, array1_tofit)
		print('Truncated PL (weighted) (KS p = {:.0e})'.format(kstest_result_truncatedpowerlaw_1[1]))

	elif abs(flag) == 1:

		# Determine chi2 for power law vs Schechter cases
		simplepowerlaw_out = simplepowerlaw(array1_tofit, *popt_simplepowerlaw)
		schechter_out = schechter(array1_tofit, *popt_schechter)
		powerlawchisq = reducedchisq(simplepowerlaw_out, array2_tofit, 2, sd = array2_tofit_err)
		schechterchisq = reducedchisq(schechter_out, array2_tofit, 3, sd = array2_tofit_err)
		print('PL Reduced Chi2 = {:.2e}, Schechter Reduced Chi2 = {:.2e}'.format(powerlawchisq, schechterchisq))

	# Print fit parameters for simple power law
	# print('  --- Fit Parameters ---')
	print('Simple Power Law Fit')
	print('  A: {:.2e} +/- {:.2e}, B: {:.2f} +/- {:.2f}'.format(popt_simplepowerlaw[0], np.sqrt(abs(pcov_simplepowerlaw[0][0])), popt_simplepowerlaw[1], np.sqrt(abs(pcov_simplepowerlaw[1][1]))))
	print('  Covariance (00, 11): ({:.2e}, {:.2f})'.format(pcov_simplepowerlaw[0][0], pcov_simplepowerlaw[1][1]))
	print('  log A: {:.2f} +/- {:.2f}'.format(np.log10(popt_simplepowerlaw[0]), 0.434 * np.sqrt(abs(pcov_simplepowerlaw[0][0])) / popt_simplepowerlaw[0]))

	# Plot results for simple power law
	if flag_plot > 0:
		if abs(flag) == 1:
			if flag_plot_simple == 1:
				plt.plot(array1_plot, simplepowerlaw(array1_plot, *popt_simplepowerlaw), 'r--', label = r'PL - $\alpha$ = {:.2f} $\pm$ {:.2f}'.format(popt_simplepowerlaw[1], np.sqrt(abs(pcov_simplepowerlaw[1][1]))) +  '\n' + r'              ($\chi^2_r$ = {:.2f})'.format(powerlawchisq))
			else:
				plt.plot(array1_plot, simplepowerlaw(array1_plot, *popt_simplepowerlaw), 'r--', label = r'PL - log M$_0$ = {:.2f}, $\gamma$ = {:.2f}'.format(np.log10(popt_simplepowerlaw[0]), popt_simplepowerlaw[1]) +  '\n' + r'              ($\chi^2_r$ = {:.2f})'.format(powerlawchisq))
		else:
			plt.plot(array1_plot, simplepowerlaw(array1_plot, *popt_simplepowerlaw), 'r--', label = r'PL - log M$_0$ = {:.2f}, $\gamma$ = {:.2f}'.format(np.log10(popt_simplepowerlaw[0]), popt_simplepowerlaw[1]))

	#  Print fit parameters for Schechter function
	if abs(flag) == 1:

		print('Schechter Function Fit')
		print('  A: {:.2e} +/- {:.2e}, B: {:.2e} +/- {:.2e}, C: {:.2f} +/- {:.2f}'.format(popt_schechter[0], np.sqrt(abs(pcov_schechter[0][0])), popt_schechter[1], np.sqrt(abs(pcov_schechter[1][1])), popt_schechter[2], np.sqrt(abs(pcov_schechter[2][2]))))
		print('  Covariance (00, 11, 22): ({:.2e}, {:.2e}, {:.2e})'.format(pcov_schechter[0][0], pcov_schechter[1][1], pcov_schechter[2][2]))
		print('  log A: {:.2f} +/- {:.2f}, log B: {:.2f} +/- {:.2f}'.format(np.log10(popt_schechter[0]), 0.434 * np.sqrt(abs(pcov_schechter[0][0])) / popt_schechter[0], np.log10(popt_schechter[1]), 0.434 * np.sqrt(abs(pcov_schechter[1][1])) / popt_schechter[1]))
	
		# Plot results for Schechter function
		if flag_plot > 0:
			if flag_plot_simple == 1:
				plt.plot(array1_plot, schechter(array1_plot, *popt_schechter), 'b-.', label = r'Sch. - log L$_0$ = {:.2f} $\pm$ {:.2f}'.format(np.log10(popt_schechter[1]), 0.434 * np.sqrt(abs(pcov_schechter[1][1])) / popt_schechter[1]) + '\n' + r'              ($\chi^2_r$ = {:.2f})'.format(schechterchisq))
			else:
				plt.plot(array1_plot, schechter(array1_plot, *popt_schechter), 'b-.', label = r'Sch. - log M$_0$ = {:.2f}, $\gamma$ = {:.2f}'.format(np.log10(popt_schechter[1]), popt_schechter[2]) + '\n' + r'              ($\chi^2_r$ = {:.2f})'.format(schechterchisq))

	# Print fit parameters for truncated power law
	if abs(flag) == 2:

		print('Truncated Power Law Fit')
		print('A: {:.2e} +/- {:.2e}, B: {:.2e} +/- {:.2e}, C: {:.2f} +/- {:.2f}'.format(popt_truncatedpowerlaw[0], np.sqrt(abs(pcov_truncatedpowerlaw[0][0])), popt_truncatedpowerlaw[1], np.sqrt(abs(pcov_truncatedpowerlaw[1][1])), popt_truncatedpowerlaw[2], np.sqrt(abs(pcov_truncatedpowerlaw[2][2]))))
		print('Covariance (00, 11, 22): ({:.2e}, {:.2e}, {:.2e})'.format(pcov_truncatedpowerlaw[0][0], pcov_truncatedpowerlaw[1][1], pcov_truncatedpowerlaw[2][2]))
		print('log A: {:.2f} +/- {:.2f}, log B: {:.2f} +/- {:.2f}'.format(np.log10(popt_truncatedpowerlaw[0]), 0.434 * np.sqrt(abs(pcov_truncatedpowerlaw[0][0])) / popt_truncatedpowerlaw[0], np.log10(popt_truncatedpowerlaw[1]), 0.434 * np.sqrt(abs(pcov_truncatedpowerlaw[1][1])) / popt_truncatedpowerlaw[1]))

		print('Truncated Power Law Fit (weighted)')
		print('A: {:.2e} +/- {:.2e}, B: {:.2e} +/- {:.2e}, C: {:.2f} +/- {:.2f}'.format(popt_truncatedpowerlaw_1[0], np.sqrt(abs(pcov_truncatedpowerlaw_1[0][0])), popt_truncatedpowerlaw_1[1], np.sqrt(abs(pcov_truncatedpowerlaw_1[1][1])), popt_truncatedpowerlaw_1[2], np.sqrt(abs(pcov_truncatedpowerlaw_1[2][2]))))
		print('Covariance (00, 11, 22): ({:.2e}, {:.2e}, {:.2e})'.format(pcov_truncatedpowerlaw_1[0][0], pcov_truncatedpowerlaw_1[1][1], pcov_truncatedpowerlaw_1[2][2]))
		print('log A: {:.2f} +/- {:.2f}, log B: {:.2f} +/- {:.2f}'.format(np.log10(popt_truncatedpowerlaw_1[0]), 0.434 * np.sqrt(abs(pcov_truncatedpowerlaw_1[0][0])) / popt_truncatedpowerlaw_1[0], np.log10(popt_truncatedpowerlaw_1[1]), 0.434 * np.sqrt(abs(pcov_truncatedpowerlaw_1[1][1])) / popt_truncatedpowerlaw_1[1]))

		# Plot results for truncated power law
		if flag_plot > 0:
			plt.plot(array1_plot, truncatedpowerlaw(array1_plot, *popt_truncatedpowerlaw), 'g:', label = r'Truncated PL - N$_0$ = {:.2f}, log M$_0$ = {:.2f}'.format(popt_truncatedpowerlaw[0], np.log10(popt_truncatedpowerlaw[1])) + '\n' + r'                $\gamma$ = {:.2f}'.format(popt_truncatedpowerlaw[2]))

	return popt_truncatedpowerlaw, pcov_truncatedpowerlaw, popt_simplepowerlaw, pcov_simplepowerlaw, powerlawchisq, popt_schechter, pcov_schechter, schechterchisq

def curve_fit_test_simplepowerlaw(array1, array2, mass_lim):

	'''
	Function: Runs simplified fitting routine for power law functions
	'''

	array1_tofit = array1[array2 > 0]
	array2_tofit = array2[array2 > 0]
	try:
		popt_simplepowerlaw, pcov_simplepowerlaw = optimize.curve_fit(simplepowerlaw_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 100000, bounds = [[1E2, -5], [1E12, 0]], absolute_sigma = True, sigma = optimizesigma * np.ones(len(array1_tofit)), p0 = [1E6, -2])
	except ValueError:
		popt_simplepowerlaw, pcov_simplepowerlaw = optimize.curve_fit(simplepowerlaw_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 100000, bounds = [[1E2, -5], [1E12, 0]], absolute_sigma = True, sigma = optimizesigma * np.ones(len(array1_tofit)))

	return popt_simplepowerlaw, pcov_simplepowerlaw

def curve_fit_test_truncatedpowerlaw(array1, array2, mass_lim):

	'''
	Function: Runs simplified fitting routine for truncated power law functions
	'''

	array1_tofit = array1[array2 > 0]
	array2_tofit = array2[array2 > 0]
	try:
		popt_simplepowerlaw, pcov_simplepowerlaw = optimize.curve_fit(simplepowerlaw_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 100000, bounds = [[1E2, -5], [1E12, 0]], absolute_sigma = True, sigma = optimizesigma * np.ones(len(array1_tofit)), p0 = [1E6, -2])
	except ValueError:
		popt_simplepowerlaw, pcov_simplepowerlaw = optimize.curve_fit(simplepowerlaw_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 100000, bounds = [[1E2, -5], [1E12, 0]], absolute_sigma = True, sigma = optimizesigma * np.ones(len(array1_tofit)))

	try:
		popt_truncatedpowerlaw, pcov_truncatedpowerlaw = optimize.curve_fit(truncatedpowerlaw_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 100000, bounds = [[0, 1E2, -5], [1E12, 1E12, 0]], absolute_sigma = True, sigma = optimizesigma * np.ones(len(array1_tofit)), p0 = [1, np.nanmax(array2_tofit), popt_simplepowerlaw[1]])
	except ValueError:
		try:
			popt_truncatedpowerlaw, pcov_truncatedpowerlaw = optimize.curve_fit(truncatedpowerlaw_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 100000, bounds = [[0, 1E2, -5], [1E12, 1E12, 0]], absolute_sigma = True, sigma = optimizesigma * np.ones(len(array1_tofit)), p0 = [1, 1E7, popt_simplepowerlaw[1]])
		except ValueError:
			popt_truncatedpowerlaw, pcov_truncatedpowerlaw = optimize.curve_fit(truncatedpowerlaw_log, np.log10(array1_tofit), np.log10(array2_tofit), maxfev = 100000, bounds = [[0, 1E2, -5], [1E12, 1E12, 0]], absolute_sigma = True, sigma = optimizesigma * np.ones(len(array1_tofit)))

	return popt_truncatedpowerlaw, pcov_truncatedpowerlaw

def curve_fit1slope(array1_1, array1_2, array2_1, array2_2, array3, mass_lim_1, mass_lim_2, flag_label):

	'''
	Function: Run fitting function for two functions together
	'''

	# print(len(array1_1), len(array1_2), len(array2_1), len(array2_2), len(array3))
	array_plot = np.power(10, np.linspace(np.log10(np.min(array3)), np.log10(np.max(array3))))
	array1_1_tofit = array1_1[array1_1 > mass_lim_1]
	array1_2_tofit = array1_2[array1_1 > mass_lim_1]
	array2_1_tofit = array2_1[array2_1 > mass_lim_2]
	array2_2_tofit = array2_2[array2_1 > mass_lim_2]

	# First, fit the first array using a simple power law
	try:
		popt_simplepowerlaw_1, pcov_simplepowerlaw_1 = optimize.curve_fit(simplepowerlaw_log, np.log10(array1_1_tofit), np.log10(array1_2_tofit), maxfev = 100000, bounds = [[1E2, -5], [1E12, 0]], absolute_sigma = True, sigma = optimizesigma * np.ones(len(array1_1_tofit)), p0 = [1E6, -2])
	except ValueError:
		popt_simplepowerlaw_1, pcov_simplepowerlaw_1 = optimize.curve_fit(simplepowerlaw_log, np.log10(array1_1_tofit), np.log10(array1_2_tofit), maxfev = 100000, bounds = [[1E2, -5], [1E12, 0]], absolute_sigma = True, sigma = optimizesigma * np.ones(len(array1_1_tofit)))

	def simplepowerlaw_fixed_log(logM, M0):
		return (popt_simplepowerlaw_1[1] * logM) - (popt_simplepowerlaw_1[1] * np.log10(M0))
	def simplepowerlaw_fixed(M, M0):
		return (np.power((M / M0), popt_simplepowerlaw_1[1]))

	# Next, fit the second array using the same slope	
	popt_simplepowerlaw_2, pcov_simplepowerlaw_2 = optimize.curve_fit(simplepowerlaw_fixed_log, np.log10(array2_1_tofit), np.log10(array2_2_tofit), maxfev = 100000, bounds = [[1E2], [1E12]], absolute_sigma = True, sigma = optimizesigma * np.ones(len(array2_1_tofit)))

	if flag_label == 0:
		plt.plot(array_plot, simplepowerlaw(array_plot, *popt_simplepowerlaw_1), 'y--')
	elif flag_label == 1:		
		plt.plot(array_plot, simplepowerlaw(array_plot, *popt_simplepowerlaw_1), 'y--')
		plt.plot(array_plot, simplepowerlaw_fixed(array_plot, *popt_simplepowerlaw_2), 'b:')
	elif flag_label == 2:
		plt.plot(array_plot, simplepowerlaw(array_plot, *popt_simplepowerlaw_1), 'y--', label = r'SC - log M$_0$ = {:.2f}, $\beta$ = {:.2f}'.format(np.log10(popt_simplepowerlaw_1[0]), popt_simplepowerlaw_1[1]))
		plt.plot(array_plot, simplepowerlaw_fixed(array_plot, *popt_simplepowerlaw_2), 'b:', label = r'GMC - log M$_0$ = {:.2f}'.format(np.log10(popt_simplepowerlaw_2[0])))
	else:
		plt.plot(array_plot, simplepowerlaw_fixed(array_plot, *popt_simplepowerlaw_2), 'r:', label = r'GMC ($\alpha$$_C$$_O$ = 0.8) - log M$_0$ = {:.2f}'.format(np.log10(popt_simplepowerlaw_2[0])))

	return popt_simplepowerlaw_1, popt_simplepowerlaw_2

def curve_fit1slopev2(array1_1, array1_2, array2_1, array2_2, array3, mass_lim_1, mass_lim_2, flag_label):

	'''
	Function: Run fitting function for two functions together (use 2nd function as basis)
	flag_label --> 0 = oneline, 1, twolines, 3
	'''

	array_plot = np.power(10, np.linspace(np.log10(np.min(array3)), np.log10(np.max(array3))))
	array1_1_tofit = array1_1[array1_1 > mass_lim_1]
	array1_2_tofit = array1_2[array1_1 > mass_lim_1]
	array2_1_tofit = array2_1[array2_1 > mass_lim_2]
	array2_2_tofit = array2_2[array2_1 > mass_lim_2]

	# Print length of 2 arrays
	print ('Length of 2 arrays:', len(array1_1_tofit), len(array2_2_tofit))

	# First, fit the first array using a simple power law
	try:
		popt_simplepowerlaw_2, pcov_simplepowerlaw_2 = optimize.curve_fit(simplepowerlaw_log, np.log10(array2_1_tofit), np.log10(array2_2_tofit), maxfev = 100000, bounds = [[1E2, -5], [1E12, 0]], absolute_sigma = True, sigma = optimizesigma * np.ones(len(array2_1_tofit)), p0 = [1E6, -2])
	except ValueError:
		popt_simplepowerlaw_2, pcov_simplepowerlaw_2 = optimize.curve_fit(simplepowerlaw_log, np.log10(array2_1_tofit), np.log10(array2_2_tofit), maxfev = 100000, bounds = [[1E2, -5], [1E12, 0]], absolute_sigma = True, sigma = optimizesigma * np.ones(len(array2_1_tofit)))

	def simplepowerlaw_fixed_log(logM, M0):
		return (popt_simplepowerlaw_2[1] * logM) - (popt_simplepowerlaw_2[1] * np.log10(M0))
	def simplepowerlaw_fixed(M, M0):
		return (np.power((M / M0), popt_simplepowerlaw_2[1]))

	# Next, fit the second array using the same slope	
	popt_simplepowerlaw_1, pcov_simplepowerlaw_1 = optimize.curve_fit(simplepowerlaw_fixed_log, np.log10(array1_1_tofit), np.log10(array1_2_tofit), maxfev = 100000, bounds = [[1E2], [1E12]], absolute_sigma = True, sigma = optimizesigma * np.ones(len(array1_1_tofit)))

	if flag_label == 0:
		plt.plot(array_plot, simplepowerlaw_fixed(array_plot, *popt_simplepowerlaw_1), 'y--')
	elif flag_label == 1:		
		plt.plot(array_plot, simplepowerlaw_fixed(array_plot, *popt_simplepowerlaw_1), 'y--')
		plt.plot(array_plot, simplepowerlaw(array_plot, *popt_simplepowerlaw_2), 'b:')
	elif flag_label == 2:
		plt.plot(array_plot, simplepowerlaw(array_plot, *popt_simplepowerlaw_2), 'b:', label = r'GMC - log M$_0$ = {:.2f}, $\beta$ = {:.2f}'.format(np.log10(popt_simplepowerlaw_2[0]), popt_simplepowerlaw_2[1]))
		plt.plot(array_plot, simplepowerlaw_fixed(array_plot, *popt_simplepowerlaw_1), 'y--', label = r'SC - log M$_0$ = {:.2f}'.format(np.log10(popt_simplepowerlaw_1[0])))
	else:
		plt.plot(array_plot, simplepowerlaw(array_plot, *popt_simplepowerlaw_2), 'r:', label = r'GMC ($\alpha$ = 0.8) - log M$_0$ = {:.2f}, $\beta$ = {:.2f}'.format(np.log10(popt_simplepowerlaw_2[0]), popt_simplepowerlaw_2[1]))

	return popt_simplepowerlaw_1, popt_simplepowerlaw_2

#------------------------------------------------------------------------------
###
# (9) Code Snippets (Other)
###

# Import angle and unit fuctions
from astropy.coordinates import Angle
from astropy import units as u

def sextodecimal(ra, dec):

	'''
	Function: Convert RA and DEC from sexagesimal to decimal
	'''

	input_ra = Angle(ra, unit = u.hour)
	input_dec = Angle(dec, unit = u.deg)
	ra_out = input_ra.degree
	dec_out = input_dec.degree

	return ra_out, dec_out

def decimaltosex(ra, dec):

	'''
	Function: Convert RA and DEC from decimal to sexagesimal
	'''

	input_ra = Angle(ra, unit = u.deg)
	input_dec = Angle(dec, unit = u.deg)
	ra_out = input_ra.to_string(unit = u.hour, sep = ':')
	dec_out = input_dec.to_string(unit = u.deg, sep = ':')
	print(ra_out)
	print(dec_out)

	return ra_out, dec_out

def returnposinfo(galname):
	
	'''
	Function: Return information about a chosen galaxy 
	'''

	POLY1 = [-1., -1.]
	POLY2 = [-1., -1.]
	POLY3 = [-1., -1.]
	POLY4 = [-1., -1.]
	REGPATH = mplpath.Path(np.array([POLY1, POLY2, POLY3, POLY4]))
	POLYPATH = [np.array([POLY1, POLY2, POLY3, POLY4])]

	if galname in ['Antennae']: #, 'Antennae_cut', 'Antennae_spec', 'Antennae_comp'

		RA = 180.4715417
		DEC = -18.8772000
		LOGD25 = 1.
		LOGR25 = 1.
		PA = 0.
		POLY1 = sextodecimal('12:01:56.9', '-18:52:59.1')
		POLY2 = sextodecimal('12:01:52.6', '-18:53:38.3')
		POLY3 = sextodecimal('12:01:51.3', '-18:53:11.6')
		POLY4 = sextodecimal('12:01:53.5', '-18:52:54.6')
		POLY5 = sextodecimal('12:01:53.5', '-18:52:23.7')
		POLY6 = sextodecimal('12:01:51.5', '-18:52:23.7')
		POLY7 = sextodecimal('12:01:51.5', '-18:51:42.9')
		POLY8 = sextodecimal('12:01:54.5', '-18:51:42.9')
		POLY9 = sextodecimal('12:01:54.5', '-18:52:15.2')
		POLY10 = sextodecimal('12:01:56.5', '-18:52:15.2')
		REGPATH = mplpath.Path(np.array([POLY1, POLY2, POLY3, POLY4, POLY5, POLY6, POLY7, POLY8, POLY9, POLY10]))
		POLYPATH = [np.array([POLY1, POLY2, POLY3, POLY4, POLY5, POLY6, POLY7, POLY8, POLY9, POLY10])]

	elif galname == 'NGC4038':	# Antennae

		#J120153.02-185205.4
		RA = 180.4709167
		DEC = -18.8681667
		LOGD25 = 1.73
		LOGR25 = 0.16
		PA = 80.0
		
	elif galname == 'NGC4039': # Antennae

		#J120153.63-185310.9
		RA = 180.4734583
		DEC = -18.8863611
		LOGD25 = 1.73
		LOGR25 = 0.29
		PA = 130.0

	elif galname == 'ARP220':

		# J153457.31+233010.4
		RA = 233.7387917
		DEC = 23.5028889
		LOGD25 = 1.04
		LOGR25 = 0.19
		PA = 96.5

	elif galname in ['Haro11', 'ESO350-IG038']:

		# J003652.51-333317.3
		RA = 9.2187917
		DEC = -33.5548056
		LOGD25 = 0.74
		LOGR25 = 0.12
		PA = 104.9

	elif galname in ['IC1487', 'NGC7649']:

		# J232420.09+143849.3
		RA = 351.0837083    
		DEC = 14.6470278
		LOGD25 = 1.21
		LOGR25 = 0.20
		PA = 76.1

	elif galname in ['LMC']: # , 'LMC_cut', 'LMC_comp', 'LMC_comp_cut'

		# J052334.64-694522.0
		RA = 80.8943333
		DEC = -69.7561111
		LOGD25 = 3.81
		LOGR25 = 0.07
		PA = 170.0
		POLY1 = sextodecimal('5:40:37.8', '-69:41:26.2')
		POLY2 = sextodecimal('5:27:01.6', '-69:41:26.2')
		POLY3 = sextodecimal('5:27:01.6', '-70:56:15.7')
		POLY4 = sextodecimal('4:46:31.8', '-70:56:15.7')
		POLY5 = sextodecimal('4:46:31.8', '-66:57:31.8')
		POLY6 = sextodecimal('5:04:47.3', '-66:57:31.8')
		POLY7 = sextodecimal('5:04:47.3', '-67:40:34.4')
		POLY8 = sextodecimal('5:25:33.8', '-67:40:34.4')
		POLY9 = sextodecimal('5:25:33.8', '-67:11:45.9')
		POLY10 = sextodecimal('5:40:37.8', '-67:11:45.9')
		REGPATH = mplpath.Path(np.array([POLY1, POLY2, POLY3, POLY4, POLY5, POLY6, POLY7, POLY8, POLY9, POLY10]))
		POLYPATH = [np.array([POLY1, POLY2, POLY3, POLY4, POLY5, POLY6, POLY7, POLY8, POLY9, POLY10])]

	elif galname in ['M31']: # , 'M31_N'

		RA = 10.8383458
		DEC = 41.3899500
		LOGD25 = 1.
		LOGR25 = 1.
		PA = 0.

	elif galname in ['M33']: # , 'M33_IR'

		# J013350.91+303935.5
		RA = 23.4621250
		DEC = 30.6598611
		LOGD25 = 2.79
		LOGR25 = 0.23
		PA = 22.7

	elif galname in ['M51']: #  'M51_cut', 'M51_LEG', 'M51_LEG2', 'M51_comp', 'M51_comp_cut'

		# J132952.71+471142.7
		RA = 202.4696250
		DEC = 47.1951944
		LOGD25 = 2.14
		LOGR25 = 0.07
		PA = 163.0
		POLY1 = sextodecimal('13:29:59.9', '47:10:27.0')
		POLY2 = sextodecimal('13:29:40.0', '47:10:27.0')
		POLY3 = sextodecimal('13:29:40.0', '47:11:19.4')
		POLY4 = sextodecimal('13:29:45.4', '47:12:56.1')
		POLY5 = sextodecimal('13:30:04.5', '47:12:56.1')
		POLY6 = sextodecimal('13:30:04.5', '47:11:55.3')
		REGPATH = mplpath.Path(np.array([POLY1, POLY2, POLY3, POLY4, POLY5, POLY6]))
		POLYPATH = [np.array([POLY1, POLY2, POLY3, POLY4, POLY5, POLY6])]

	elif galname in ['M83']: # , 'M83_R1', 'M83_R2', 'M83_R3', 'M83_R4', 'M83_N', 'M83_cut', 'M83_cut_N', 'M83_SV', 'M83_SV_R1', 'M83_SV_R2', 'M83_SV_R3', 'M83_SV_R4', 'M83_All', 'M83_HMXB', 'M83_IMXB', 'M83_LMXB', 'M83_XB00A', 'M83_XB00B', 'M83_XB00C', 'M83_XB01', 'M83_XB02', 'M83_XB03', 'M83_XB04', 'M83_XB05', 'M83_XB06', 'M83_XB07', 'M83_XB08', 'M83_XB09', 'M83_XB10', 'M83_XB11', 'M83_comp'

		# J133700.94-295156.1
		RA = 204.2539167
		DEC = -29.8655833
		LOGD25 = 2.13
		LOGR25 = 0.01
		PA = 44.9
		POLY1 = sextodecimal('13:37:14.3', '-29:53:32.3')
		POLY2 = sextodecimal('13:36:55.2', '-29:53:32.3')
		POLY3 = sextodecimal('13:36:55.2', '-29:47:01.1')
		POLY4 = sextodecimal('13:37:14.3', '-29:47:01.1')
		REGPATH = mplpath.Path(np.array([POLY1, POLY2, POLY3, POLY4]))
		POLYPATH = [np.array([POLY1, POLY2, POLY3, POLY4])]

	elif galname in ['M100']:

		# J122254.89+154920.3
		RA = 185.7287083
		DEC = 15.8223056
		LOGD25 = 1.78
		LOGR25 = 0.04
		PA = 80.0
		# (51037)

	elif galname in ['M101']: # , 'M101L1', 'M101L2', 'M101L1_cut', 'M101L2_cut', 'M101_All', 'M101_HMXB', 'M101_IMXB', 'M101_LMXB', 'M101_XB00A', 'M101_XB00B', 'M101_XB00C', 'M101_XB01', 'M101_XB02', 'M101_XB03', 'M101_XB04', 'M101_XB05', 'M101_XB06', 'M101_XB07', 'M101_XB08', 'M101_XB09', 'M101_XB10', 'M101_XB11'

		# J140312.59+542056.7
		RA = 210.8024583
		DEC = 54.3490833
		LOGD25 = 2.38
		LOGR25 = 0.02
		PA = 0
		POLY1 = sextodecimal('14:03:26.64', '54:26:42.86')
		POLY2 = sextodecimal('14:02:44.74', '54:24:26.35')
		POLY3 = sextodecimal('14:02:53.01', '54:21:47.20')
		POLY4 = sextodecimal('14:02:47.34', '54:21:33.56')
		POLY5 = sextodecimal('14:02:48.63', '54:20:59.95')
		POLY6 = sextodecimal('14:02:27.14', '54:20:08.74')
		POLY7 = sextodecimal('14:02:34.96', '54:16:53.08')
		POLY8 = sextodecimal('14:02:55.69', '54:17:47.11')
		POLY9 = sextodecimal('14:03:02.24', '54:15:09.08')
		POLY10 = sextodecimal('14:03:25.27', '54:16:08.70')
		POLY11 = sextodecimal('14:03:31.86', '54:13:26.02')
		POLY12 = sextodecimal('14:03:54.34', '54:14:24.12')
		POLY13 = sextodecimal('14:03:45.97', '54:17:41.28')
		POLY14 = sextodecimal('14:04:06.07', '54:18:54.61')
		POLY15 = sextodecimal('14:03:56.18', '54:21:44.22')
		POLY16 = sextodecimal('14:03:58.88', '54:21:54.19')
		POLY17 = sextodecimal('14:03:47.91', '54:24:52.54')
		POLY18 = sextodecimal('14:03:35.41', '54:24:10.82')
		REGPATH = mplpath.Path(np.array([POLY1, POLY2, POLY3, POLY4, POLY5, POLY6, POLY7, POLY8, POLY9, POLY10, POLY11, POLY12, POLY13, POLY14, POLY15, POLY16, POLY17, POLY18]))
		POLYPATH = [np.array([POLY1, POLY2, POLY3, POLY4, POLY5, POLY6, POLY7, POLY8, POLY9, POLY10, POLY11, POLY12, POLY13, POLY14, POLY15, POLY16, POLY17, POLY18])]

	elif galname in ['NGC45', 'NGC0045']:

		# J001403.98-231054.8
		RA = 3.5165833
		DEC = -23.1818889
		LOGD25 = 1.79
		LOGR25 = 0.14
		PA = 158.7	

	elif galname == 'NGC300':

		# J005453.54-374104.3
		RA = 13.7230833
		DEC = -37.6845278
		LOGD25 = 2.29
		LOGR25 = 0.17
		PA = 114.3

	elif galname == 'NGC0625':

		# J013504.64-412610.5
		RA = 23.7693333
		DEC = -41.4362500
		LOGD25 = 1.82
		LOGR25 = 0.50
		PA = 91.9

	elif galname == 'NGC0628':

		# J013641.81+154700.3
		RA = 24.1742083
		DEC = 15.7834167
		LOGD25 = 2.00
		LOGR25 = 0.03
		PA = 25.0

	elif galname == 'NGC0853':

		# J021141.34-091819.5
		RA = 32.9222500
		DEC = -9.3054167
		LOGD25 = 1.25
		LOGR25 = 0.19
		PA = 50.1

	elif galname == 'NGC1012':

		# J023914.96+300905.8
		RA = 39.8123333
		DEC = 30.1516111
		LOGD25 = 1.32
		LOGR25 = 0.30
		PA = 23.5

	elif galname == 'NGC1013':

		# J023750.46-113026.2
		RA = 39.4602500
		DEC = -11.5072778
		LOGD25 = 1.03
		LOGR25 = 0.20
		PA = 66.0

	elif galname == 'NGC1035':

		# J023929.10-080759.3
		RA = 39.8712500
		DEC = -8.1331389
		LOGD25 = 1.32
		LOGR25 = 0.48
		PA = 152.9

	elif galname == 'NGC1433':

		# J034201.55-471319.4
		RA = 55.5064583
		DEC = -47.2220556
		LOGD25 = 1.79
		LOGR25 = 0.32
		PA = 95.1

	elif galname == 'NGC1559':

		# J041735.81-624701.3
		RA = 64.3992083
		DEC = -62.7836944
		LOGD25 = 1.62
		LOGR25 = 0.28
		PA = 62.9

	elif galname == 'NGC1566':

		# J042000.42-545616.2
		RA = 65.0017500
		DEC = -54.9378333
		LOGD25 = 1.86
		LOGR25 = 0.16
		PA = 50.3

	elif galname == 'NGC1614':

		# J043400.03-083444.8
		RA = 68.5001250
		DEC = -8.5791111
		LOGD25 = 1.10
		LOGR25 = 0.12
		PA = 0

	elif galname == 'NGC1705':

		# J045413.53-532139.7
		RA = 73.5563750
		DEC = -53.3610278
		LOGD25 = 1.27
		LOGR25 = 0.11
		PA = 53.1

	elif galname in ['NGC3256', 'NGC3256_cut', 'NGC3256_spec', 'NGC3256_comp']:

		# J102751.29-435413.4
		RA = 156.9637083
		DEC = -43.9037222
		LOGD25 = 1.52
		LOGR25 = 0.16
		PA = 83.2
		POLY1 = sextodecimal('10:27:52.4', '-43:54:07.7')
		POLY2 = sextodecimal('10:27:50.5', '-43:54:07.7')
		POLY3 = sextodecimal('10:27:49.4', '-43:54:18.7')
		POLY4 = sextodecimal('10:27:50.1', '-43:54:32.4')
		POLY5 = sextodecimal('10:27:52.9', '-43:54:15.8')
		REGPATH = mplpath.Path(np.array([POLY1, POLY2, POLY3, POLY4, POLY5]))
		POLYPATH = [np.array([POLY1, POLY2, POLY3, POLY4, POLY5])]

	elif galname == 'NGC3344':

		# J104331.16+245520.3
		RA = 160.8798333
		DEC = 24.9223056
		LOGD25 = 1.83
		LOGR25 = 0.02
		PA = 0

	elif galname in ['NGC3627', 'NGC3627_2', 'NGC3627_cut', 'NGC3627_comp']:

		# J112015.02+125930.0
		RA = 170.0625833
		DEC = 12.9916667
		LOGD25 = 2.01
		LOGR25 = 0.35
		PA = 168.1
		POLY1 = sextodecimal('11:20:16.6', '12:59:36.4')
		POLY2 = sextodecimal('11:20:10.8', '12:59:07.8')
		POLY3 = sextodecimal('11:20:10.8', '12:56:10.2')
		POLY4 = sextodecimal('11:20:19.9', '12:56:10.2')
		POLY5 = sextodecimal('11:20:19.9', '12:58:46.9')
		REGPATH = mplpath.Path(np.array([POLY1, POLY2, POLY3, POLY4, POLY5]))
		POLYPATH = [np.array([POLY1, POLY2, POLY3, POLY4, POLY5])]

	elif galname in ['NGC3351']: #, 'NGC3351_P'

		# J104357.71+114213.5
		RA = 160.9904583
		DEC = 11.7037500
		LOGD25 = 1.86
		LOGR25 = 0.21
		PA = 10.7

	elif galname in ['NGC3738', 'Dwarf2_NGC3738']:

		# J113548.81+543128.2
		RA = 173.9533750
		DEC = 54.5245000
		LOGD25 = 1.36
		LOGR25 = 0.14
		PA = 156.4
		POLY1 = sextodecimal('11:35:50.8', '54:32:07.7')
		POLY2 = sextodecimal('11:35:43.1', '54:32:05.2')
		POLY3 = sextodecimal('11:35:46.7', '54:30:44.2')
		POLY4 = sextodecimal('11:35:53.9', '54:31:01.2')
		REGPATH = mplpath.Path(np.array([POLY1, POLY2, POLY3, POLY4]))
		POLYPATH = [np.array([POLY1, POLY2, POLY3, POLY4])]

	elif galname == 'NGC4130': # NGC4129

		# J120853.21-090210.9
		RA = 182.2217083
		DEC = -9.0363611
		LOGD25 = 1.41
		LOGR25 = 0.56
		PA = 93.6

	elif galname == 'NGC4150':

		# J121033.67+302405.9
		RA = 182.6402917
		DEC = 30.4016389
		LOGD25 = 1.30
		LOGR25 = 0.17
		PA = 147.8

	elif galname == 'NGC4214':

		# J121539.44+361935.2
		RA = 183.9143333
		DEC = 36.3264444
		LOGD25 = 1.83
		LOGR25 = 0.11
		PA = 20.0

	elif galname == 'NGC4242':

		# J121730.17+453709.3
		RA = 184.3757083
		DEC = 45.6192500
		LOGD25 = 1.58
		LOGR25 = 0.15
		PA = 22.6

	elif galname == 'NGC4310':

		# J122226.30+291232.0
		RA = 185.60958333
		DEC = 29.20888889
		LOGD25 = 1.29
		LOGR25 = 0.40
		PA = 147.9


	elif galname == 'NGC4376':

		# J122518.08+054428.3
		RA = 186.3253333
		DEC = 5.7411944
		LOGD25 = 1.12
		LOGR25 = 0.22
		PA = 152.0

	elif galname == 'NGC4395':

		# J122548.91+333248.7
		RA = 186.4537917
		DEC = 33.5468611
		LOGD25 = 1.62
		LOGR25 = 0.48
		PA = 127.8

	elif galname == 'NGC4396':

		# J122559.11+154015.2
		RA = 186.4962917
		DEC = 15.6708889
		LOGD25 = 1.47
		LOGR25 = 0.45
		PA = 124.3

	elif galname == 'NGC4451':

		# J122840.55+091532.3
		RA = 187.1689583
		DEC = 9.2589722
		LOGD25 = 1.13
		LOGR25 = 0.20
		PA = 166.4

	elif galname in ['NGC4449', 'NGC4449_LEG', 'Dwarf', 'Dwarf2', 'Dwarf2_Low', 'Dwarf2_Med', 'Dwarf2_High', 'Dwarf2_NoSB', 'Dwarf2_NGC4449']:

		# J122811.11+440537.3
		RA = 187.0462917
		DEC = 44.0936944
		LOGD25 = 1.67
		LOGR25 = 0.23
		PA = 50.8

	elif galname == 'NGC4526':

		# J123403.02+074157.6
		RA = 188.5125833
		DEC = 7.6993333
		LOGD25 = 1.84
		LOGR25 = 0.45
		PA = 138.0

	elif galname in ['NGC4656', 'Dwarf2_NGC4656']:

		# J124357.71+321010.9
		RA = 190.9904583
		DEC = 32.1696944
		LOGD25 = 1.81
		LOGR25 = 0.99
		PA = 35.6
		POLY1 = sextodecimal('12:44:05.4', '32:11:43.5')
		POLY2 = sextodecimal('12:44:00.2', '32:11:49.8')
		POLY3 = sextodecimal('12:43:52.4', '32:09:42.7')
		POLY4 = sextodecimal('12:43:58.0', '32:09:13.4')
		REGPATH = mplpath.Path(np.array([POLY1, POLY2, POLY3, POLY4]))
		POLYPATH = [np.array([POLY1, POLY2, POLY3, POLY4])]

	elif galname == 'NGC4632':

		# J124232.34-000454.2
		RA = 190.6347500
		DEC = -0.0817222
		LOGD25 = 1.41
		LOGR25 = 0.42
		PA = 59.0

	elif galname == 'NGC4701':

		# J124911.60+032319.7
		RA = 192.2983333
		DEC = 3.3888056
		LOGD25 = 1.24
		LOGR25 = 0.13
		PA = 38.9

	elif galname == 'NGC4826':

		# J125643.69+214055.8
		RA = 194.1820417
		DEC = 21.6821667
		LOGD25 = 2.02
		LOGR25 = 0.29
		PA = 114.0

	elif galname == 'NGC5194':

		# J132952.71+471142.7
		RA = 202.4696250
		DEC = 47.1951944
		LOGD25 = 2.14
		LOGR25 = 0.07
		PA = 163.0

	elif galname == 'NGC5238':

		# J133442.52+513649.4
		RA = 203.6771667
		DEC = 51.6137222
		LOGD25 = 1.24
		LOGR25 = 0.08
		PA = 161.7

	elif galname in ['NGC5253', 'Dwarf2_NGC5253']:

		# J133956.00-313823.9
		RA = 204.9833333
		DEC = -31.6399722
		LOGD25 = 1.70
		LOGR25 = 0.37
		PA = 43.2
		POLY1 = sextodecimal('13:40:00.0', '-31:38:09.0')
		POLY2 = sextodecimal('13:39:57.0', '-31:37:33.9')
		POLY3 = sextodecimal('13:39:50.6', '-31:38:59.4')
		POLY4 = sextodecimal('13:39:54.7', '-31:39:33.1')
		REGPATH = mplpath.Path(np.array([POLY1, POLY2, POLY3, POLY4]))
		POLYPATH = [np.array([POLY1, POLY2, POLY3, POLY4])]

	elif galname == 'NGC5457':

		# J140312.59+542056.7
		RA = 210.8024583
		DEC = 54.3490833
		LOGD25 = 2.38
		LOGR25 = 0.02
		PA = 0

	elif galname == 'NGC5474':

		# J140501.52+533944.3
		RA = 211.2563333
		DEC = 53.6623056
		LOGD25 = 1.38
		LOGR25 = 0.18
		PA = 99.7

	elif galname == 'NGC5477':

		# J140533.32+542739.4
		RA = 211.3888333
		DEC = 54.4609444
		LOGD25 = 1.04
		LOGR25 = 0.16
		PA = 87.3

	elif galname == 'NGC5692':

		# J143818.12+032437.3
		RA = 219.57550000
		DEC = 3.41036111
		LOGD25 = 0.98
		LOGR25 = 0.20
		PA = 35.5

	elif galname == 'NGC5962':

		# J153631.69+163628.2
		RA = 234.1320417
		DEC = 16.6078333
		LOGD25 = 1.40
		LOGR25 = 0.19
		PA = 106.3

	elif galname == 'NGC6106':

		# J161847.00+072434.0
		RA = 244.6958333
		DEC = 7.4094444
		LOGD25 = 1.34
		LOGR25 = 0.27
		PA = 139.0

	elif galname == 'NGC6503':

		# J174926.56+700839.6
		RA = 267.3606667
		DEC = 70.1443333
		LOGD25 = 1.77
		LOGR25 = 0.48
		PA = 123.3

	elif galname == 'NGC6744':

		# J190946.03-635127.0
		RA = 287.44179167
		DEC = -63.85750000
		LOGD25 = 2.19
		LOGR25 = 0.21
		PA = 15.4

	elif galname == 'NGC6946':

		# J203452.63+600912.5
		RA = 308.7192917
		DEC = 60.1534722
		LOGD25 = 2.06
		LOGR25 = 0.02
		PA = 60

	elif galname == 'NGC7793':

		# J235749.83-323528.1
		RA = 359.4576250
		DEC = -32.5911389
		LOGD25 = 2.02
		LOGR25 = 0.24
		PA = 91.8

	elif galname == 'SMC':

		# J005238.01-724801.0
		RA = 13.1583750
		DEC = -72.8002778
		LOGD25 = 3.48
		LOGR25 = 0.22
		PA = 45

	elif galname in ['UGC1249', 'Dwarf2_UGC1249']:

		# J014729.90+271958.6
		RA = 26.8745833
		DEC = 27.3329444
		LOGD25 = 1.81
		LOGR25 = 0.43
		PA = 150.3
		POLY1 = sextodecimal('1:47:30.5', '27:21:12.4')
		POLY2 = sextodecimal('1:47:22.6', '27:21:05.8')
		POLY3 = sextodecimal('1:47:28.1', '27:18:20.2')
		POLY4 = sextodecimal('1:47:35.9', '27:19:30.3')
		REGPATH = mplpath.Path(np.array([POLY1, POLY2, POLY3, POLY4]))
		POLYPATH = [np.array([POLY1, POLY2, POLY3, POLY4])]

	elif galname == 'UGC8516':

		# J133152.57+200004.0
		RA = 202.9690417
		DEC = 20.0011111
		LOGD25 = 1.07
		LOGR25 = 0.17
		PA = 35.0

	else:

		RA = -1
		DEC = -1
		LOGD25 = 0.1
		LOGR25 = 0.1
		PA = 0.1

	# PA: clockwise from N --> couterclockwise from east	
	PA = 90 + PA
	R25 = np.power(10, LOGD25) / (10. * 60. * 2.)
	LONGAXIS = R25
	SHORTAXIS = R25 /  np.power(10, LOGR25)
	print('- {} returnposinfo: {}, {}, {}, {:.7f}, {:.7f}'.format(galname, RA, DEC, PA, LONGAXIS, SHORTAXIS))

	return RA, DEC, PA, LONGAXIS, SHORTAXIS, REGPATH, POLYPATH

def makearrayhist(array1, array2, mass_lim, massindex = 4):

	'''
	Function: Create bins with equal width from array
	'''

	n, bins = np.histogram([array1[:,massindex]], bins = array2)
	n_fit = []
	n_fit_err = []
	n_dM = []
	n_fit_dM = []
	n_fit_dM_err = []
	n_dlogM = []
	n_fit_dlogM = []
	n_fit_dlogM_err = []
	bins_fit = []
	bins_width = np.diff(bins)
	bins_width_log = np.diff(np.log10(bins))
	bins_centre = np.power(10, (np.log10(bins[:-1]) + np.log10(bins[1:])) / 2.)
	for i in range(0, len(n)):
		n_dM.append(n[i] / bins_width[i])
		n_dlogM.append(n[i] / bins_width_log[i])
		if n[i] > 0 and bins_centre[i] > mass_lim:
			bins_fit.append(bins_centre[i])
			n_fit.append(n[i])
			n_fit_dM.append(n[i] / (bins_width[i]))
			n_fit_dlogM.append(n[i] / bins_width_log[i])
			n_fit_err.append(np.sqrt(n[i]))
			n_fit_dM_err.append(np.sqrt(n[i]) / (bins_width[i]))
			n_fit_dlogM_err.append(np.sqrt(n[i]) / bins_width_log[i])
	print('Equal Width Bins Created for N = {}; {} bins with largest bin = {}'.format(len(array1[:,massindex]), np.count_nonzero(n), np.nanmax(n)))
	###
	ncum = np.cumsum(n[::-1])[::-1]
	ncum_fit = np.cumsum(n_fit[::-1])[::-1]

	return np.asarray(n), np.asarray(bins), np.asarray(bins_width), np.asarray(bins_centre), np.asarray(n_fit), np.asarray(bins_fit), np.asarray(n_dM), np.asarray(n_fit_dM), np.asarray(n_dlogM), np.asarray(n_fit_dlogM), np.asarray(ncum), np.asarray(ncum_fit), np.asarray(n_fit_err), np.asarray(n_fit_dM_err), np.asarray(n_fit_dlogM_err)

def makearrayhistequal(array1, mass_lim, flag, numgal_bin_in = 5, massindex = 4):
	
	'''
	Function: Create bins with equal numbers from array
	'''

	if len(array1) > 1:

		# Sort data and find out number above mass limit
		sorted_data = np.sort(array1[:,massindex])
		reverse_sorted_data = sorted_data[::-1]
		lensorted = len(sorted_data[sorted_data > mass_lim])

		if flag == 0:

			if lensorted < 10000:
				numgal_bin = 20
			else:
				numgal_bin = 1000

		elif flag == 1:

			if lensorted < 10000:
				numgal_bin = 10
			else:
				numgal_bin = 500

		elif flag == 2:

			if lensorted < 10000:
				numgal_bin = 40
			else:
				numgal_bin = 2000

		elif flag == 3:

			numgal_bin = 5

		else:

			numgal_bin = numgal_bin_in

		nequal_reverse = []
		binsequal_reverse = []
		binsaverage_reverse = []

		# Create first point in bin-edge
		binsequal_reverse.append(reverse_sorted_data[0])

		# For remaining edges, find next bins where numgal - 1 (i.e. 0-19)
		# Next if number / numgal_bin == (numgal - 1)
		# Sum up total number in between, set bin_location
		for i in range(0, len(reverse_sorted_data)):
			if i == numgal_bin - 1 or i % numgal_bin == (numgal_bin - 1):
				n_temp_sum = np.sum(reverse_sorted_data[i - (numgal_bin - 1):(i + 1)])
				binsaverage_reverse.append(n_temp_sum / float(numgal_bin))
				nequal_reverse.append(numgal_bin)
				binsequal_reverse.append(reverse_sorted_data[i])

		nequal = nequal_reverse[::-1]
		binsequal = binsequal_reverse[::-1]
		binsaverage = binsaverage_reverse[::-1]
		
		nequal_fit = []
		nequal_fit_err = []
		nequal_dM = []
		nequal_fit_dM = []
		nequal_fit_dM_err = []
		nequal_dlogM = []
		nequal_fit_dlogM = []
		nequal_fit_dlogM_err = []
		binsequal_fit = []
		binsequal_centre = []

		binsequal_width = np.diff(binsequal)
		binsequal_width_log = np.diff(np.log10(binsequal))

		for i in range(0, len(binsequal) - 1):
			bincentre = np.power(10, (np.log10(binsequal[i]) + np.log10(binsequal[i + 1])) / 2.)
			binsequal_centre.append(bincentre)
		

		for i in range(0, len(nequal)):
			nequal_dM.append(nequal[i] / binsequal_width[i])
			nequal_dlogM.append(nequal[i] / binsequal_width_log[i])
			if nequal[i] > 0 and binsequal_width[i] > 0 and binsequal_centre[i] > mass_lim:
				binsequal_fit.append(binsequal_centre[i])
				nequal_fit.append(nequal[i])
				nequal_fit_dM.append(nequal[i] / (binsequal_width[i]))
				nequal_fit_dlogM.append(nequal[i] / binsequal_width_log[i])
				nequal_fit_err.append(np.sqrt(nequal[i]))
				nequal_fit_dM_err.append(np.sqrt(nequal[i]) / (binsequal_width[i]))
				nequal_fit_dlogM_err.append(np.sqrt(nequal[i]) / binsequal_width_log[i])
		
		if len(nequal) > 0:
			print('Equal Number Bins Created for N = {}; {} bins with number in each bin = {}'.format(len(array1[:,massindex]), np.count_nonzero(nequal), np.nanmax(nequal)))
		###
		nequalcum = np.cumsum(nequal[::-1])[::-1]
		nequalcum_fit = np.cumsum(nequal_fit[::-1])[::-1]

	else:

		nequal = []
		binsequal = []
		binsequal_fit = []
		binsequal_width = []
		binsequal_centre = []
		nequal_fit = []
		nequal_fit_err = []
		nequal_fit_dM = []
		nequal_dM = []
		nequal_fit_dM = []
		nequal_fit_dM_err = []
		nequal_dlogM = []
		nequal_fit_dlogM = []
		nequal_fit_dlogM_err = []
		nequalcum = []
		nequalcum_fit = []

	return np.asarray(nequal), np.asarray(binsequal), np.asarray(binsequal_width), np.asarray(binsequal_centre), np.asarray(nequal_fit), np.asarray(binsequal_fit), np.asarray(nequal_dM), np.asarray(nequal_fit_dM), np.asarray(nequal_dlogM), np.asarray(nequal_fit_dlogM), np.asarray(nequalcum), np.asarray(nequalcum_fit), np.asarray(nequal_fit_err), np.asarray(nequal_fit_dM_err), np.asarray(nequal_fit_dlogM_err)

def makearrayhisttime(array1, array2, time_lim):

	'''
	Function: Create bins with equal width from array
	'''

	n, bins = np.histogram([array1[:,6]], bins = array2)
	n_fit = []
	n_fit_err = []
	n_dT = []
	n_fit_dT = []
	n_fit_dT_err = []
	n_dlogT = []
	n_fit_dlogT = []
	n_fit_dlogT_err = []
	bins_fit = []
	bins_width = np.diff(bins)
	bins_width_log = np.diff(np.log10(bins))
	bins_centre = np.power(10, (np.log10(bins[:-1]) + np.log10(bins[1:])) / 2.)
	for i in range(0, len(n)):
		n_dT.append(n[i] / bins_width[i])
		n_dlogT.append(n[i] / bins_width_log[i])
		if n[i] > 0 and bins_centre[i] > time_lim:
			bins_fit.append(bins_centre[i])
			n_fit.append(n[i])
			n_fit_dT.append(n[i] / (bins_width[i]))
			n_fit_dlogT.append(n[i] / bins_width_log[i])
			n_fit_err.append(np.sqrt(n[i]))
			n_fit_dT_err.append(np.sqrt(n[i]) / (bins_width[i]))
			n_fit_dlogT_err.append(np.sqrt(n[i]) / bins_width_log[i])
	print('Equal Width Bins Created for N = {}; {} bins with largest bin = {}'.format(len(array1[:,6]), np.count_nonzero(n), np.nanmax(n)))
	###
	ncum = np.cumsum(n[::-1])[::-1]
	ncum_fit = np.cumsum(n_fit[::-1])[::-1]

	return np.asarray(n), np.asarray(bins), np.asarray(bins_width), np.asarray(bins_centre), np.asarray(n_fit), np.asarray(bins_fit), np.asarray(n_dT), np.asarray(n_fit_dT), np.asarray(n_dlogT), np.asarray(n_fit_dlogT), np.asarray(ncum), np.asarray(ncum_fit), np.asarray(n_fit_err), np.asarray(n_fit_dT_err), np.asarray(n_fit_dlogT_err)

def makearrayhistequaltime(array1, time_lim, flag):
	
	'''
	Function: Create bins with equal numbers from array
	'''

	if len(array1) > 19:

		# Sort data and find out number above time limit
		sorted_data = np.sort(array1[:,6])
		reverse_sorted_data = sorted_data[::-1]
		lensorted = len(sorted_data[sorted_data > time_lim])

		if flag == 0:

			if lensorted < 10000:
				numgal_bin = 20
			else:
				numgal_bin = 1000

		elif flag == 1:

			if lensorted < 10000:
				numgal_bin = 10
			else:
				numgal_bin = 500

		elif flag == 2:

			if lensorted < 10000:
				numgal_bin = 40
			else:
				numgal_bin = 2000

		nequal_reverse = []
		binsequal_reverse = []
		binsaverage_reverse = []

		# Create first point in bin-edge
		binsequal_reverse.append(reverse_sorted_data[0])

		# For remaining edges, find next bins where numgal - 1 (i.e. 0-19)
		# Next if number / numgal_bin == (numgal - 1)
		# Sum up total number in between, set bin_location
		for i in range(0, len(reverse_sorted_data)):
			if i == numgal_bin - 1 or i % numgal_bin == (numgal_bin - 1):
				n_temp_sum = np.sum(reverse_sorted_data[i - (numgal_bin - 1):(i + 1)])
				binsaverage_reverse.append(n_temp_sum / float(numgal_bin))
				nequal_reverse.append(numgal_bin)
				binsequal_reverse.append(reverse_sorted_data[i])

		nequal = nequal_reverse[::-1]
		binsequal = binsequal_reverse[::-1]
		binsaverage = binsaverage_reverse[::-1]
		
		nequal_fit = []
		nequal_fit_err = []
		nequal_dT = []
		nequal_fit_dT = []
		nequal_fit_dT_err = []
		nequal_dlogT = []
		nequal_fit_dlogT = []
		nequal_fit_dlogT_err = []
		binsequal_fit = []
		binsequal_centre = []

		binsequal_width = np.diff(binsequal)
		binsequal_width_log = np.diff(np.log10(binsequal))

		for i in range(0, len(binsequal) - 1):
			bincentre = np.power(10, (np.log10(binsequal[i]) + np.log10(binsequal[i + 1])) / 2.)
			binsequal_centre.append(bincentre)
		

		for i in range(0, len(nequal)):
			nequal_dT.append(nequal[i] / binsequal_width[i])
			nequal_dlogT.append(nequal[i] / binsequal_width_log[i])
			if nequal[i] > 0 and binsequal_width[i] > 0 and binsequal_centre[i] > time_lim:
				binsequal_fit.append(binsequal_centre[i])
				nequal_fit.append(nequal[i])
				nequal_fit_dT.append(nequal[i] / (binsequal_width[i]))
				nequal_fit_dlogT.append(nequal[i] / binsequal_width_log[i])
				nequal_fit_err.append(np.sqrt(nequal[i]))
				nequal_fit_dT_err.append(np.sqrt(nequal[i]) / (binsequal_width[i]))
				nequal_fit_dlogT_err.append(np.sqrt(nequal[i]) / binsequal_width_log[i])
		
		if len(nequal) > 0:
			print('Equal Number Bins Created for N = {}; {} bins with number in each bin = {}'.format(len(array1[:,6]), np.count_nonzero(nequal), np.nanmax(nequal)))
		###
		nequalcum = np.cumsum(nequal[::-1])[::-1]
		nequalcum_fit = np.cumsum(nequal_fit[::-1])[::-1]

	else:

		nequal = []
		binsequal = []
		binsequal_fit = []
		binsequal_width = []
		binsequal_centre = []
		nequal_fit = []
		nequal_fit_err = []
		nequal_fit_dT = []
		nequal_dT = []
		nequal_fit_dT = []
		nequal_fit_dT_err = []
		nequal_dlogT = []
		nequal_fit_dlogT = []
		nequal_fit_dlogT_err = []
		nequalcum = []
		nequalcum_fit = []

	return np.asarray(nequal), np.asarray(binsequal), np.asarray(binsequal_width), np.asarray(binsequal_centre), np.asarray(nequal_fit), np.asarray(binsequal_fit), np.asarray(nequal_dT), np.asarray(nequal_fit_dT), np.asarray(nequal_dlogT), np.asarray(nequal_fit_dlogT), np.asarray(nequalcum), np.asarray(nequalcum_fit), np.asarray(nequal_fit_err), np.asarray(nequal_fit_dT_err), np.asarray(nequal_fit_dlogT_err)

def makecdffun(array1, complimits_val):

	'''
	Function: Create a cumulative distribution function and plot the result
	'''

	# Sort data array, create x and y arrays
	sorted_data = np.sort(array1)
	xdata = np.concatenate([sorted_data[::-1]])
	ydata = np.arange(sorted_data.size) + 1

	# Filter array to those > complimit, output results
	filt_array = np.where(sorted_data >= complimits_val)
	data_lim = sorted_data[filt_array]
	sorted_data_lim = np.sort(data_lim)

	# Output results
	xdata_lim = np.concatenate([sorted_data_lim[::-1]])
	ydata_lim = np.arange(sorted_data_lim.size) + 1
	xdata_lim_float = xdata_lim.astype(float)
	ydata_lim_float = ydata_lim.astype(float)
	
	# Print result to screen
	print('CDF Function Results {:.2e}:'.format(complimits_val))
	print('[Len: {}, Min: {:.2e}, Max:{:.2e}]'.format(len(xdata), np.nanmin(xdata), np.nanmax(xdata)))
	print('[Len: {}, Min: {:.2e}, Max:{:.2e}]'.format(len(xdata_lim_float), np.nanmin(xdata_lim_float), np.nanmax(xdata_lim_float)))
	
	plt.step(xdata, ydata)

	return xdata_lim_float, ydata_lim_float

def makecdffunnoplot(array1, complimits_val):

	'''
	Function: Create a cumulative distribution function and returns results
	'''

	# Sort data array, create x and y arrays
	sorted_data = np.sort(array1)
	xdata = np.concatenate([sorted_data[::-1]])
	ydata = np.arange(sorted_data.size) + 1

	# Filter array to those > complimit, output results
	filt_array = np.where(sorted_data >= complimits_val)
	data_lim = sorted_data[filt_array]
	sorted_data_lim = np.sort(data_lim)

	# Output results
	xdata_lim = np.concatenate([sorted_data_lim[::-1]])
	ydata_lim = np.arange(sorted_data_lim.size) + 1
	xdata_lim_float = xdata_lim.astype(float)
	ydata_lim_float = ydata_lim.astype(float)
	
	# Print result to screen
	print('CDF Function Results {:.2e}:'.format(complimits_val))
	print('[Len: {}, Min: {:.2e}, Max:{:.2e}]'.format(len(xdata), np.nanmin(xdata), np.nanmax(xdata)))
	print('[Len: {}, Min: {:.2e}, Max:{:.2e}]'.format(len(xdata_lim_float), np.nanmin(xdata_lim_float), np.nanmax(xdata_lim_float)))

	return xdata, ydata, xdata_lim_float, ydata_lim_float

def galnameoutfun(galname):

	'''
	Function: Create output galaxy name
	'''

	# Change Output Name for Special Cases
	if galname == 'PowerLaw_ContLog':
		galnameout = 'PL'
	elif galname == 'PowerLaw_ContLog_Ant':
		galnameout = 'PL (Ant)'
	elif galname == 'PowerLaw_ContLog_LMC':
		galnameout = 'PL (LMC)'
	elif galname == 'PowerLaw_MError_ContLog':
		galnameout = 'PL w/ 15% Err'
	elif galname == 'PowerLaw_MError_ContLog_Ant':
		galnameout = 'PL (Ant) w/ 15% Err'
	elif galname == 'PowerLaw_MError_ContLog_LMC':
		galnameout = 'PL (LMC) w/ 15% Err'
	elif galname == 'Schechter5_ContLog':
		galnameout = 'Sch. (1E5)'
	elif galname == 'Schechter5_ContLog_Ant':
		galnameout = 'Sch. (1E5, Ant)'
	elif galname == 'Schechter5_ContLog_LMC':
		galnameout = 'Sch. (1E5, LMC)'
	elif galname == 'Schechter5_MError_ContLog':
		galnameout = 'Sch. (1E5) w/ 15% Err'
	elif galname == 'Schechter5_MError_ContLog_Ant':
		galnameout = 'Sch. (1E5, Ant) w/ 15% Err'
	elif galname == 'Schechter5_MError_ContLog_LMC':
		galnameout = 'Sch. (1E5, LMC) w/ 15% Err'
	elif galname == 'Schechter6_ContLog':
		galnameout = 'Sch. (1E6)'
	elif galname == 'Schechter6_ContLog_Ant':
		galnameout = 'Sch. (1E6, Ant)'
	elif galname == 'Schechter6_ContLog_LMC':
		galnameout = 'Sch. (1E6, LMC)'
	elif galname == 'Schechter6_MError_ContLog':
		galnameout = 'Sch. (1E6) w/ 15% Err'
	elif galname == 'Schechter6_MError_ContLog_Ant':
		galnameout = 'Sch. (1E6, Ant) w/ 15% Err'
	elif galname == 'Schechter6_MError_ContLog_LMC':
		galnameout = 'Sch. (1E6, LMC) w/ 15% Err'
	elif galname == 'Schechter6_MError2_ContLog':
		galnameout = 'Sch. (1E6) w/ 50% Err'
	elif galname == 'Schechter6_MError2_ContLog_Ant':
		galnameout = 'Sch. (1E6, Ant) w/ 50% Err'
	elif galname == 'Schechter6_MError2_ContLog_LMC':
		galnameout = 'Sch. (1E6, LMC) w/ 50% Err'
	elif galname == 'Schechter7_ContLog':
		galnameout = 'Sch. (1E7)'
	elif galname == 'Schechter7_ContLog_Ant':
		galnameout = 'Sch. (1E7, Ant)'
	elif galname == 'Schechter7_ContLog_LMC':
		galnameout = 'Sch. (1E7, LMC)'
	elif galname == 'Schechter7_MError_ContLog':
		galnameout = 'Sch. (1E7) w/ 15% Err'
	elif galname == 'Schechter7_MError_ContLog_Ant':
		galnameout = 'Sch. (1E7, Ant) w/ 15% Err'
	elif galname == 'Schechter7_MError_ContLog_LMC':
		galnameout = 'Sch. (1E7, LMC) w/ 15% Err'
	elif galname in ['M101_All', 'M101_XB00A', 'M101_XB00B', 'M101_XB00C', 'M101_XB01', 'M101_XB02', 'M101_XB03', 'M101_XB04', 'M101_XB05', 'M101_XB06', 'M101_XB07', 'M101_XB08', 'M101_XB09', 'M101_XB10', 'M101_XB11', 'M101L1', 'M101L2', 'M101L1_cut', 'M101L2_cut']:
		galnameout = 'M101'
	elif galname in ['M83_All', 'M83_XB00A', 'M83_XB00B', 'M83_XB00C', 'M83_XB01', 'M83_XB02', 'M83_XB03', 'M83_XB04', 'M83_XB05', 'M83_XB06', 'M83_XB07', 'M83_XB08', 'M83_XB09', 'M83_XB10', 'M83_XB11', 'M83_cut', 'M83_cut_N']:
		galnameout = 'M83'
	elif galname == 'M83_Q2_XB_H':
		galnameout = 'M83 - HMXB'
	elif galname == 'M83_Q2_XB_I':
		galnameout = 'M83 - IMXB'
	elif galname == 'M83_Q2_XB_L':
		galnameout = 'M83 - LMXB'
	elif galname == 'M83_Q2_XB_LI':
		galnameout = 'M83 - LMXB + IMXB'
	elif galname == 'M83_Q2_XB_A':
		galnameout = 'M83 - All'
	elif galname in  ['Dwarf2', 'Dwarf2_Low', 'Dwarf2_Med', 'Dwarf2_High', 'Dwarf2_NoSB']:
		galnameout = 'Dwarf'
	elif galname in ['M83_SV_R1', 'M83_SV_R2', 'M83_SV_R3', 'M83_SV_R4', 'M83_R1', 'M83_R2', 'M83_R3', 'M83_R4']:
		galnameout = 'M83'
	elif galname in ['Dwarf2_NGC4656', 'Dwarf2_NGC4449', 'Dwarf2_NGC5253', 'Dwarf2_NGC3738', 'Dwarf2_UGC1249']:
		galnameout = galnameout.lstrip('Dwarf2_')
	elif galname in ['NGC3351_SCP_C', 'NGC3351_SCP_B', 'NGC3351_SCP3_C', 'NGC3351_SCP3_B']:
		galnameout = 'NGC3351'
	else:
		galname_tmp1 = galname.rstrip('_LEG2')
		galname_tmp2 = galname_tmp1.rstrip('_LEG')
		galname_tmp3 = galname_tmp2.rstrip('_cut')
		galname_tmp4 = galname_tmp3.rstrip('_N')
		galname_tmp5 = galname_tmp4.rstrip('_IR')
		galname_tmp6 = galname_tmp5.rstrip('') # _HMXB
		galname_tmp7 = galname_tmp6.rstrip('') # _IMXB
		galname_tmp8 = galname_tmp7.rstrip('') # _LMXB
		galname_tmp9 = galname_tmp8.rstrip('_spec')
		galname_tmp10 = galname_tmp9.rstrip('_comp')
		galname_tmp11 = galname_tmp10.rstrip('_comp_cut')
		galname_tmp12 = galname_tmp11.rstrip('_P')
		galnameout = galname_tmp12.rstrip('_SV')

	# Extra Final Work to Remove to Final 2
	if galnameout == 'NGC424':
		galnameout = 'NGC4242'
	if galnameout == 'NGC3627_2':
		galnameout = 'NGC3627'
	if galnameout == 'Antenna':
		galnameout = 'Antennae'

	return galnameout

#------------------------------------------------------------------------------
###
# (10) Code Snippets (Old - masteranalysisfunction)
###

def masteranalysisfunction(galname, gal_array, range_mass, typeflag, complimits, flag_idl, flag_likelihood):

	'''
	Function: Make plots/functions for single galaxy
	>>> Typeflags
	0 = Clump
	0.5 = XB
	1 = GMC
	2 = SC
	3 = SC (simulated)
	'''

	# Print type of analysis chosen
	print('>>>')
	if typeflag == 0:
		typename = 'C'
		print('>>> Start - C Analysis')
	elif typeflag == 0.5:
		typename = 'XB'
		print('>>> Start - XB Analysis')
	elif typeflag == 1:
		typename = 'GMC'
		print('>>> Start - GMC Analysis')
	elif typeflag == 2:
		typename = 'SC'
		print('>>> Start - SC Analysis')
	elif typeflag == 3:
		typename = 'SimSC'
		print('>>> Start - SimSC Analysis')
	else:
		typename = 'Summary'
		print('>>> Start - Summary Analysis')
	print('>>>')
	print('Array Variable + Length: {} - {}'.format(galname, len(gal_array)))

	# Set flag whether it is an XB dataset
	if 'XB' in galname or typeflag == 0.5:
		flag_XB = 1
		print('Flag XB is on')
	else:
		flag_XB = 0
		print('Flag XB is off')

	# Set up parameters
	galnameout = galnameoutfun(galname)
	print ('Galnameout Variable: {}'.format(galnameout))

	# Define bins for the following code
	mass_bins_log = np.power(10, np.linspace(2, 9, num = 29))
	mass_bins_log_test1 = np.power(10, np.linspace(2, 9, num = 43))
	mass_bins_log_test2 = np.power(10, np.linspace(2, 9, num = 15))
	age_bins_log = np.power(10, np.linspace(4, 11, num = 29))
	mass_bins_log_plot = np.power(10, np.linspace(np.log10(np.min(range_mass)), np.log10(np.max(range_mass))))

	# If Flag = SC or SimSC
	if typeflag > 1:

		# Create classic 3 age bins
		print('Input Completeness Limits: {:.1e}, {:.1e}, {:.1e}'.format(complimits[0], complimits[1], complimits[2]))
		print('1 = (0, 10] Myr + input completeness limit')
		gal_array_age1 = gal_array[gal_array[:,6] <= 10.01*1E6]
		gal_array_age1_masslimit = gal_array_age1[gal_array_age1[:,4] > complimits[0]]
		age1_label = r' ($\tau \leq$ 10 Myr)'
		print('2 = (10, 100] Myr + input completeness limit')
		gal_array_age2_tmp = gal_array[gal_array[:,6] > 10.01*1E6]
		gal_array_age2 = gal_array_age2_tmp[gal_array_age2_tmp[:,6] <= 100.01*1E6]
		gal_array_age2_masslimit = gal_array_age2[gal_array_age2[:,4] > complimits[1]]
		age2_label = r' (10 < $\tau \leq$ 100 Myr)'
		print('3 = (100, 400] Myr + input completeness limit')
		gal_array_age3_tmp = gal_array[gal_array[:,6] > 100.01*1E6]
		gal_array_age3 = gal_array_age3_tmp[gal_array_age3_tmp[:,6] <= 400.01*1E6]
		gal_array_age3_masslimit = gal_array_age3[gal_array_age3[:,4] > complimits[2]]
		age3_label = r' (100 < $\tau \leq$ 400 Myr)'
		print('Number by Age in Bins: ({}/{}/{}), Total = {}'.format(len(gal_array_age1), len(gal_array_age2), len(gal_array_age3), len(gal_array_age1) + len(gal_array_age2) + len(gal_array_age3)))
		print('Fraction Above Mass Completeness Limits: {} / {}, {} / {}, {} / {}, Total = {} / {}'.format(len(gal_array_age1_masslimit), len(gal_array_age1), len(gal_array_age2_masslimit), len(gal_array_age2), len(gal_array_age3_masslimit), len(gal_array_age3), len(gal_array_age1_masslimit) + len(gal_array_age2_masslimit) + len(gal_array_age3_masslimit), len(gal_array_age1) + len(gal_array_age2) + len(gal_array_age3)))
		gal_array_masslimit = np.concatenate([gal_array_age1_masslimit, gal_array_age2_masslimit, gal_array_age3_masslimit])

		# Create test age bins (can be modified)
		print('Test 1 = (0, 5] Myr + input completeness limit')
		gal_array_test1 = gal_array[gal_array[:,6] <= 5.01*1E6]
		gal_array_test1_masslimit = gal_array_test1[gal_array_test1[:,4] > complimits[0]]
		test1_label = r' ($\tau \leq$ 5 Myr)'
		print('Test 2 = (5, 10] Myr + input completeness limit')
		gal_array_test2_tmp = gal_array[gal_array[:,6] > 5.01*1E6]
		gal_array_test2 = gal_array_test2_tmp[gal_array_test2_tmp[:,6] <= 10.01*1E6]
		gal_array_test2_masslimit = gal_array_test2[gal_array_test2[:,4] > complimits[0]]
		test2_label = r' (5 < $\tau \leq$ 10 Myr)'
		
		# Create other age bins
		print('5 = [1, 200] Myr + input completeness limit 100-400 Myr')
		gal_array_age5_tmp = gal_array[gal_array[:,6] > 0.99 * 1E6]
		gal_array_age5 = gal_array_age5_tmp[gal_array_age5_tmp[:,6] <= 200.01 * 1E6]
		age5_masslimit = complimits[2]
		gal_array_age5_masslimit = gal_array_age5[gal_array_age5[:,4] > age5_masslimit]
		age5_label = r' (1 <= $\tau \leq$ 200 Myr)'
		print('6 = (100, 200] Myr + input completeness limit 100-400 Myr')
		gal_array_age6_tmp = gal_array[gal_array[:,6] > 100.01 * 1E6]
		gal_array_age6 = gal_array_age6_tmp[gal_array_age6_tmp[:,6] <= 200.01 * 1E6]
		age6_masslimit = complimits[2]
		age6_label = r' (100 < $\tau \leq$ 200 Myr)'
		gal_array_age6_masslimit = gal_array_age6[gal_array_age6[:,4] > age6_masslimit]
		print('7 = (0, 100] Myr  + Completeness Limit for 10-100 Myr')
		age7_label = r' ($\tau \leq$ 100 Myr)'
		gal_array_age7 = gal_array[gal_array[:,6] <= 100.01 * 1E6]
		age7_masslimit = complimits[1]
		gal_array_age7_masslimit = gal_array_age7[gal_array_age7[:,4] > age7_masslimit]
		
		# Switch
		if False:
			print('7 = [30, 400] Myr  + Completeness Limit for 100-400 Myr')
			gal_array_age7a_tmp = gal_array[gal_array[:,6] >= 29.99 * 1E6]
			gal_array_age7a = gal_array_age7a_tmp[gal_array_age7a_tmp[:,6] <= 400.01 * 1E6]
			gal_array_age7a_masslimit = gal_array_age7a[gal_array_age7a[:,4] > complimits[2]]
			age7a_label = r' (30 <= $\tau \leq$ 400 Myr)'
			print('7 = [50, 400] Myr  + Completeness Limit for 100-400 Myr')
			gal_array_age7b_tmp = gal_array[gal_array[:,6] >= 49.99 * 1E6]
			gal_array_age7b = gal_array_age7b_tmp[gal_array_age7b_tmp[:,6] <= 400.01 * 1E6]
			gal_array_age7b_masslimit = gal_array_age7b[gal_array_age7b[:,4] > complimits[2]]
			age7b_label = r' (50 <= $\tau \leq$ 400 Myr)'
			print('7 = [80, 400] Myr  + Completeness Limit for 100-400 Myr')
			gal_array_age7c_tmp = gal_array[gal_array[:,6] >= 79.99 * 1E6]
			gal_array_age7c = gal_array_age7c_tmp[gal_array_age7c_tmp[:,6] <= 400.01 * 1E6]
			gal_array_age7c_masslimit = gal_array_age7c[gal_array_age7c[:,4] > complimits[2]]
			age7c_label = r' (80 <= $\tau \leq$ 400 Myr)'
		
		print('8 = [100, 200] Myr + 5000 M')
		gal_array_age8_tmp = gal_array[gal_array[:,6] >= 99.99 * 1E6]
		gal_array_age8 = gal_array_age8_tmp[gal_array_age8_tmp[:,6] <= 200.01 * 1E6]
		age8_masslimit = 5000
		gal_array_age8_masslimit = gal_array_age8[gal_array_age8[:,4] > age8_masslimit]
		age8_label = r' (100 <= $\tau \leq$ 200 Myr)'
		print('9 = (0, 100] + 5000 M')
		gal_array_age9 = gal_array[gal_array[:,6] <= 100.01 * 1E6]
		age9_masslimit = 5000
		gal_array_age9_masslimit = gal_array_age9[gal_array_age9[:,4] > age9_masslimit]
		age9_label = r' ($\tau \leq$ 100 Myr)'

	# Else Flag = GMC
	else:

		# Just output full array above mass limit
		gal_array_masslimit = gal_array[gal_array[:,4] > np.nanmax(complimits)]

	# Create different mass bins
	gal_array_mass1 = gal_array[gal_array[:,4] <= 1E4]
	gal_array_mass2_tmp = gal_array[gal_array[:,4] > 1E4]
	gal_array_mass2 = gal_array_mass2_tmp[gal_array_mass2_tmp[:,4] <= 1E6]
	gal_array_mass3 = gal_array[gal_array[:,4] > 1E6]
	gal_array_len = len(gal_array_mass1) + len(gal_array_mass2) + len(gal_array_mass3)
	print('Number by Mass: ({}/{}/{}) --> Total = {}'.format(len(gal_array_mass1), len(gal_array_mass2), len(gal_array_mass3), gal_array_len))

	# Creating Equal Width Bins
	print('>>>')
	print('>>> Creating Equal Width Bins')
	print('>>>')
	print('- All Clusters Above Mass Limit:')
	n, bins, bins_width, bins_centre, n_fit, bins_fit, n_dM, n_fit_dM, n_dlogM, n_fit_dlogM, ncum, ncum_fit, n_fit_err, n_fit_dM_err, n_fit_dlogM_err = makearrayhist(gal_array_masslimit, mass_bins_log, np.nanmax(complimits))
	print('- Clusters in Mass Bin 1:')
	n_mass1, bins_mass1, bins_width_mass1, bins_centre_mass1, n_fit_mass1, bins_fit_mass1, n_dM_mass1, n_fit_dM_mass1, n_dlogM_mass1, n_fit_dlogM_mass1, ncum_mass1, ncum_fit_mass1, n_fit_mass1_err, n_fit_dM_mass1_err, n_fit_dlogM_mass1_err = makearrayhist(gal_array_mass1, mass_bins_log, np.nanmax(complimits))
	print('- Clusters in Mass Bin 2:')
	n_mass2, bins_mass2, bins_width_mass2, bins_centre_mass2, n_fit_mass2, bins_fit_mass2, n_dM_mass2, n_fit_dM_mass2, n_dlogM_mass2, n_fit_dlogM_mass2, ncum_mass2, ncum_fit_mass2, n_fit_mass2_err, n_fit_dM_mass2_err, n_fit_dlogM_mass2_err = makearrayhist(gal_array_mass2, mass_bins_log, np.nanmax(complimits))
	print('- Clusters in Mass Bin 3:')
	n_mass3, bins_mass3, bins_width_mass3, bins_centre_mass3, n_fit_mass3, bins_fit_mass3, n_dM_mass3, n_fit_dM_mass3, n_dlogM_mass3, n_fit_dlogM_mass3, ncum_mass3, ncum_fit_mass3, n_fit_mass3_err, n_fit_dM_mass3_err, n_fit_dlogM_mass3_err = makearrayhist(gal_array_mass3, mass_bins_log, np.nanmax(complimits))
	if typeflag > 1:
		print('- Clusters in Age Bin 1:')
		n_age1, bins_age1, bins_width_age1, bins_centre_age1, n_fit_age1, bins_fit_age1, n_dM_age1, n_fit_dM_age1, n_dlogM_age1, n_fit_dlogM_age1, ncum_age1, ncum_fit_age1, n_fit_age1_err, n_fit_dM_age1_err, n_fit_dlogM_age1_err = makearrayhist(gal_array_age1_masslimit, mass_bins_log, complimits[0])
		print('- Clusters in Age Bin 2:')
		n_age2, bins_age2, bins_width_age2, bins_centre_age2, n_fit_age2, bins_fit_age2, n_dM_age2, n_fit_dM_age2, n_dlogM_age2, n_fit_dlogM_age2, ncum_age2, ncum_fit_age2, n_fit_age2_err, n_fit_dM_age2_err, n_fit_dlogM_age2_err = makearrayhist(gal_array_age2_masslimit, mass_bins_log, complimits[1])
		print('- Clusters in Age Bin 3:')
		n_age3, bins_age3, bins_width_age3, bins_centre_age3, n_fit_age3, bins_fit_age3, n_dM_age3, n_fit_dM_age3, n_dlogM_age3, n_fit_dlogM_age3, ncum_age3, ncum_fit_age3, n_fit_age3_err, n_fit_dM_age3_err, n_fit_dlogM_age3_err = makearrayhist(gal_array_age3_masslimit, mass_bins_log, complimits[2])
		print('- All Clusters Above Mass Limit (Test 1):')
		test1_n, test1_bins, test1_bins_width, test1_bins_centre, test1_n_fit, test1_bins_fit, test1_n_dM, test1_n_fit_dM, test1_n_dlogM, test1_n_fit_dlogM, test1_ncum, test1_ncum_fit, test1_n_fit_err, test1_n_fit_dM_err, test1_n_fit_dlogM_err = makearrayhist(gal_array_test1_masslimit, mass_bins_log_test1, complimits[0])
		print('- All Clusters Above Mass Limit (Test 2):')
		test2_n, test2_bins, test2_bins_width, test2_bins_centre, test2_n_fit, test2_bins_fit, test2_n_dM, test2_n_fit_dM, test2_n_dlogM, test2_n_fit_dlogM, test2_ncum, test2_ncum_fit, test2_n_fit_err, test2_n_fit_dM_err, test3_n_fit_dlogM_err = makearrayhist(gal_array_test2_masslimit, mass_bins_log_test2, complimits[0])
		print('- Clusters in Test Age Bin 5:')
		n_age5, bins_age5, bins_width_age5, bins_centre_age5, n_fit_age5, bins_fit_age5, n_dM_age5, n_fit_dM_age5, n_dlogM_age5, n_fit_dlogM_age5, ncum_age5, ncum_fit_age5, n_fit_age5_err, n_fit_dM_age5_err, n_fit_dlogM_age5_err = makearrayhist(gal_array_age5_masslimit, mass_bins_log, age5_masslimit)
		print('- Clusters in Test Age Bin 6:')
		n_age6, bins_age6, bins_width_age6, bins_centre_age6, n_fit_age6, bins_fit_age6, n_dM_age6, n_fit_dM_age6, n_dlogM_age6, n_fit_dlogM_age6, ncum_age6, ncum_fit_age6, n_fit_age6_err, n_fit_dM_age6_err, n_fit_dlogM_age6_err = makearrayhist(gal_array_age6_masslimit, mass_bins_log, age6_masslimit)
		print('- Clusters in Test Age Bin 7:')
		n_age7, bins_age7, bins_width_age7, bins_centre_age7, n_fit_age7, bins_fit_age7, n_dM_age7, n_fit_dM_age7, n_dlogM_age7, n_fit_dlogM_age7, ncum_age7, ncum_fit_age7, n_fit_age7_err, n_fit_dM_age7_err, n_fit_dlogM_age7_err = makearrayhist(gal_array_age7_masslimit, mass_bins_log, age7_masslimit)
		print('- Clusters in Test Age Bin 8:')
		n_age8, bins_age8, bins_width_age8, bins_centre_age8, n_fit_age8, bins_fit_age8, n_dM_age8, n_fit_dM_age8, n_dlogM_age8, n_fit_dlogM_age8, ncum_age8, ncum_fit_age8, n_fit_age8_err, n_fit_dM_age8_err, n_fit_dlogM_age8_err = makearrayhist(gal_array_age8_masslimit, mass_bins_log, age8_masslimit)
		print('- Clusters in Test Age Bin 9:')
		n_age9, bins_age9, bins_width_age9, bins_centre_age9, n_fit_age9, bins_fit_age9, n_dM_age9, n_fit_dM_age9, n_dlogM_age9, n_fit_dlogM_age9, ncum_age9, ncum_fit_age9, n_fit_age9_err, n_fit_dM_age9_err, n_fit_dlogM_age9_err = makearrayhist(gal_array_age9_masslimit, mass_bins_log, age9_masslimit)
	
	# Creating Equal Number Bins
	print('>>>')
	print('>>> Creating Equal Number Bins')
	print('>>>')
	print('- All Clusters Above Mass Limit:')
	nequal, binsequal, binsequal_width, binsequal_centre, nequal_fit, binsequal_fit, nequal_dM, nequal_fit_dM, nequal_dlogM, nequal_fit_dlogM, nequalcum, nequalcum_fit, nequal_fit_err, nequal_fit_dM_err, nequal_fit_dlogM_err = makearrayhistequal(gal_array_masslimit, np.nanmax(complimits), 0)
	if typeflag > 1:
		#if galname != 'M31':
		print('- Clusters in Age Bin 1:')
		nequal_age1, binsequal_age1, binsequal_width_age1, binsequal_centre_age1, nequal_fit_age1, binsequal_fit_age1, nequal_dM_age1, nequal_fit_dM_age1, nequal_dlogM_age1, nequal_fit_dlogM_age1, nequalcum_age1, nequalcum_fit_age1, nequal_fit_age1_err, nequal_fit_dM_age1_err, nequal_fit_dlogM_age1_err = makearrayhistequal(gal_array_age1_masslimit, complimits[0], 0)
		print('- Clusters in Age Bin 2:')
		nequal_age2, binsequal_age2, binsequal_width_age2, binsequal_centre_age2, nequal_fit_age2, binsequal_fit_age2, nequal_dM_age2, nequal_fit_dM_age2, nequal_dlogM_age2, nequal_fit_dlogM_age2, nequalcum_age2, nequalcum_fit_age2, nequal_fit_age2_err, nequal_fit_dM_age2_err, nequal_fit_dlogM_age2_err = makearrayhistequal(gal_array_age2_masslimit, complimits[1], 0)
		print('- Clusters in Age Bin 3:')
		nequal_age3, binsequal_age3, binsequal_width_age3, binsequal_centre_age3, nequal_fit_age3, binsequal_fit_age3, nequal_dM_age3, nequal_fit_dM_age3, nequal_dlogM_age3, nequal_fit_dlogM_age3, nequalcum_age3, nequalcum_fit_age3, nequal_fit_age3_err, nequal_fit_dM_age3_err, nequal_fit_dlogM_age3_err = makearrayhistequal(gal_array_age3_masslimit, complimits[2], 0)
		print('- All Clusters Above Mass Limit (Test 1):')
		test1_nequal, test1_binsequal, test1_binsequal_width, test1_binsequal_centre, test1_nequal_fit, test1_binsequal_fit, test1_nequal_dM, test1_nequal_fit_dM, test1_nequal_dlogM, test1_nequal_fit_dlogM, test1_nequalcum, test1_nequalcum_fit, test1_nequal_fit_err, test1_nequal_fit_dM_err, test1_nequal_fit_dlogM_err = makearrayhistequal(gal_array_test1_masslimit, complimits[0], 1)
		print('- All Clusters Above Mass Limit (Test 2):')
		test2_nequal, test2_binsequal, test2_binsequal_width, test2_binsequal_centre, test2_nequal_fit, test2_binsequal_fit, test2_nequal_dM, test2_nequal_fit_dM, test2_nequal_dlogM, test2_nequal_fit_dlogM, test2_nequalcum, test2_nequalcum_fit, test2_nequal_fit_err, test2_nequal_fit_dM_err, test3_nequal_fit_dlogM_err = makearrayhistequal(gal_array_test2_masslimit, complimits[0], 2)
		print('- Clusters in Test Age Bin 5:')
		nequal_age5, binsequal_age5, binsequal_width_age5, binsequal_centre_age5, nequal_fit_age5, binsequal_fit_age5, nequal_dM_age5, nequal_fit_dM_age5, nequal_dlogM_age5, nequal_fit_dlogM_age5, nequalcum_age5, nequalcum_fit_age5, nequal_fit_age5_err, nequal_fit_dM_age5_err, nequal_fit_dlogM_age5_err = makearrayhistequal(gal_array_age5_masslimit, age5_masslimit, 0)
		print('- Clusters in Test Age Bin 6:')
		nequal_age6, binsequal_age6, binsequal_width_age6, binsequal_centre_age6, nequal_fit_age6, binsequal_fit_age6, nequal_dM_age6, nequal_fit_dM_age6, nequal_dlogM_age6, nequal_fit_dlogM_age6, nequalcum_age6, nequalcum_fit_age6, nequal_fit_age6_err, nequal_fit_dM_age6_err, nequal_fit_dlogM_age6_err = makearrayhistequal(gal_array_age6_masslimit, age6_masslimit, 0)
		print('- Clusters in Test Age Bin 7:')
		nequal_age7, binsequal_age7, binsequal_width_age7, binsequal_centre_age7, nequal_fit_age7, binsequal_fit_age7, nequal_dM_age7, nequal_fit_dM_age7, nequal_dlogM_age7, nequal_fit_dlogM_age7, nequalcum_age7, nequalcum_fit_age7, nequal_fit_age7_err, nequal_fit_dM_age7_err, nequal_fit_dlogM_age7_err = makearrayhistequal(gal_array_age7_masslimit, age7_masslimit, 0)
		print('- Clusters in Test Age Bin 8:')
		nequal_age8, binsequal_age8, binsequal_width_age8, binsequal_centre_age8, nequal_fit_age8, binsequal_fit_age8, nequal_dM_age8, nequal_fit_dM_age8, nequal_dlogM_age8, nequal_fit_dlogM_age8, nequalcum_age8, nequalcum_fit_age8, nequal_fit_age8_err, nequal_fit_dM_age8_err, nequal_fit_dlogM_age8_err = makearrayhistequal(gal_array_age8_masslimit, age8_masslimit, 0)
		print('- Clusters in Test Age Bin 9:')
		nequal_age9, binsequal_age9, binsequal_width_age9, binsequal_centre_age9, nequal_fit_age9, binsequal_fit_age9, nequal_dM_age9, nequal_fit_dM_age9, nequal_dlogM_age9, nequal_fit_dlogM_age9, nequalcum_age9, nequalcum_fit_age9, nequal_fit_age9_err, nequal_fit_dM_age9_err, nequal_fit_dlogM_age9_err = makearrayhistequal(gal_array_age9_masslimit, age9_masslimit, 0)

	# Find a new completeness limit to be 1% of the maximum (GMC) mass
	percentile_gal_value = np.nanmax(gal_array[:,4]) * 0.05
	print('5 percent of maximum mass: {:.3e} or log M = {:.2f}'.format(percentile_gal_value, np.log10(percentile_gal_value)))

	# Flag = SC or GMC and not a combination of galaxies
	# Plot location in the sky
	if typeflag < 3 and galname not in ['Dwarf', 'Dwarf2', 'Dwarf2_Low', 'Dwarf2_Med', 'Dwarf2_High', 'Dwarf2_NoSB'] and '_12' not in galname and '_13' not in galname and typeflag != 0.5:

		print('>>> Creating DSS Plots')
		RA, DEC, PA, LONGAXIS, SHORTAXIS, REGPATH, POLYPATH = returnposinfo(galnameout)

		### --==--==--==-- ###
		plt.rcParams['xtick.labelsize'] = 18
		plt.rcParams['ytick.labelsize'] = 18

		# Special case for Antennae (two galaxies)
		# if galnameout == 'Antennae':
		if True:
		# else:

			### --==--==--==-- ###
			print('>>>')
			print('>>> Radius_' + galname)
			print('>>>')
			gc = aplpy.FITSFigure(DSS_folder + galnameout + '_DSS.fits')
			gc.show_grayscale()
			gc.set_title(galnameout)
			if galname != 'M31':
				gc.recenter(RA, DEC + (0.25 * LONGAXIS), LONGAXIS * plotstretchfactor)
				gc.show_ellipses(RA, DEC, LONGAXIS, SHORTAXIS, angle = PA, facecolor = 'None', edgecolor = 'green')
				gc.show_ellipses(RA, DEC, 0.5*LONGAXIS, 0.5*SHORTAXIS, angle = PA, facecolor = 'None', edgecolor = 'green')
				gc.show_ellipses(RA, DEC, 0.1*LONGAXIS, 0.1*SHORTAXIS, angle = PA, facecolor = 'None', edgecolor = 'green')
			gc.save('./FiguresSummary/Radius/Radius_' + galname + '.png')
			gc.close()

			if typeflag == 2:

				### --==--==--==-- ###
				print('>>>')
				print('>>> ' + galname + '_P0A1_' + typename + '_Position_Age_01')
				print('>>>')
				gc = aplpy.FITSFigure(DSS_folder + galnameout + '_DSS.fits')
				gc.show_grayscale()
				gc.set_title(galnameout)
				if galname != 'M31':
					gc.recenter(RA, DEC + (0.25 * LONGAXIS), LONGAXIS * plotstretchfactor)
				if len(gal_array_age1_masslimit[:,2]) > 0:
					gc.show_markers(np.asarray(gal_array_age1_masslimit[:,2], dtype = np.float32), np.asarray(gal_array_age1_masslimit[:,3], dtype = np.float32), marker = 'o', c = 'blue', s = 4, alpha = 0.95, label = age1_label)
				if len(gal_array_age2_masslimit[:,2]) > 0:
					gc.show_markers(np.asarray(gal_array_age2_masslimit[:,2], dtype = np.float32), np.asarray(gal_array_age2_masslimit[:,3], dtype = np.float32), marker = 'o', c = 'green', s = 4, alpha = 0.95, label = age2_label)
				if len(gal_array_age3_masslimit[:,2]) > 0:
					gc.show_markers(np.asarray(gal_array_age3_masslimit[:,2], dtype = np.float32), np.asarray(gal_array_age3_masslimit[:,3], dtype = np.float32), marker = 'o', c = 'red', s = 4, alpha = 0.95, label = age3_label)
				plt.legend(loc = 'upper right', title = 'N = {}'.format(len(gal_array_masslimit)))
				gc.show_polygons(POLYPATH, lw = 2, alpha = 0.5)
				gc.save('./Figures' + typename + '/' + galname + '_P0A1_' + typename + '_Position_Age_01.png')
				gc.close()

				### --==--==--==-- ###
				print('>>>')
				print('>>> ' + galname + '_P0A1_' + typename + '_Position_Age_02')
				print('>>>')
				gc = aplpy.FITSFigure(DSS_folder + galnameout + '_DSS.fits')
				gc.show_grayscale()
				gc.set_title(galnameout)
				if galname != 'M31':
					gc.recenter(RA, DEC + (0.25 * LONGAXIS), LONGAXIS * plotstretchfactor)
				if len(gal_array_age1[:,2]) > 0:
					gc.show_markers(np.asarray(gal_array_age1[:,2], dtype = np.float32), np.asarray(gal_array_age1[:,3], dtype = np.float32), marker = 'o', c = 'blue', s = 4, alpha = 0.95, label = age1_label)
				if len(gal_array_age2[:,2]) > 0:
					gc.show_markers(np.asarray(gal_array_age2[:,2], dtype = np.float32), np.asarray(gal_array_age2[:,3], dtype = np.float32), marker = 'o', c = 'green', s = 4, alpha = 0.95, label = age2_label)
				if len(gal_array_age3[:,2]) > 0:
					gc.show_markers(np.asarray(gal_array_age3[:,2], dtype = np.float32), np.asarray(gal_array_age3[:,3], dtype = np.float32), marker = 'o', c = 'red', s = 4, alpha = 0.95, label = age3_label)
				plt.legend(loc = 'upper right', title = 'N = {}'.format(len(gal_array)))
				gc.show_polygons(POLYPATH, lw = 2, alpha = 0.5)
				gc.save('./Figures' + typename + '/' + galname + '_P0A1_' + typename + '_Position_Age_02.png')
				gc.close()

			### --==--==--==-- ###
			print('>>>')
			print('>>> ' + galname + '_P0A2_' + typename + '_Position_Mass')
			print('>>>')
			gc = aplpy.FITSFigure(DSS_folder + galnameout + '_DSS.fits')
			gc.show_grayscale()
			gc.set_title(galnameout)
			if galname != 'M31':
				gc.recenter(RA, DEC + (0.25 * LONGAXIS), LONGAXIS * plotstretchfactor)
			if flag_XB != 1:
				if len(gal_array_mass1[:,2]) > 0:
					gc.show_markers(np.asarray(gal_array_mass1[:,2], dtype = np.float32), np.asarray(gal_array_mass1[:,3], dtype = np.float32), marker = 'o', c = 'blue', s = 4, alpha = 0.95, label = r'M <= 10$^{4}$ M$_\odot$')
				if len(gal_array_mass2[:,2]) > 0:
					gc.show_markers(np.asarray(gal_array_mass2[:,2], dtype = np.float32), np.asarray(gal_array_mass2[:,3], dtype = np.float32), marker = 'o', c = 'green', s = 4, alpha = 0.95, label = r'10$^{4}$ M$_\odot$ < M <= 10$^{6}$ M$_\odot$')
				if len(gal_array_mass3[:,2]) > 0:
					gc.show_markers(np.asarray(gal_array_mass3[:,2], dtype = np.float32), np.asarray(gal_array_mass3[:,3], dtype = np.float32), marker = 'o', c = 'red', s = 4, alpha = 0.95, label = r'M > 10$^{6}$ M$_\odot$')
			else:
				if len(gal_array_mass1[:,2]) > 0:
					gc.show_markers(np.asarray(gal_array_mass1[:,2], dtype = np.float32), np.asarray(gal_array_mass1[:,3], dtype = np.float32), marker = 'o', c = 'blue', s = 4, alpha = 0.95, label = r'L <= 10$^{36}$ ergs/s')
				if len(gal_array_mass2[:,2]) > 0:
					gc.show_markers(np.asarray(gal_array_mass2[:,2], dtype = np.float32), np.asarray(gal_array_mass2[:,3], dtype = np.float32), marker = 'o', c = 'green', s = 4, alpha = 0.95, label = r'10$^{36}$ ergs/s < L <= 10$^{38}$ ergs/s')
				if len(gal_array_mass3[:,2]) > 0:
					gc.show_markers(np.asarray(gal_array_mass3[:,2], dtype = np.float32), np.asarray(gal_array_mass3[:,3], dtype = np.float32), marker = 'o', c = 'red', s = 4, alpha = 0.95, label = r'L > 10$^{38}$ ergs/s')
			gc.show_polygons(POLYPATH, lw = 2, alpha = 0.5)
			plt.legend(loc = 'upper right', title = 'N = {}'.format(gal_array_len))
			gc.show_polygons(POLYPATH, lw = 2, alpha = 0.5)
			gc.save('./Figures' + typename + '/' + galname + '_P0A2_' + typename + '_Position_Mass.png')
			gc.close()

	### --==--==--==-- ###
	plt.rcParams['xtick.labelsize'] = 42
	plt.rcParams['ytick.labelsize'] = 42

	# Flag = SC or SimSC
	if typeflag > 1:

		### --==--==--==-- ###
		print('>>>')
		print('>>> ' + galname + '_P1_' + typename + '_Age_Histogram')
		print('>>>')
		fig = plt.figure(figsize = (12, 12))
		ax1 = fig.add_subplot(111)
		n_age, bins_age, patches_age = plt.hist([gal_array_masslimit[:,6]], bins = age_bins_log, color = ['k'], histtype = 'step', stacked = True)
		plt.xscale('log', nonposx = 'clip')
		plt.yscale('log', nonposy = 'clip')
		plt.axis([1E4, 1E11] + [np.power(10, -0.3) * np.nanmin(n_age), np.power(10, 1.3) * np.nanmax(n_age)])
		ax1.xaxis.set_major_formatter(log10_labels_format)
		ax1.yaxis.set_major_formatter(log10_labels_format)
		plt.xlabel(r'log (Age/yr)')
		plt.ylabel(r'log N (per bin)')
		plt.savefig('./Figures' + typename + '/' + galname + '_P1_' + typename + '_Age_Histogram_Ex.png')
		plt.close()

	# Set of plots with equal width histograms	
	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_P2A1_' + typename + '_Histogram')
	print('>>>')
	fig = plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111)
	plt.step(bins, np.append(n[0], n), color = 'k', alpha = 0.5)
	plt.errorbar(bins_fit, n_fit, marker = '^', linestyle = 'None', markerfacecolor = 'g', yerr = n_fit_err)
	curve_fit3(bins_fit, n_fit, range_mass, np.nanmax(complimits), 1, 1, n_fit_err)
	plt.plot([np.nanmax(complimits), np.nanmax(complimits)], [1E-10, 1E10], 'k--')
	if typeflag > 1:
		plt.legend(loc = 'upper right', title = galnameout + r' ($\tau \leq$ 400 Myr)')
	else:
		plt.legend(loc = 'upper right', title = galnameout)
	plt.xscale('log', nonposx = 'clip')
	plt.yscale('log', nonposy = 'clip')
	plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit), np.power(10, 1.3) * np.nanmax(n)])
	ax1.xaxis.set_major_formatter(log10_labels_format)
	ax1.yaxis.set_major_formatter(log10_labels_format)
	if typeflag == 0.5:
		plt.xlabel(r'log Luminosity [$10^{32}$ erg/s]')
	else:
		plt.xlabel(r'log (M/M$_\odot$)')
	plt.ylabel(r'log N (per bin)')
	plt.savefig('./Figures' + typename + '/' + galname + '_P2A1_' + typename + '_Histogram_Ex.png')
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_P2A2_' + typename + '_Histogram_dM')
	print('>>>')
	fig = plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111)
	plt.step(bins, np.append(n_dM[0], n_dM), color = 'k', alpha = 0.5)
	plt.errorbar(bins_fit, n_fit_dM, marker = '^', linestyle = 'None', markerfacecolor = 'g', yerr = n_fit_dM_err)
	curve_fit3(bins_fit, n_fit_dM, range_mass, np.nanmax(complimits), 1, 1, n_fit_dM_err)
	p9, = plt.plot([np.nanmax(complimits), np.nanmax(complimits)], [1E-10, 1E10], 'k--')
	if typeflag > 1:
		plt.legend(loc = 'upper right', title = galnameout + r' ($\tau \leq$ 400 Myr)')
	else:
		plt.legend(loc = 'upper right', title = galnameout)
	plt.xscale('log', nonposx = 'clip')
	plt.yscale('log', nonposy = 'clip')
	plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit_dM), np.power(10, 1.3) * np.nanmax(n_dM)])
	ax1.xaxis.set_major_formatter(log10_labels_format)
	ax1.yaxis.set_major_formatter(log10_labels_format)
	if typeflag == 0.5:
		plt.xlabel(r'log Luminosity [$10^{32}$ erg/s]')
		plt.ylabel(r'log (dN/dL)')
	else:
		plt.xlabel(r'log (M/M$_\odot$)')
		plt.ylabel(r'log (dN/dM)')
	plt.savefig('./Figures' + typename + '/' + galname + '_P2A2_' + typename + '_Histogram_dM_Ex.png')
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_P2A3_' + typename + '_Histogram_dlogM')
	print('>>>')
	fig = plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111)
	plt.step(bins, np.append(n_dlogM[0], n_dlogM), color = 'k', alpha = 0.5)
	plt.errorbar(bins_fit, n_fit_dlogM, marker = '^', linestyle = 'None', markerfacecolor = 'g', yerr = n_fit_dlogM_err)
	curve_fit3(bins_fit, n_fit_dlogM, range_mass, np.nanmax(complimits), 1, 1, n_fit_dlogM_err)
	p9, = plt.plot([np.nanmax(complimits), np.nanmax(complimits)], [1E-10, 1E10], 'k--')
	if typeflag > 1:
		plt.legend(loc = 'upper right', title = galnameout + r' ($\tau \leq$ 400 Myr)')
	else:
		plt.legend(loc = 'upper right', title = galnameout)
	plt.xscale('log', nonposx = 'clip')
	plt.yscale('log', nonposy = 'clip')
	plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit_dlogM), np.power(10, 1.3) * np.nanmax(n_dlogM)])
	ax1.xaxis.set_major_formatter(log10_labels_format)
	ax1.yaxis.set_major_formatter(log10_labels_format)
	if typeflag == 0.5:
		plt.xlabel(r'log Luminosity [$10^{32}$ erg/s]')
		plt.ylabel(r'log (dN/dlogL)')
	else:
		plt.xlabel(r'log (M/M$_\odot$)')
		plt.ylabel(r'log (dN/dlogM)')
	plt.savefig('./Figures' + typename + '/' + galname + '_P2A3_' + typename + '_Histogram_dlogM_Ex.png')
	plt.close()

	# Set of plots with equal width histograms for different age bins
	if typeflag > 1:
	
		if len(n_fit_age1) > 0:

			### --==--==--==-- ###
			print('>>>')
			print('>>> ' + galname + '_P2B1_A1_' + typename + '_Histogram')
			print('>>>')
			fig = plt.figure(figsize = (12, 12))
			ax1 = fig.add_subplot(111)
			plt.step(bins_age1, np.append(n_age1[0], n_age1), color = 'k', alpha = 0.5)
			plt.plot(bins_fit_age1, n_fit_age1, 'g^')
			curve_fit3(bins_fit_age1, n_fit_age1, range_mass, complimits[0], 1, 1, n_fit_age1_err)
			p9, = plt.plot([complimits[0], complimits[0]], [1E-10, 1E10], 'k--')
			plt.legend(loc = 'upper right', title = galnameout + age1_label)
			plt.xscale('log', nonposx = 'clip')
			plt.yscale('log', nonposy = 'clip')
			plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit), np.power(10, 1.3) * np.nanmax(n_age1)])
			ax1.xaxis.set_major_formatter(log10_labels_format)
			ax1.yaxis.set_major_formatter(log10_labels_format)
			plt.xlabel(r'log (M/M$_\odot$)')
			plt.ylabel(r'log N (per bin)')
			plt.savefig('./Figures' + typename + '/' + galname + '_P2B1_A1_' + typename + '_Histogram_Ex.png')
			plt.close()

			### --==--==--==-- ###
			print('>>>')
			print('>>> ' + galname + '_P2B2_A1_' + typename + '_Histogram_dM')
			print('>>>')
			fig = plt.figure(figsize = (12, 12))
			ax1 = fig.add_subplot(111)
			plt.step(bins_age1, np.append(n_dM_age1[0], n_dM_age1), color = 'k', alpha = 0.5)
			plt.plot(bins_fit_age1, n_fit_dM_age1, 'g^')
			curve_fit3(bins_fit_age1, n_fit_dM_age1, range_mass, complimits[0], 1, 1, n_fit_dM_age1_err)
			p9, = plt.plot([complimits[0], complimits[0]], [1E-10, 1E10], 'k--')
			plt.legend(loc = 'upper right', title = galnameout + age1_label)
			plt.xscale('log', nonposx = 'clip')
			plt.yscale('log', nonposy = 'clip')
			plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit_dM), np.power(10, 1.3) * np.nanmax(n_dM_age1)])
			ax1.xaxis.set_major_formatter(log10_labels_format)
			ax1.yaxis.set_major_formatter(log10_labels_format)
			plt.xlabel(r'log (M/M$_\odot$)')
			plt.ylabel(r'log (dN/dM)')
			plt.savefig('./Figures' + typename + '/' + galname + '_P2B2_A1_' + typename + '_Histogram_dM_Ex.png')
			plt.close()

			### --==--==--==-- ###
			print('>>>')
			print('>>> ' + galname + '_P2B3_A1_' + typename + '_Histogram_dlogM')
			print('>>>')
			fig = plt.figure(figsize = (12, 12))
			ax1 = fig.add_subplot(111)
			plt.step(bins_age1, np.append(n_dlogM_age1[0], n_dlogM_age1), color = 'k', alpha = 0.5)
			plt.plot(bins_fit_age1, n_fit_dlogM_age1, 'g^')
			curve_fit3(bins_fit_age1, n_fit_dlogM_age1, range_mass, complimits[0], 1, 1, n_fit_dlogM_age1_err)
			p9, = plt.plot([complimits[0], complimits[0]], [1E-10, 1E10], 'k--')
			plt.legend(loc = 'upper right', title = galnameout + age1_label)
			plt.xscale('log', nonposx = 'clip')
			plt.yscale('log', nonposy = 'clip')
			plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit_dlogM), np.power(10, 1.3) * np.nanmax(n_dlogM_age1)])
			ax1.xaxis.set_major_formatter(log10_labels_format)
			ax1.yaxis.set_major_formatter(log10_labels_format)
			plt.xlabel(r'log (M/M$_\odot$)')
			plt.ylabel(r'log (dN/dlogM)')
			plt.savefig('./Figures' + typename + '/' + galname + '_P2B3_A1_' + typename + '_Histogram_dlogM_Ex.png')
			plt.close()

		if len(n_fit_age2) > 0:

			### --==--==--==-- ###
			print('>>>')
			print('>>> ' + galname + '_P2B1_A2_' + typename + '_Histogram')
			print('>>>')
			fig = plt.figure(figsize = (12, 12))
			ax1 = fig.add_subplot(111)
			plt.step(bins_age2, np.append(n_age2[0], n_age2), color = 'k', alpha = 0.5)
			plt.plot(bins_fit_age2, n_fit_age2, 'g^')
			curve_fit3(bins_fit_age2, n_fit_age2, range_mass, complimits[1], 1, 1, n_fit_age2_err)
			p9, = plt.plot([complimits[1], complimits[1]], [1E-10, 1E10], 'k--')
			plt.legend(loc = 'upper right', title = galnameout + age2_label)
			plt.xscale('log', nonposx = 'clip')
			plt.yscale('log', nonposy = 'clip')
			plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit), np.power(10, 1.3) * np.nanmax(n_age2)])
			ax1.xaxis.set_major_formatter(log10_labels_format)
			ax1.yaxis.set_major_formatter(log10_labels_format)
			plt.xlabel(r'log (M/M$_\odot$)')
			plt.ylabel(r'log N (per bin)')
			plt.savefig('./Figures' + typename + '/' + galname + '_P2B1_A2_' + typename + '_Histogram_Ex.png')
			plt.close()

			if galname not in ['NGC5238']:

				### --==--==--==-- ###
				print('>>>')
				print('>>> ' + galname + '_P2B2_A2_' + typename + '_Histogram_dM')
				print('>>>')
				fig = plt.figure(figsize = (12, 12))
				ax1 = fig.add_subplot(111)
				plt.step(bins_age2, np.append(n_dM_age2[0], n_dM_age2), color = 'k', alpha = 0.5)
				plt.plot(bins_fit_age2, n_fit_dM_age2, 'g^')
				curve_fit3(bins_fit_age2, n_fit_dM_age2, range_mass, complimits[1], 1, 1, n_fit_dM_age2_err)
				p9, = plt.plot([complimits[1], complimits[1]], [1E-10, 1E10], 'k--')
				plt.legend(loc = 'upper right', title = galnameout + age2_label)
				plt.xscale('log', nonposx = 'clip')
				plt.yscale('log', nonposy = 'clip')
				plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit_dM), np.power(10, 1.3) * np.nanmax(n_dM_age2)])
				ax1.xaxis.set_major_formatter(log10_labels_format)
				ax1.yaxis.set_major_formatter(log10_labels_format)
				plt.xlabel(r'log (M/M$_\odot$)')
				plt.ylabel(r'log (dN/dM)')
				plt.savefig('./Figures' + typename + '/' + galname + '_P2B2_A2_' + typename + '_Histogram_dM_Ex.png')
				plt.close()

			### --==--==--==-- ###
			print('>>>')
			print('>>> ' + galname + '_P2B3_A2_' + typename + '_Histogram_dlogM')
			print('>>>')
			fig = plt.figure(figsize = (12, 12))
			ax1 = fig.add_subplot(111)
			plt.step(bins_age2, np.append(n_dlogM_age2[0], n_dlogM_age2), color = 'k', alpha = 0.5)
			plt.plot(bins_fit_age2, n_fit_dlogM_age2, 'g^')
			curve_fit3(bins_fit_age2, n_fit_dlogM_age2, range_mass, complimits[1], 1, 1, n_fit_dlogM_age2_err)
			p9, = plt.plot([complimits[1], complimits[1]], [1E-10, 1E10], 'k--')
			plt.legend(loc = 'upper right', title = galnameout + age2_label)
			plt.xscale('log', nonposx = 'clip')
			plt.yscale('log', nonposy = 'clip')
			plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit_dlogM), np.power(10, 1.3) * np.nanmax(n_dlogM_age2)])
			ax1.xaxis.set_major_formatter(log10_labels_format)
			ax1.yaxis.set_major_formatter(log10_labels_format)
			plt.xlabel(r'log (M/M$_\odot$)')
			plt.ylabel(r'log (dN/dlogM)')
			plt.savefig('./Figures' + typename + '/' + galname + '_P2B3_A2_' + typename + '_Histogram_dlogM_Ex.png')
			plt.close()

		if len(n_fit_age3) > 0 and galname not in ['NGC1705']:

			### --==--==--==-- ###
			print('>>>')
			print('>>> ' + galname + '_P2B1_A3_' + typename + '_Histogram')
			print('>>>')
			fig = plt.figure(figsize = (12, 12))
			ax1 = fig.add_subplot(111)
			plt.step(bins_age3, np.append(n_age3[0], n_age3), color = 'k', alpha = 0.5)
			plt.plot(bins_fit_age3, n_fit_age3, 'g^')
			curve_fit3(bins_fit_age3, n_fit_age3, range_mass, complimits[2], 1, 1, n_fit_age3_err)
			p9, = plt.plot([complimits[2], complimits[2]], [1E-10, 1E10], 'k--')
			plt.legend(loc = 'upper right', title = galnameout + age3_label)
			plt.xscale('log', nonposx = 'clip')
			plt.yscale('log', nonposy = 'clip')
			plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit), np.power(10, 1.3) * np.nanmax(n_age3)])
			ax1.xaxis.set_major_formatter(log10_labels_format)
			ax1.yaxis.set_major_formatter(log10_labels_format)
			plt.xlabel(r'log (M/M$_\odot$)')
			plt.ylabel(r'log N (per bin)')
			plt.savefig('./Figures' + typename + '/' + galname + '_P2B1_A3_' + typename + '_Histogram_Ex.png')
			plt.close()

			### --==--==--==-- ###
			print('>>>')
			print('>>> ' + galname + '_P2B2_A3_' + typename + '_Histogram_dM')
			print('>>>')
			fig = plt.figure(figsize = (12, 12))
			ax1 = fig.add_subplot(111)
			###
			plt.step(bins_age3, np.append(n_dM_age3[0], n_dM_age3), color = 'k', alpha = 0.5)
			plt.plot(bins_fit_age3, n_fit_dM_age3, 'g^')
			curve_fit3(bins_fit_age3, n_fit_dM_age3, range_mass, complimits[2], 1, 1, n_fit_dM_age3_err)
			p9, = plt.plot([complimits[2], complimits[2]], [1E-10, 1E10], 'k--')
			plt.legend(loc = 'upper right', title = galnameout + age3_label)
			plt.xscale('log', nonposx = 'clip')
			plt.yscale('log', nonposy = 'clip')
			plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit_dM), np.power(10, 1.3) * np.nanmax(n_dM_age3)])
			ax1.xaxis.set_major_formatter(log10_labels_format)
			ax1.yaxis.set_major_formatter(log10_labels_format)
			plt.xlabel(r'log (M/M$_\odot$)')
			plt.ylabel(r'log (dN/dM)')
			plt.savefig('./Figures' + typename + '/' + galname + '_P2B2_A3_' + typename + '_Histogram_dM_Ex.png')
			plt.close()

			### --==--==--==-- ###
			print('>>>')
			print('>>> ' + galname + '_P2B3_A3_' + typename + '_Histogram_dlogM')
			print('>>>')
			fig = plt.figure(figsize = (12, 12))
			ax1 = fig.add_subplot(111)
			plt.step(bins_age3, np.append(n_dlogM_age3[0], n_dlogM_age3), color = 'k', alpha = 0.5)
			plt.plot(bins_fit_age3, n_fit_dlogM_age3, 'g^')
			curve_fit3(bins_fit_age3, n_fit_dlogM_age3, range_mass, complimits[2], 1, 1, n_fit_dlogM_age3_err)
			p9, = plt.plot([complimits[2], complimits[2]], [1E-10, 1E10], 'k--')
			plt.legend(loc = 'upper right', title = galnameout + age3_label)
			plt.xscale('log', nonposx = 'clip')
			plt.yscale('log', nonposy = 'clip')
			plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit_dlogM), np.power(10, 1.3) * np.nanmax(n_dlogM_age3)])
			ax1.xaxis.set_major_formatter(log10_labels_format)
			ax1.yaxis.set_major_formatter(log10_labels_format)
			plt.xlabel(r'log (M/M$_\odot$)')
			plt.ylabel(r'log (dN/dlogM)')
			plt.savefig('./Figures' + typename + '/' + galname + '_P2B3_A3_' + typename + '_Histogram_dlogM_Ex.png')
			plt.close()

	# Set of plots with cumulative histograms
	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_P3A_' + typename + '_CHistogram')
	print('>>>')
	fig = plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111)
	plt.step(bins, np.append(ncum[0], ncum), color = 'k', alpha = 0.5)
	plt.plot(bins_fit, ncum_fit, 'g^')
	if galname != 'NGC7793':
		curve_fit3(bins_fit, ncum_fit, range_mass, np.nanmax(complimits), 2, 1, np.sqrt(ncum_fit))
	p9, = plt.plot([np.nanmax(complimits), np.nanmax(complimits)], [1E-10, 1E10], 'k--')
	if typeflag > 1:
		plt.legend(loc = 'upper right', title = galnameout + r' ($\tau \leq$ 400 Myr)')
	else:
		plt.legend(loc = 'upper right', title = galnameout)
	plt.xscale('log', nonposx = 'clip')
	plt.yscale('log', nonposy = 'clip')
	plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(ncum_fit), np.power(10, 1.3) * np.nanmax(ncum)])
	ax1.xaxis.set_major_formatter(log10_labels_format)
	ax1.yaxis.set_major_formatter(log10_labels_format)
	if typeflag == 0.5:
		plt.xlabel(r'log Luminosity [$10^{32}$ erg/s]')
		plt.ylabel(r'log N (> L)')
	else:
		plt.xlabel(r'log (M/M$_\odot$)')
		plt.ylabel(r'log N (> M)')
	plt.savefig('./Figures' + typename + '/' + galname + '_P3A_' + typename + '_CHistogram_Ex.png')
	plt.close()

	if typeflag > 1:

		if len(ncum_fit_age1) > 0:

			### --==--==--==-- ###
			print('>>>')
			print('>>> ' + galname + '_P3B_A1_' + typename + '_CHistogram')
			print('>>>')
			fig = plt.figure(figsize = (12, 12))
			ax1 = fig.add_subplot(111)
			plt.step(bins_age1, np.append(ncum_age1[0], ncum_age1), color = 'k', alpha = 0.5)
			plt.plot(bins_fit_age1, ncum_fit_age1, 'g^')
			curve_fit3(bins_fit_age1, ncum_fit_age1, range_mass, complimits[0], 2, 1, np.sqrt(ncum_fit_age1))
			p9, = plt.plot([complimits[0], complimits[0]], [1E-10, 1E10], 'k--')
			plt.legend(loc = 'upper right', title = galnameout + age1_label)
			plt.xscale('log', nonposx = 'clip')
			plt.yscale('log', nonposy = 'clip')
			plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(ncum_fit), np.power(10, 1.3) * np.nanmax(ncum_age1)])
			ax1.xaxis.set_major_formatter(log10_labels_format)
			ax1.yaxis.set_major_formatter(log10_labels_format)
			plt.xlabel(r'log (M/M$_\odot$)')
			plt.ylabel(r'log N (> M)')
			plt.savefig('./Figures' + typename + '/' + galname + '_P3B_A1_' + typename + '_CHistogram_Ex.png')
			plt.close()

		if len(ncum_fit_age2) > 0:

			### --==--==--==-- ###
			print('>>>')
			print('>>> ' + galname + '_P3B_A2_' + typename + '_CHistogram')
			print('>>>')
			fig = plt.figure(figsize = (12, 12))
			ax1 = fig.add_subplot(111)
			plt.step(bins_age2, np.append(ncum_age2[0], ncum_age2), color = 'k', alpha = 0.5)
			plt.plot(bins_fit_age2, ncum_fit_age2, 'g^')
			curve_fit3(bins_fit_age2, ncum_fit_age2, range_mass, complimits[1], 2, 1, np.sqrt(ncum_fit_age2))
			p9, = plt.plot([complimits[1], complimits[1]], [1E-10, 1E10], 'k--')
			plt.legend(loc = 'upper right', title = galnameout + age2_label)
			plt.xscale('log', nonposx = 'clip')
			plt.yscale('log', nonposy = 'clip')
			plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(ncum_fit), np.power(10, 1.3) * np.nanmax(ncum_age2)])
			ax1.xaxis.set_major_formatter(log10_labels_format)
			ax1.yaxis.set_major_formatter(log10_labels_format)
			plt.xlabel(r'log (M/M$_\odot$)')
			plt.ylabel(r'log N (> M)')
			plt.savefig('./Figures' + typename + '/' + galname + '_P3B_A2_' + typename + '_CHistogram_Ex.png')
			plt.close()

		if len(ncum_fit_age3) > 0 and galname not in ['NGC1566', 'NGC4656']:

			### --==--==--==-- ###
			print('>>>')
			print('>>> ' + galname + '_P3B_A3_' + typename + '_CHistogram')
			print('>>>')
			fig = plt.figure(figsize = (12, 12))
			ax1 = fig.add_subplot(111)
			plt.step(bins_age3, np.append(ncum_age3[0], ncum_age3), color = 'k', alpha = 0.5)
			plt.plot(bins_fit_age3, ncum_fit_age3, 'g^')
			curve_fit3(bins_fit_age3, ncum_fit_age3, range_mass, complimits[2], 2, 1, np.sqrt(ncum_fit_age3))
			p9, = plt.plot([complimits[2], complimits[2]], [1E-10, 1E10], 'k--')
			plt.legend(loc = 'upper right', title = galnameout + age3_label)
			plt.xscale('log', nonposx = 'clip')
			plt.yscale('log', nonposy = 'clip')
			plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(ncum_fit), np.power(10, 1.3) * np.nanmax(ncum_age3)])
			ax1.xaxis.set_major_formatter(log10_labels_format)
			ax1.yaxis.set_major_formatter(log10_labels_format)
			plt.xlabel(r'log (M/M$_\odot$)')
			plt.ylabel(r'log N (> M)')
			plt.savefig('./Figures' + typename + '/' + galname + '_P3B_A3_' + typename + '_CHistogram_Ex.png')
			plt.close()

	# Set of plots with full cumulative functions
	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_P4A1_' + typename + '_CDF_Fit')
	print('>>>')
	fig = plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111)
	xdata_lim_float, ydata_lim_float = makecdffun(gal_array[:,4], np.nanmax(complimits))
	# Run IDL Code
	if flag_idl == 1 and typeflag < 2:
		prevdir = os.getcwd()
		from idlpy import IDL
		print('P4A2 - IDL (No Error)')
		fit_pl = IDL.mspecfit(np.array(xdata_lim_float, dtype = np.float32), 1E-6*np.ones(len(xdata_lim_float), dtype = np.float32), notrunc = 'notrunc')
		fit_pl_out = [fit_pl[1], fit_pl[2] + 1]
		error_pl_out = [0, 0]
		print('Fit PL:', fit_pl)
		fit = IDL.mspecfit(np.array(xdata_lim_float, dtype = np.float32), 1E-6*np.ones(len(xdata_lim_float), dtype = np.float32))
		fit_out = [fit[0], fit[1], fit[2] + 1]
		error_out = [0, 0, 0]
		print('Fit:', fit)
		os.chdir(prevdir)
	if flag_idl == 2 and typeflag < 2:	
		prevdir = os.getcwd()
		from idlpy import IDL
		print('P4A1 - IDL')
		fit_pl = IDL.mspecfit(np.array(xdata_lim_float, dtype = np.float32), 1E-6*np.ones(len(xdata_lim_float), dtype = np.float32), notrunc = 'notrunc', bootiter = 100)
		fit_pl_out = [fit_pl[1], fit_pl[2] + 1]
		error_pl_out = [fit_pl[4], fit_pl[5]]
		print('Fit PL:', fit_pl)
		# if galname in ['NGC4449', 'PowerLaw_ContLog', 'PowerLaw_ContLin']:
		if galname in ['Test'] or (galname == 'NGC3256_cut' and typeflag == 1):
			fit = IDL.mspecfit(np.array(xdata_lim_float, dtype = np.float32), 1E-6*np.ones(len(xdata_lim_float), dtype = np.float32))
			fit_out = [fit[0], fit[1], fit[2] + 1]
			error_out = [0, 0, 0]
		else:
			fit = IDL.mspecfit(np.array(xdata_lim_float, dtype = np.float32), 1E-6*np.ones(len(xdata_lim_float), dtype = np.float32), bootiter = 100)
			fit_out = [fit[0], fit[1], fit[2] + 1]
			error_out = [fit[3], fit[4], fit[5]]
		print('Fit:', fit)
		os.chdir(prevdir)
	if typeflag < 2:
		plt.legend(loc = 'upper right', title = galnameout)
	else:
		plt.legend(loc = 'upper right', title = galnameout + r' ($\tau \leq$ 400 Myr)')
	p9, = plt.plot([np.nanmax(complimits), np.nanmax(complimits)], [1E-10, 1E10], 'k--')
	plt.xscale('log', nonposx = 'clip')
	plt.yscale('log', nonposy = 'clip')
	plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(ncum_fit), np.power(10, 1.3) * np.nanmax(ncum)])
	ax1.xaxis.set_major_formatter(log10_labels_format)
	ax1.yaxis.set_major_formatter(log10_labels_format)
	if typeflag == 0.5:
		plt.xlabel(r'log Luminosity [$10^{32}$ erg/s]')
		plt.ylabel(r'log N (> L)')
	else:
		plt.xlabel(r'log (M/M$_\odot$)')
		plt.ylabel(r'log N (> M)')
	if flag_idl == 1 and typeflag < 2:
		if flag_XB == 1:
			plt.plot(mass_bins_log_plot, simplepowerlaw(mass_bins_log_plot, *fit_pl_out), 'k--', label = r'PL ($\alpha$ = {:.2f})'.format(fit_pl_out[1] - 1))
			plt.plot(mass_bins_log_plot, truncatedpowerlaw(mass_bins_log_plot, *fit_out), 'k:', label = r'TPL (N$_o$ = {:.2f}, L$_o$ = {:.2f})'.format(fit_out[0], np.log10(fit_out[1])))
		else:
			plt.plot(mass_bins_log_plot, simplepowerlaw(mass_bins_log_plot, *fit_pl_out), 'k--', label = r'PL ($\beta$ = {:.2f})'.format(fit_pl_out[1] - 1))
			plt.plot(mass_bins_log_plot, truncatedpowerlaw(mass_bins_log_plot, *fit_out), 'k:', label = r'TPL (N$_o$ = {:.2f}, M$_o$ = {:.2f})'.format(fit_out[0], np.log10(fit_out[1])))
		if typeflag < 2:
			plt.legend(loc = 'upper right', title = galnameout)
		else:
			plt.legend(loc = 'upper right', title = galnameout + r' ($\tau \leq$ 400 Myr)')
	if flag_idl == 2 and typeflag < 2:
		if flag_XB == 1:
			plt.plot(mass_bins_log_plot, simplepowerlaw(mass_bins_log_plot, *fit_pl_out), 'k--', label = r'PL ($\alpha$ = {:.2f} $\pm$ {:.2f})'.format(fit_pl_out[1] - 1, error_pl_out[1]))
			plt.plot(mass_bins_log_plot, truncatedpowerlaw(mass_bins_log_plot, *fit_out), 'k:', label = r'TPL (N$_o$ = {:.2f} $\pm$ {:.2f}, L$_o$ = {:.2f} $\pm$ {:.2f})'.format(fit_out[0], error_out[0], np.log10(fit_out[1]), 0.434*error_out[1]/fit_out[1]))
		else:
			plt.plot(mass_bins_log_plot, simplepowerlaw(mass_bins_log_plot, *fit_pl_out), 'k--', label = r'PL ($\beta$ = {:.2f} $\pm$ {:.2f})'.format(fit_pl_out[1] - 1, error_pl_out[1]))
			plt.plot(mass_bins_log_plot, truncatedpowerlaw(mass_bins_log_plot, *fit_out), 'k:', label = r'TPL (N$_o$ = {:.2f} $\pm$ {:.2f}, M$_o$ = {:.2f} $\pm$ {:.2f})'.format(fit_out[0], error_out[0], np.log10(fit_out[1]), 0.434*error_out[1]/fit_out[1]))
		if typeflag < 2:
			plt.legend(loc = 'upper right', title = galnameout)
		else:
			plt.legend(loc = 'upper right', title = galnameout + r' ($\tau \leq$ 400 Myr)')	
	plt.savefig('./Figures' + typename + '/' + galname + '_P4A_' + typename + '_CDF_Fit_Ex.png')
	plt.close()

	# Start likelihood analysis (GMC)
	if flag_likelihood == 1 and typeflag == 1:
		print('Start - Likelihood Analysis (GMC)')
		### --==--==--==-- ###
		fig, axes = plt.subplots(1, 1, sharex = 'all', sharey = 'all', figsize = (12, 12))
		Out_2sigma = likelihoodplot(gal_array_masslimit, galname, 'GMC', axes, 0, 0, 0, 'Test')
		fig.text(0.5, 0.025, r'log (M$_*$/M$_\odot$)', ha = 'center', fontsize = 30)
		fig.text(0.05, 0.5, r'$-\beta$', va = 'center', rotation = 'vertical', fontsize = 30)
		plt.axis([4.5, 9.0] + [0, 3])
		plt.xticks([5.0, 6.0, 7.0, 8.0])
		plt.yticks([0.0, 1.0, 2.0])
		plt.savefig('./Figures' + typename + '/' + galname + '_P4L_' + typename + '_Likelihood.png')
		plt.close()

	# Start likelihood analysis (XB)
	if flag_likelihood == 1 and typeflag == 0.5:
		print('Start - Likelihood Analysis (XB)')		
		### --==--==--==-- ###
		fig, axes = plt.subplots(1, 1, sharex = 'all', sharey = 'all', figsize = (12, 12))
		Out_2sigma = likelihoodplot(gal_array_masslimit, galname, 'XB', axes, [5.00, 2.75], 0, 0, 'Test')
		fig.text(0.5, 0.025, r'log (L/$10^{32}$ erg/s)', ha = 'center', fontsize = 30)
		fig.text(0.05, 0.5, r'$\alpha$', va = 'center', rotation = 'vertical', fontsize = 30)
		plt.axis([4.5, 8.0] + [0, 3])
		plt.xticks([5.0, 6.0, 7.0, 8.0])
		plt.yticks([0.0, 1.0, 2.0])
		plt.savefig('./Figures' + typename + '/' + galname + '_P4L_' + typename + '_Likelihood.png')
		plt.close()

	# Start likelihood analysis (C)
	if flag_likelihood == 1 and typeflag == 0:
		print('Start - Likelihood Analysis (C)')
		### --==--==--==-- ###
		fig, axes = plt.subplots(1, 1, sharex = 'all', sharey = 'all', figsize = (12, 12))
		Out_2sigma = likelihoodplot(gal_array_masslimit, galname, 'C', axes, 0, 0, 0, 'Test')
		fig.text(0.5, 0.025, r'log (M$_*$/M$_\odot$)', ha = 'center', fontsize = 30)
		fig.text(0.05, 0.5, r'$-\beta$', va = 'center', rotation = 'vertical', fontsize = 30)
		plt.axis([0.0, 6.0] + [0, 3])
		plt.xticks([2.0, 4.0])
		plt.yticks([0.0, 1.0, 2.0])
		plt.savefig('./Figures' + typename + '/' + galname + '_P4L_' + typename + '_Likelihood.png')
		plt.close()

	###
	# Set of plots in individual age ranges
	###

	if typeflag > 1:

		def makeplotforagebin(gal_array, gal_array_masslimit, complimits_val, flag_idl, excludedmspeclist, agetag, agelabel, galname, typename, ncum_fit):

			gal_array4 = gal_array[:,4]
			gal_array4_masslimit = gal_array_masslimit[:,4]

			### --==--==--==-- ###
			print('>>>')
			print('>>> ' + galname + '_P4B' + agetag + '_' + typename + '_CDF_Fit')
			print('>>> ')

			if len(gal_array4) > 0:

				if flag_idl == 1 and len(gal_array4) > 9:
			
					prevdir = os.getcwd()
					from idlpy import IDL
					print('IDL')
					fit_pl = IDL.mspecfit(np.array(gal_array4_masslimit, dtype = np.float32), 1E-6*np.ones(len(gal_array4_masslimit), dtype = np.float32), notrunc = 'notrunc', bootiter = 100)
					fit_pl_out = [fit_pl[1], fit_pl[2] + 1]
					error_pl_out = [fit_pl[4], fit_pl[5]]
					print('Fit PL:', fit_pl)
					if galname in excludedmspeclist:
						fit = IDL.mspecfit(np.array(gal_array4_masslimit, dtype = np.float32), 1E-6*np.ones(len(gal_array4_masslimit), dtype = np.float32))
						fit_out = [fit[0], fit[1], fit[2] + 1]
						error_out = [0, 0, 0]
					else:
						fit = IDL.mspecfit(np.array(gal_array4_masslimit, dtype = np.float32), 1E-6*np.ones(len(gal_array4_masslimit), dtype = np.float32), bootiter = 100)
						fit_out = [fit[0], fit[1], fit[2] + 1]
						error_out = [fit[3], fit[4], fit[5]]
					print('Fit:', fit)
					os.chdir(prevdir)

				# typeflag < 3 and
				if flag_likelihood == 1:
					print('Start - Likelihood Analysis')
					# res_loglikesimplepowerlaw, res_loglikeschechter = likelihoodtesting(gal_array4_masslimit, typename, galname, agetag, 0, 0, 0)
					
					### --==--==--==-- ###
					fig, axes = plt.subplots(1, 1, sharex = 'all', sharey = 'all', figsize = (12, 12))
					Out_2sigma = likelihoodplot(gal_array_masslimit, galname, agetag, axes, 0, 0, 0, 'Test')
					fig.text(0.5, 0.025, r'log (M$_*$/M$_\odot$)', ha = 'center', fontsize = 30)
					fig.text(0.05, 0.5, r'$-\beta$', va = 'center', rotation = 'vertical', fontsize = 30)
					plt.axis([3.5, 7.5] + [0, 3])
					plt.xticks([4.0, 5.0, 6.0, 7.0])
					plt.yticks([0.0, 1.0, 2.0])
					plt.savefig('./Figures' + typename + '/' + galname + '_P4L' + agetag + '_' + typename + '_Likelihood.png')
					plt.close()

				# Switch: False - Personal fitting routine (Just use IDL technique)
				if False:

					### --==--==--==-- ###
					fig = plt.figure(figsize = (12, 12))
					ax1 = fig.add_subplot(111)
					xdata_lim_float, ydata_lim_float = makecdffun(gal_array4, complimits_val)
					curve_fit3(xdata_lim_float, ydata_lim_float, range_mass, complimits_val, 2, 1, np.sqrt(ydata_lim_float))
					plt.legend(loc = 'upper right', title = galnameout + agelabel + '\n Number = {:.0f}'.format(len(ydata_lim_float)))
					p9, = plt.plot([complimits_val, complimits_val], [1E-10, 1E10], 'k--')
					plt.xscale('log', nonposx = 'clip')
					plt.yscale('log', nonposy = 'clip')
					plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(ncum_fit), np.power(10, 1.3) * np.nanmax(ncum_fit)])
					ax1.xaxis.set_major_formatter(log10_labels_format)
					ax1.yaxis.set_major_formatter(log10_labels_format)
					plt.xlabel(r'log (M/M$_\odot$)')
					plt.ylabel(r'log N (> M)')
					plt.savefig('./Figures' + typename + '/' + galname + '_P4B' + agetag + '_' + typename + '_CDF_Fit_Ex.png')
					plt.close()

				### --==--==--==-- ###
				fig = plt.figure(figsize = (12, 12))
				ax1 = fig.add_subplot(111)
				xdata_lim_float, ydata_lim_float = makecdffun(gal_array4, complimits_val)
				p9, = plt.plot([complimits_val, complimits_val], [1E-10, 1E10], 'k--')
				plt.xscale('log', nonposx = 'clip')
				plt.yscale('log', nonposy = 'clip')
				plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(ncum_fit), np.power(10, 1.3) * np.nanmax(ncum_fit)])
				ax1.xaxis.set_major_formatter(log10_labels_format)
				ax1.yaxis.set_major_formatter(log10_labels_format)
				plt.xlabel(r'log (M/M$_\odot$)')
				plt.ylabel(r'log N (> M)')
				if flag_idl == 1 and len(gal_array4) > 10:
					plt.plot(mass_bins_log_plot, simplepowerlaw(mass_bins_log_plot, *fit_pl_out), 'k--', label = r'PL ($\beta$ = {:.2f} $\pm$ {:.2f})'.format(fit_pl_out[1] - 1, error_pl_out[1]))
					plt.plot(mass_bins_log_plot, truncatedpowerlaw(mass_bins_log_plot, *fit_out), 'k:', label = r'TPL (N$_o$ = {:.2f} $\pm$ {:.2f}, M$_o$ = {:.2f} $\pm$ {:.2f})'.format(fit_out[0], error_out[0], np.log10(fit_out[1]), 0.434*error_out[1]/fit_out[1]))
					plt.legend(loc = 'upper right', title = galnameout + agelabel + '\n Number = {:.0f}'.format(len(ydata_lim_float)))
					plt.savefig('./Figures' + typename + '/' + galname + '_P4B' + agetag + '_' + typename + '_CDF_Fit_Ez.png')
				plt.close()

			return 0

		# A1
		makeplotforagebin(gal_array_age1, gal_array_age1_masslimit, complimits[0], flag_idl, ['Test', 'NGC4449', 'NGC5238'], '_A1', age1_label, galname, typename, ncum_fit)
		# A2
		makeplotforagebin(gal_array_age2, gal_array_age2_masslimit, complimits[1], flag_idl, ['Test', 'NGC4214', 'NGC4395', 'NGC5238', 'NGC5477'], '_A2', age2_label, galname, typename, ncum_fit)
		# A3, if galname not in []:,
		makeplotforagebin(gal_array_age3, gal_array_age3_masslimit, complimits[2], flag_idl, ['Test', 'NGC1566', 'NGC3738', 'NGC3351', 'NGC1433', 'NGC1705', 'NGC3738', 'NGC4395'], '_A3', age3_label, galname, typename, ncum_fit)

		# makeplotforagebin(gal_array_test1, gal_array_test1_masslimit, complimits[0], flag_idl, ['Test', 'NGC4214', 'NGC4395', 'NGC5238', 'NGC5477'], '_T1', test1_label, galname, typename, ncum_fit)
		# makeplotforagebin(gal_array_test2, gal_array_test2_masslimit, complimits[0], flag_idl, ['Test', 'NGC4214', 'NGC4395', 'NGC5238', 'NGC5477'], '_T2', test2_label, galname, typename, ncum_fit)

		# A3 - Run removal test
		if galname in ['M83']:
			gal_array_age3_sorted = np.sort(gal_array_age3[:,4])
			gal_array_age3_masslimit_sorted = np.sort(gal_array_age3_masslimit[:,4])
			makeplotforagebin(gal_array_age3_sorted[:-1], gal_array_age3_masslimit_sorted[:-1], complimits[2], flag_idl, ['Test'], '_A3_1R', age3_label, galname, typename, ncum_fit)
			makeplotforagebin(gal_array_age3_sorted[:-2], gal_array_age3_masslimit_sorted[:-2], complimits[2], flag_idl, ['Test'], '_A3_2R', age3_label, galname, typename, ncum_fit)
		# A5
		makeplotforagebin(gal_array_age5, gal_array_age5_masslimit, age5_masslimit, flag_idl, ['Test', 'NGC5238'], '_A5', age5_label, galname, typename, ncum_fit)
		
		# Switch: Don't run other age bins
		if False:

			# A6
			if galname not in ['NGC4395']:
				makeplotforagebin(gal_array_age6, gal_array_age6_masslimit, age6_masslimit, flag_idl, ['Test', 'Dwarf2_Low', 'NGC1433', 'NGC1566', 'NGC1705', 'NGC3344', 'NGC3351', 'NGC3738', 'NGC4242'], '_A6', age6_label, galname, typename, ncum_fit)
			else:
				makeplotforagebin(gal_array_age6, gal_array_age6_masslimit, age6_masslimit, 0, ['Test'], '_A6', age6_label, galname, typename, ncum_fit)
			# A7
			makeplotforagebin(gal_array_age7, gal_array_age7_masslimit, age7_masslimit, flag_idl, ['Test', 'NGC5238'], '_A7', age7_label, galname, typename, ncum_fit)
			# A8
			makeplotforagebin(gal_array_age8, gal_array_age8_masslimit, age8_masslimit, flag_idl, ['Test', 'SMC', 'NGC0045', 'NGC1566', 'NGC1705', 'NGC3351', 'NGC4242', 'NGC4395', 'NGC5238', 'UGC1249', 'Dwarf2_NGC4656', 'Dwarf2_NGC4449', 'Dwarf2_NGC5253', 'Dwarf2_NGC3738', 'Dwarf2_UGC1249'], '_A8', age8_label, galname, typename, ncum_fit)
			# A9
			if galname not in ['Antennae', 'NGC7793']:
				makeplotforagebin(gal_array_age9, gal_array_age9_masslimit, age9_masslimit, flag_idl, ['Test', 'NGC1433', 'NGC5238', 'NGC5477'], '_A9', age9_label, galname, typename, ncum_fit)
			else:
				makeplotforagebin(gal_array_age9, gal_array_age9_masslimit, age9_masslimit, 0, ['Test'], '_A9', age9_label, galname, typename, ncum_fit)

	if len(nequalcum) > 0:
		
		# Plot Equal Number Histogram
		### --==--==--==-- ###
		print('>>>')
		print('>>> ' + galname + '_P5A1_' + typename + '_Histogram_Equal')
		print('>>>')
		fig = plt.figure(figsize = (12, 12))
		ax1 = fig.add_subplot(111)
		plt.step(binsequal, np.append(nequalcum[0], nequalcum), color = 'k', alpha = 0.5)
		curve_fit3(binsequal_fit, nequalcum_fit, range_mass, np.nanmax(complimits), 1, 1, np.sqrt(nequalcum_fit))
		plt.errorbar(binsequal_fit, nequalcum_fit, marker = '^', linestyle = 'None', markerfacecolor = 'g', yerr = np.sqrt(nequalcum_fit))
		p2, = plt.plot([np.nanmax(complimits), np.nanmax(complimits)], [1E-10, 1E10], 'k--')
		if typeflag > 1:
			plt.legend(loc = 'upper right', title = galnameout + r' ($\tau \leq$ 400 Myr)' + '\n Number per bin = {:.0f}'.format(np.average(nequal)))
		else:
			plt.legend(loc = 'upper right', title = galnameout + '\n Number per bin = {:.0f}'.format(np.average(nequal)))
		plt.xscale('log', nonposx = 'clip')
		plt.yscale('log', nonposy = 'clip')
		plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(nequalcum_fit), np.power(10, 1.3) * np.nanmax(nequalcum)])
		ax1.xaxis.set_major_formatter(log10_labels_format)
		ax1.yaxis.set_major_formatter(log10_labels_format)
		if typeflag == 0.5:
			plt.xlabel(r'log Luminosity [$10^{32}$ erg/s]')
			plt.ylabel(r'log N (> L)')
		else:
			plt.xlabel(r'log (M/M$_\odot$)')
			plt.ylabel(r'log N (> M)')
		plt.savefig('./Figures' + typename + '/' + galname + '_P5A1_' + typename + '_Histogram_Equal_Ex.png')
		plt.close()

		### --==--==--==-- ###
		print('>>>')
		print('>>> ' + galname + '_P5A2_' + typename + '_Histogram_Equal_dM')
		print('>>>')
		fig = plt.figure(figsize = (12, 12))
		ax1 = fig.add_subplot(111)
		plt.step(binsequal, np.append(nequal_dM[0], nequal_dM), color = 'k', alpha = 0.5)
		plt.errorbar(binsequal_fit, nequal_fit_dM, marker = '^', linestyle = 'None', markerfacecolor = 'g', yerr = nequal_fit_dM_err)
		curve_fit3(binsequal_fit, nequal_fit_dM, range_mass, np.nanmax(complimits), 1, 1, nequal_fit_dM_err)
		p9, = plt.plot([np.nanmax(complimits), np.nanmax(complimits)], [1E-10, 1E10], 'k--')
		if typeflag > 1:
			plt.legend(loc = 'upper right', title = galnameout + r' ($\tau \leq$ 400 Myr)' + '\n Number per bin = {:.0f}'.format(np.average(nequal)))
		else:
			plt.legend(loc = 'upper right', title = galnameout + '\n Number per bin = {:.0f}'.format(np.average(nequal)))
		plt.xscale('log', nonposx = 'clip')
		plt.yscale('log', nonposy = 'clip')
		plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit_dM), np.power(10, 1.3) * np.nanmax(n_dM)])
		ax1.xaxis.set_major_formatter(log10_labels_format)
		ax1.yaxis.set_major_formatter(log10_labels_format)
		if typeflag == 0.5:
			plt.xlabel(r'log Luminosity [$10^{32}$ erg/s]')
			plt.ylabel(r'log (dN/dL)')
		else:
			plt.xlabel(r'log (M/M$_\odot$)')
			plt.ylabel(r'log (dN/dM)')
		plt.savefig('./Figures' + typename + '/' + galname + '_P5A2_' + typename + '_Histogram_Equal_dM_Ex.png')
		plt.close()

		### --==--==--==-- ###
		print('>>>')
		print('>>> ' + galname + '_P5A3_' + typename + '_Histogram_Equal_dlogM')
		print('>>>')
		fig = plt.figure(figsize = (12, 12))
		ax1 = fig.add_subplot(111)
		plt.step(binsequal, np.append(nequal_dlogM[0], nequal_dlogM), color = 'k', alpha = 0.5)
		plt.errorbar(binsequal_fit, nequal_fit_dlogM, marker = '^', linestyle = 'None', markerfacecolor = 'g', yerr = nequal_fit_dlogM_err)
		curve_fit3(binsequal_fit, nequal_fit_dlogM, range_mass, np.nanmax(complimits), 1, 1, nequal_fit_dlogM_err)
		p9, = plt.plot([np.nanmax(complimits), np.nanmax(complimits)], [1E-10, 1E10], 'k--')
		if typeflag > 1:
			plt.legend(loc = 'upper right', title = galnameout + r' ($\tau \leq$ 400 Myr)' + '\n Number per bin = {:.0f}'.format(np.average(nequal)))
		else:
			plt.legend(loc = 'upper right', title = galnameout + '\n Number per bin = {:.0f}'.format(np.average(nequal)))
		plt.xscale('log', nonposx = 'clip')
		plt.yscale('log', nonposy = 'clip')
		plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit_dlogM), np.power(10, 1.3) * np.nanmax(n_dlogM)])
		ax1.xaxis.set_major_formatter(log10_labels_format)
		ax1.yaxis.set_major_formatter(log10_labels_format)
		if typeflag == 0.5:
			plt.xlabel(r'log Luminosity [$10^{32}$ erg/s]')
			plt.ylabel(r'log (dN/dlogL)')
		else:
			plt.xlabel(r'log (M/M$_\odot$)')
			plt.ylabel(r'log (dN/dlogM)')
		plt.savefig('./Figures' + typename + '/' + galname + '_P5A3_' + typename + '_Histogram_Equal_dlogM_Ex.png')
		plt.close()

	# Flag: Only for SC and SimSC
	if typeflag > 1:

		if len(nequalcum_age1) > 0:

			### --==--==--==-- ###
			print('>>>')
			print('>>> ' + galname + '_P5B1_A1_' + typename + '_Histogram_Equal')
			print('>>>')
			fig = plt.figure(figsize = (12, 12))
			ax1 = fig.add_subplot(111)
			plt.step(binsequal_age1, np.append(nequalcum_age1[0], nequalcum_age1), color = 'k', alpha = 0.5)
			plt.errorbar(binsequal_fit_age1, nequalcum_fit_age1, marker = '^', linestyle = 'None', markerfacecolor = 'g', yerr = np.sqrt(nequalcum_fit_age1))
			curve_fit3(binsequal_fit_age1, nequalcum_fit_age1, range_mass, complimits[0], 1, 1, np.sqrt(nequalcum_fit_age1))
			p9, = plt.plot([complimits[0], complimits[0]], [1E-10, 1E10], 'k--')
			plt.legend(loc = 'upper right', title = galnameout + age1_label + '\n Number per bin = {}'.format(np.average(nequal_age1)))
			plt.xscale('log', nonposx = 'clip')
			plt.yscale('log', nonposy = 'clip')
			plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(nequalcum_fit), np.power(10, 1.3) * np.nanmax(nequalcum_age1)])
			ax1.xaxis.set_major_formatter(log10_labels_format)
			ax1.yaxis.set_major_formatter(log10_labels_format)
			plt.xlabel(r'log (M/M$_\odot$)')
			plt.ylabel(r'log N (> M)')
			plt.savefig('./Figures' + typename + '/' + galname + '_P5B1_A1_' + typename + '_Histogram_Equal_Ex.png')
			plt.close()

			### --==--==--==-- ###
			print('>>>')
			print('>>> ' + galname + '_P5B2_A1_' + typename + '_Histogram_Equal_dM')
			print('>>>')
			fig = plt.figure(figsize = (12, 12))
			ax1 = fig.add_subplot(111)
			plt.step(binsequal_age1, np.append(nequal_dM_age1[0], nequal_dM_age1), color = 'k', alpha = 0.5)
			plt.errorbar(binsequal_fit_age1, nequal_fit_dM_age1, marker = '^', linestyle = 'None', markerfacecolor = 'g', yerr = nequal_fit_dM_age1_err)
			curve_fit3(binsequal_fit_age1, nequal_fit_dM_age1, range_mass, complimits[0], 1, 1, nequal_fit_dM_age1_err)
			p9, = plt.plot([complimits[0], complimits[0]], [1E-10, 1E10], 'k--')
			plt.legend(loc = 'upper right', title = galnameout + age1_label + '\n Number per bin = {}'.format(np.average(nequal_age1)))
			plt.xscale('log', nonposx = 'clip')
			plt.yscale('log', nonposy = 'clip')
			plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit_dM), np.power(10, 1.3) * np.nanmax(n_dM_age1)])
			ax1.xaxis.set_major_formatter(log10_labels_format)
			ax1.yaxis.set_major_formatter(log10_labels_format)
			plt.xlabel(r'log (M/M$_\odot$)')
			plt.ylabel(r'log (dN/dM)')
			plt.savefig('./Figures' + typename + '/' + galname + '_P5B2_A1_' + typename + '_Histogram_Equal_dM_Ex.png')
			plt.close()

			### --==--==--==-- ###
			print('>>>')
			print('>>> ' + galname + '_P5B3_A1_' + typename + '_Histogram_Equal_dlogM')
			print('>>>')
			fig = plt.figure(figsize = (12, 12))
			ax1 = fig.add_subplot(111)
			plt.step(binsequal_age1, np.append(nequal_dlogM_age1[0], nequal_dlogM_age1), color = 'k', alpha = 0.5)
			plt.errorbar(binsequal_fit_age1, nequal_fit_dlogM_age1, marker = '^', linestyle = 'None', markerfacecolor = 'g', yerr = nequal_fit_dlogM_age1_err)
			curve_fit3(binsequal_fit_age1, nequal_fit_dlogM_age1, range_mass, complimits[0], 1, 1, nequal_fit_dlogM_age1_err)
			p9, = plt.plot([complimits[0], complimits[0]], [1E-10, 1E10], 'k--')
			plt.legend(loc = 'upper right', title = galnameout + age1_label + '\n Number per bin = {}'.format(np.average(nequal_age1)))
			plt.xscale('log', nonposx = 'clip')
			plt.yscale('log', nonposy = 'clip')
			plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit_dlogM), np.power(10, 1.3) * np.nanmax(n_dlogM_age1)])
			ax1.xaxis.set_major_formatter(log10_labels_format)
			ax1.yaxis.set_major_formatter(log10_labels_format)
			plt.xlabel(r'log (M/M$_\odot$)')
			plt.ylabel(r'log (dN/dlogM)')
			plt.savefig('./Figures' + typename + '/' + galname + '_P5B3_A1_' + typename + '_Histogram_Equal_dlogM_Ex.png')
			plt.close()

		if len(nequalcum_age2) > 0:

			### --==--==--==-- ###
			print('>>>')
			print('>>> ' + galname + '_P5B1_A2_' + typename + '_Histogram_Equal')
			print('>>>')
			fig = plt.figure(figsize = (12, 12))
			ax1 = fig.add_subplot(111)
			plt.step(binsequal_age2, np.append(nequalcum_age2[0], nequalcum_age2), color = 'k', alpha = 0.5)
			plt.errorbar(binsequal_fit_age2, nequalcum_fit_age2, marker = '^', linestyle = 'None', markerfacecolor = 'g', yerr = np.sqrt(nequalcum_fit_age2))
			curve_fit3(binsequal_fit_age2, nequalcum_fit_age2, range_mass, complimits[1], 1, 1, np.sqrt(nequalcum_fit_age2))
			p9, = plt.plot([complimits[1], complimits[1]], [1E-10, 1E10], 'k--')
			plt.legend(loc = 'upper right', title = galnameout + r' (10 Myr < $\tau \leq$ 100 Myr)' + '\n Number per bin = {}'.format(np.average(nequal_age2)))
			plt.xscale('log', nonposx = 'clip')
			plt.yscale('log', nonposy = 'clip')
			plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(nequalcum_fit), np.power(10, 1.3) * np.nanmax(nequalcum_age2)])
			ax1.xaxis.set_major_formatter(log10_labels_format)
			ax1.yaxis.set_major_formatter(log10_labels_format)
			plt.xlabel(r'log (M/M$_\odot$)')
			plt.ylabel(r'log N (> M)')
			plt.savefig('./Figures' + typename + '/' + galname + '_P5B1_A2_' + typename + '_Histogram_Equal_Ex.png')
			plt.close()

			### --==--==--==-- ###
			print('>>>')
			print('>>> ' + galname + '_P5B2_A2_' + typename + '_Histogram_Equal_dM')
			print('>>>')
			fig = plt.figure(figsize = (12, 12))
			ax1 = fig.add_subplot(111)
			plt.step(binsequal_age2, np.append(nequal_dM_age2[0], nequal_dM_age2), color = 'k', alpha = 0.5)
			plt.errorbar(binsequal_fit_age2, nequal_fit_dM_age2, marker = '^', linestyle = 'None', markerfacecolor = 'g', yerr = nequal_fit_dM_age2_err)
			curve_fit3(binsequal_fit_age2, nequal_fit_dM_age2, range_mass, complimits[1], 1, 1, nequal_fit_dM_age2_err)
			p9, = plt.plot([complimits[1], complimits[1]], [1E-10, 1E10], 'k--')
			plt.legend(loc = 'upper right', title = galnameout + r' (10 Myr < $\tau \leq$ 100 Myr)' + '\n Number per bin = {}'.format(np.average(nequal_age2)))
			plt.xscale('log', nonposx = 'clip')
			plt.yscale('log', nonposy = 'clip')
			plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit_dM), np.power(10, 1.3) * np.nanmax(n_dM_age2)])
			ax1.xaxis.set_major_formatter(log10_labels_format)
			ax1.yaxis.set_major_formatter(log10_labels_format)
			plt.xlabel(r'log (M/M$_\odot$)')
			plt.ylabel(r'log (dN/dM)')
			plt.savefig('./Figures' + typename + '/' + galname + '_P5B2_A2_' + typename + '_Histogram_Equal_dM_Ex.png')
			plt.close()

			### --==--==--==-- ###
			print('>>>')
			print('>>> ' + galname + '_P5B3_A2_' + typename + '_Histogram_Equal_dlogM')
			print('>>>')
			fig = plt.figure(figsize = (12, 12))
			ax1 = fig.add_subplot(111)
			plt.step(binsequal_age2, np.append(nequal_dlogM_age2[0], nequal_dlogM_age2), color = 'k', alpha = 0.5)
			plt.errorbar(binsequal_fit_age2, nequal_fit_dlogM_age2, marker = '^', linestyle = 'None', markerfacecolor = 'g', yerr = nequal_fit_dlogM_age2_err)
			curve_fit3(binsequal_fit_age2, nequal_fit_dlogM_age2, range_mass, complimits[1], 1, 1, nequal_fit_dlogM_age2_err)
			p9, = plt.plot([complimits[1], complimits[1]], [1E-10, 1E10], 'k--')
			plt.legend(loc = 'upper right', title = galnameout + r' (10 Myr < $\tau \leq$ 100 Myr)' + '\n Number per bin = {}'.format(np.average(nequal_age2)))
			plt.xscale('log', nonposx = 'clip')
			plt.yscale('log', nonposy = 'clip')
			plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit_dlogM), np.power(10, 1.3) * np.nanmax(n_dlogM_age2)])
			ax1.xaxis.set_major_formatter(log10_labels_format)
			ax1.yaxis.set_major_formatter(log10_labels_format)
			plt.xlabel(r'log (M/M$_\odot$)')
			plt.ylabel(r'log (dN/dlogM)')
			plt.savefig('./Figures' + typename + '/' + galname + '_P5B3_A2_' + typename + '_Histogram_Equal_dlogM_Ex.png')
			plt.close()

		if len(nequalcum_age3) > 0:

			### --==--==--==-- ###
			print('>>>')
			print('>>> ' + galname + '_P5B1_A3_' + typename + '_Histogram_Equal')
			print('>>>')
			fig = plt.figure(figsize = (12, 12))
			ax1 = fig.add_subplot(111)
			plt.step(binsequal_age3, np.append(nequalcum_age3[0], nequalcum_age3), color = 'k', alpha = 0.5)
			plt.errorbar(binsequal_fit_age3, nequalcum_fit_age3, marker = '^', linestyle = 'None', markerfacecolor = 'g', yerr = np.sqrt(nequalcum_fit_age3))
			curve_fit3(binsequal_fit_age3, nequalcum_fit_age3, range_mass, complimits[2], 2, 1, np.sqrt(nequalcum_fit_age3))
			p9, = plt.plot([complimits[2], complimits[2]], [1E-10, 1E10], 'k--')
			plt.legend(loc = 'upper right', title = galnameout + r' (100 Myr < $\tau \leq$ 400 Myr)' + '\n Number per bin = {}'.format(np.average(nequal_age3)))
			plt.xscale('log', nonposx = 'clip')
			plt.yscale('log', nonposy = 'clip')
			plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(nequalcum_fit), np.power(10, 1.3) * np.nanmax(nequalcum_age3)])
			ax1.xaxis.set_major_formatter(log10_labels_format)
			ax1.yaxis.set_major_formatter(log10_labels_format)
			plt.xlabel(r'log (M/M$_\odot$)')
			plt.ylabel(r'log N (> M)')
			plt.savefig('./Figures' + typename + '/' + galname + '_P5B1_A3_' + typename + '_Histogram_Equal_Ex.png')
			plt.close()

			### --==--==--==-- ###
			print('>>>')
			print('>>> ' + galname + '_P5B2_A3_' + typename + '_Histogram_Equal_dM')
			print('>>>')
			fig = plt.figure(figsize = (12, 12))
			ax1 = fig.add_subplot(111)
			plt.step(binsequal_age3, np.append(nequal_dM_age3[0], nequal_dM_age3), color = 'k', alpha = 0.5)
			plt.errorbar(binsequal_fit_age3, nequal_fit_dM_age3, marker = '^', linestyle = 'None', markerfacecolor = 'g', yerr = nequal_fit_dM_age3_err)
			curve_fit3(binsequal_fit_age3, nequal_fit_dM_age3, range_mass, complimits[2], 1, 1, nequal_fit_dM_age3_err)
			p9, = plt.plot([complimits[2], complimits[2]], [1E-10, 1E10], 'k--')
			plt.legend(loc = 'upper right', title = galnameout + r' (100 Myr < $\tau \leq$ 400 Myr)' + '\n Number per bin = {}'.format(np.average(nequal_age3)))
			plt.xscale('log', nonposx = 'clip')
			plt.yscale('log', nonposy = 'clip')
			plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit_dM), np.power(10, 1.3) * np.nanmax(n_dM_age3)])
			ax1.xaxis.set_major_formatter(log10_labels_format)
			ax1.yaxis.set_major_formatter(log10_labels_format)
			plt.xlabel(r'log (M/M$_\odot$)')
			plt.ylabel(r'log (dN/dM)')
			plt.savefig('./Figures' + typename + '/' + galname + '_P5B2_A3_' + typename + '_Histogram_Equal_dM_Ex.png')
			plt.close()

			### --==--==--==-- ###
			print('>>>')
			print('>>> ' + galname + '_P5B3_A3_' + typename + '_Histogram_Equal_dlogM')
			print('>>>')
			fig = plt.figure(figsize = (12, 12))
			ax1 = fig.add_subplot(111)
			plt.step(binsequal_age3, np.append(nequal_dlogM_age3[0], nequal_dlogM_age3), color = 'k', alpha = 0.5)
			plt.errorbar(binsequal_fit_age3, nequal_fit_dlogM_age3, marker = '^', linestyle = 'None', markerfacecolor = 'g', yerr = nequal_fit_dlogM_age3_err)
			curve_fit3(binsequal_fit_age3, nequal_fit_dlogM_age3, range_mass, complimits[2], 1, 1, nequal_fit_dlogM_age3_err)
			p9, = plt.plot([complimits[2], complimits[2]], [1E-10, 1E10], 'k--')
			plt.legend(loc = 'upper right', title = galnameout + r' (100 Myr < $\tau \leq$ 400 Myr)' + '\n Number per bin = {}'.format(np.average(nequal_age3)))
			plt.xscale('log', nonposx = 'clip')
			plt.yscale('log', nonposy = 'clip')
			plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit_dlogM), np.power(10, 1.3) * np.nanmax(n_dlogM_age3)])
			ax1.xaxis.set_major_formatter(log10_labels_format)
			ax1.yaxis.set_major_formatter(log10_labels_format)
			plt.xlabel(r'log (M/M$_\odot$)')
			plt.ylabel(r'log (dN/dlogM)')
			plt.savefig('./Figures' + typename + '/' + galname + '_P5B3_A3_' + typename + '_Histogram_Equal_dlogM_Ex.png')
			plt.close()

		### --==--==--==-- ###
		print('>>>')
		print('>>> ' + galname + '_P6A_' + typename + '_AgeMass')
		print('>>>')
		fig = plt.figure(figsize = (12, 12))
		ax1 = fig.add_subplot(111)
		plt.plot(gal_array_masslimit[:,6], gal_array_masslimit[:,4], 'ro', markersize = 4, alpha = 0.4)
		p9, = plt.plot([np.power(10, 5.8), np.power(10, 7)], [complimits[0], complimits[0]], 'k--')
		p9, = plt.plot([np.power(10, 7), np.power(10, 8)], [complimits[1], complimits[1]], 'k--')
		p9, = plt.plot([np.power(10, 8), 4 * np.power(10, 8)], [complimits[2], complimits[2]], 'k--')
		p9, = plt.plot([np.power(10, 7), np.power(10, 7)], [complimits[0], 1.5 * np.nanmax(gal_array[:,4])], 'k--')
		p9, = plt.plot([np.power(10, 8), np.power(10, 8)], [complimits[1], 1.5 * np.nanmax(gal_array[:,4])], 'k--')
		p9, = plt.plot([4 * np.power(10, 8), 4 * np.power(10, 8)], [complimits[2], 1.5 * np.nanmax(gal_array[:,4])], 'k--')
		plt.xlabel(r'Age [yr]')
		plt.ylabel(r'M [M$_\odot$]')
		plt.xscale('log', nonposx = 'clip')
		plt.yscale('log', nonposy = 'clip')
		ax1.xaxis.set_major_formatter(log10_labels_format)
		ax1.yaxis.set_major_formatter(log10_labels_format)
		plt.xlabel(r'log (Age/yr)')
		plt.ylabel(r'log (M/M$_\odot$)')
		plt.savefig('./Figures' + typename + '/' + galname + '_P6A_' + typename + '_AgeMass_Ex.png')
		plt.close()

		### --==--==--==-- ###
		print('>>>')
		print('>>> ' + galname + '_P6B_' + typename + '_AgeMass')
		print('>>>')
		fig = plt.figure(figsize = (12, 12))
		ax1 = fig.add_subplot(111)
		plt.plot(gal_array[:,6], gal_array[:,4], 'ro', markersize = 4, alpha = 0.4)
		p9, = plt.plot([np.power(10, 5.8), np.power(10, 7)], [complimits[0], complimits[0]], 'k--')
		p9, = plt.plot([np.power(10, 7), np.power(10, 8)], [complimits[1], complimits[1]], 'k--')
		p9, = plt.plot([np.power(10, 8), 4 * np.power(10, 8)], [complimits[2], complimits[2]], 'k--')
		p9, = plt.plot([np.power(10, 7), np.power(10, 7)], [complimits[0], 1.5 * np.nanmax(gal_array[:,4])], 'k--')
		p9, = plt.plot([np.power(10, 8), np.power(10, 8)], [complimits[1], 1.5 * np.nanmax(gal_array[:,4])], 'k--')
		p9, = plt.plot([4 * np.power(10, 8), 4 * np.power(10, 8)], [complimits[2], 1.5 * np.nanmax(gal_array[:,4])], 'k--')
		plt.xlabel(r'Age [yr]')
		plt.ylabel(r'M [M$_\odot$]')
		plt.xscale('log', nonposx = 'clip')
		plt.yscale('log', nonposy = 'clip')
		ax1.xaxis.set_major_formatter(log10_labels_format)
		ax1.yaxis.set_major_formatter(log10_labels_format)
		plt.xlabel(r'log (Age/yr)')
		plt.ylabel(r'log (M/M$_\odot$)')
		plt.savefig('./Figures' + typename + '/' + galname + '_P6B_' + typename + '_AgeMass_Ex.png')
		plt.close()

	return np.asarray(binsequal_fit, dtype = float), np.asarray(nequal_fit_dM, dtype = float), np.asarray(nequal_fit_dlogM, dtype = float)

def masteranalysisfunctionSCSC(galname, gal_array_1, gal_array_2, range_mass, complimits_1, complimits_2):

	'''
	Function: Perform analysis using a combination of cluster mass distributions
	'''

	print('>>>')
	print('>>> Start - Analysis (SC1 + SC2)')
	print('>>>')

	typename = 'SC'
	galnameout = galnameoutfun(galname)

	# Create classic 3 age bins (1)
	print('Array #1')
	print('Input Completeness Limits: {:.1e}, {:.1e}, {:.1e}'.format(complimits_1[0], complimits_1[1], complimits_1[2]))
	print('1 = (0, 10] Myr + input completeness limit')
	gal_array_1_age1 = gal_array_1[gal_array_1[:,6] <= 10.01*1E6]
	gal_array_1_age1_masslimit = gal_array_1_age1[gal_array_1_age1[:,4] > complimits_1[0]]
	age1_label = r' ($\tau \leq$ 10 Myr)'
	print('2 = (10, 100] Myr + input completeness limit')
	gal_array_1_age2_tmp = gal_array_1[gal_array_1[:,6] > 10.01*1E6]
	gal_array_1_age2 = gal_array_1_age2_tmp[gal_array_1_age2_tmp[:,6] <= 100.01*1E6]
	gal_array_1_age2_masslimit = gal_array_1_age2[gal_array_1_age2[:,4] > complimits_1[1]]
	age2_label = r' (10 < $\tau \leq$ 100 Myr)'
	print('3 = (100, 400] Myr + input completeness limit')
	gal_array_1_age3_tmp = gal_array_1[gal_array_1[:,6] > 100.01*1E6]
	gal_array_1_age3 = gal_array_1_age3_tmp[gal_array_1_age3_tmp[:,6] <= 400.01*1E6]
	gal_array_1_age3_masslimit = gal_array_1_age3[gal_array_1_age3[:,4] > complimits_1[2]]
	age3_label = r' (100 < $\tau \leq$ 400 Myr)'
	print('Number by Age in Bins: ({}/{}/{}), Total = {}'.format(len(gal_array_1_age1), len(gal_array_1_age2), len(gal_array_1_age3), len(gal_array_1_age1) + len(gal_array_1_age2) + len(gal_array_1_age3)))
	print('Fraction Above Mass Completeness Limits: {} / {}, {} / {}, {} / {}, Total = {} / {}'.format(len(gal_array_1_age1_masslimit), len(gal_array_1_age1), len(gal_array_1_age2_masslimit), len(gal_array_1_age2), len(gal_array_1_age3_masslimit), len(gal_array_1_age3), len(gal_array_1_age1_masslimit) + len(gal_array_1_age2_masslimit) + len(gal_array_1_age3_masslimit), len(gal_array_1_age1) + len(gal_array_1_age2) + len(gal_array_1_age3)))
	gal_array_1_masslimit = np.concatenate([gal_array_1_age1_masslimit, gal_array_1_age2_masslimit, gal_array_1_age3_masslimit])

	# Create classic 3 age bins (2)
	print('Array #2')
	print('Input Completeness Limits: {:.1e}, {:.1e}, {:.1e}'.format(complimits_2[0], complimits_2[1], complimits_2[2]))
	print('1 = (0, 10] Myr + input completeness limit')
	gal_array_2_age1 = gal_array_2[gal_array_2[:,6] <= 10.01*1E6]
	gal_array_2_age1_masslimit = gal_array_2_age1[gal_array_2_age1[:,4] > complimits_2[0]]
	age1_label = r' ($\tau \leq$ 10 Myr)'
	print('2 = (10, 100] Myr + input completeness limit')
	gal_array_2_age2_tmp = gal_array_2[gal_array_2[:,6] > 10.01*1E6]
	gal_array_2_age2 = gal_array_2_age2_tmp[gal_array_2_age2_tmp[:,6] <= 100.01*1E6]
	gal_array_2_age2_masslimit = gal_array_2_age2[gal_array_2_age2[:,4] > complimits_2[1]]
	age2_label = r' (10 < $\tau \leq$ 100 Myr)'
	print('3 = (100, 400] Myr + input completeness limit')
	gal_array_2_age3_tmp = gal_array_2[gal_array_2[:,6] > 100.01*1E6]
	gal_array_2_age3 = gal_array_2_age3_tmp[gal_array_2_age3_tmp[:,6] <= 400.01*1E6]
	gal_array_2_age3_masslimit = gal_array_2_age3[gal_array_2_age3[:,4] > complimits_2[2]]
	age3_label = r' (100 < $\tau \leq$ 400 Myr)'
	print('Number by Age in Bins: ({}/{}/{}), Total = {}'.format(len(gal_array_2_age1), len(gal_array_2_age2), len(gal_array_2_age3), len(gal_array_2_age1) + len(gal_array_2_age2) + len(gal_array_2_age3)))
	print('Fraction Above Mass Completeness Limits: {} / {}, {} / {}, {} / {}, Total = {} / {}'.format(len(gal_array_2_age1_masslimit), len(gal_array_2_age1), len(gal_array_2_age2_masslimit), len(gal_array_2_age2), len(gal_array_2_age3_masslimit), len(gal_array_2_age3), len(gal_array_2_age1_masslimit) + len(gal_array_2_age2_masslimit) + len(gal_array_2_age3_masslimit), len(gal_array_2_age1) + len(gal_array_2_age2) + len(gal_array_2_age3)))
	gal_array_2_masslimit = np.concatenate([gal_array_2_age1_masslimit, gal_array_2_age2_masslimit, gal_array_2_age3_masslimit])

	gal_array_age1_1_masslimit = gal_array_1_age1[gal_array_1_age1[:,4] > complimits_1[2]]
	nequal_age1_1, binsequal_age1_1, binsequal_width_age1_1, binsequal_centre_age1_1, nequal_fit_age1_1, binsequal_fit_age1_1, nequal_dM_age1_1, nequal_fit_dM_age1_1, nequal_dlogM_age1_1, nequal_fit_dlogM_age1_1, nequalcum_age1_1, nequalcum_fit_age1_1, nequal_fit_age1_1_err, nequal_fit_age1_1_dM_err, nequal_fit_age1_1_dlogM_err = makearrayhistequal(gal_array_age1_1_masslimit, complimits_1[2], 0)
	gal_array_age1_2_masslimit = gal_array_2_age1[gal_array_2_age1[:,4] > complimits_2[2]]
	nequal_age1_2, binsequal_age1_2, binsequal_width_age1_2, binsequal_centre_age1_2, nequal_fit_age1_2, binsequal_fit_age1_2, nequal_dM_age1_2, nequal_fit_dM_age1_2, nequal_dlogM_age1_2, nequal_fit_dlogM_age1_2, nequalcum_age1_2, nequalcum_fit_age1_2, nequal_fit_age1_2_err, nequal_fit_age1_2_dM_err, nequal_fit_age1_2_dlogM_err = makearrayhistequal(gal_array_age1_2_masslimit, complimits_2[2], 0)
	gal_array_age2_1_masslimit = gal_array_1_age2[gal_array_1_age2[:,4] > complimits_1[2]]
	nequal_age2_1, binsequal_age2_1, binsequal_width_age2_1, binsequal_centre_age2_1, nequal_fit_age2_1, binsequal_fit_age2_1, nequal_dM_age2_1, nequal_fit_dM_age2_1, nequal_dlogM_age2_1, nequal_fit_dlogM_age2_1, nequalcum_age2_1, nequalcum_fit_age2_1, nequal_fit_age2_1_err, nequal_fit_age2_1_dM_err, nequal_fit_age2_1_dlogM_err = makearrayhistequal(gal_array_age2_1_masslimit, complimits_1[2], 0)
	gal_array_age2_2_masslimit = gal_array_2_age2[gal_array_2_age2[:,4] > complimits_2[2]]
	nequal_age2_2, binsequal_age2_2, binsequal_width_age2_2, binsequal_centre_age2_2, nequal_fit_age2_2, binsequal_fit_age2_2, nequal_dM_age2_2, nequal_fit_dM_age2_2, nequal_dlogM_age2_2, nequal_fit_dlogM_age2_2, nequalcum_age2_2, nequalcum_fit_age2_2, nequal_fit_age2_2_err, nequal_fit_age2_2_dM_err, nequal_fit_age2_2_dlogM_err = makearrayhistequal(gal_array_age2_2_masslimit, complimits_2[2], 0)
	gal_array_age3_1_masslimit = gal_array_1_age3[gal_array_1_age3[:,4] > complimits_1[2]]
	nequal_age3_1, binsequal_age3_1, binsequal_width_age3_1, binsequal_centre_age3_1, nequal_fit_age3_1, binsequal_fit_age3_1, nequal_dM_age3_1, nequal_fit_dM_age3_1, nequal_dlogM_age3_1, nequal_fit_dlogM_age3_1, nequalcum_age3_1, nequalcum_fit_age3_1, nequal_fit_age3_1_err, nequal_fit_age3_1_dM_err, nequal_fit_age3_1_dlogM_err = makearrayhistequal(gal_array_age3_1_masslimit, complimits_1[2], 0)
	gal_array_age3_2_masslimit = gal_array_2_age3[gal_array_2_age3[:,4] > complimits_2[2]]
	nequal_age3_2, binsequal_age3_2, binsequal_width_age3_2, binsequal_centre_age3_2, nequal_fit_age3_2, binsequal_fit_age3_2, nequal_dM_age3_2, nequal_fit_dM_age3_2, nequal_dlogM_age3_2, nequal_fit_dlogM_age3_2, nequalcum_age3_2, nequalcum_fit_age3_2, nequal_fit_age3_2_err, nequal_fit_age3_2_dM_err, nequal_fit_age3_2_dlogM_err = makearrayhistequal(gal_array_age3_2_masslimit, complimits_2[2], 0)

	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_Comp1_A1_' + typename + '_CDF_Fit')
	print('>>>')
	fig = plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111)
	xdata_lim_1_float, ydata_lim_1_float = makecdffun(gal_array_1_age1[:,4], complimits_1[0])
	xdata_lim_2_float, ydata_lim_2_float = makecdffun(gal_array_2_age1[:,4], complimits_2[0])
	p1, = plt.plot([-1, -1, -1, -1], color = 'steelblue', linestyle = '-', label = 'Chandar et al. (2015)')
	p1, = plt.plot([-1, -1, -1, -1], color = 'orange', linestyle = '-', label = 'LEGUS')
	plt.legend(loc = 'upper right', title = galname + age1_label + '\n Number = {:.0f} / {:.0f}'.format(len(ydata_lim_1_float), len(ydata_lim_2_float)))
	p9, = plt.plot([complimits_1[0], complimits_1[0]], [1E-10, 1E10], 'k--')
	p9, = plt.plot([complimits_2[0], complimits_2[0]], [1E-10, 1E10], 'k--')
	plt.xlabel(r'M [M$_\odot$]')
	plt.ylabel(r'N (> M)')
	plt.xscale('log', nonposx = 'clip')
	plt.yscale('log', nonposy = 'clip')
	plt.axis(range_mass + [0.1, 10000.])
	ax1.xaxis.set_major_formatter(log10_labels_format)
	ax1.yaxis.set_major_formatter(log10_labels_format)
	plt.xlabel(r'log (M/M$_\odot$)')
	plt.ylabel(r'log N (> M)')
	plt.savefig('./FiguresSummary/' + galname + '_Comp1_A1_' + typename + '_CDF_Fit_Ex.png')
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_Comp1_A2_' + typename + '_CDF_Fit')
	print('>>>')
	fig = plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111)
	xdata_lim_1_float, ydata_lim_1_float = makecdffun(gal_array_1_age2[:,4], complimits_1[1])
	xdata_lim_2_float, ydata_lim_2_float = makecdffun(gal_array_2_age2[:,4], complimits_2[1])
	p1, = plt.plot([-1, -1, -1, -1], color = 'steelblue', linestyle = '-', label = 'Chandar et al. (2015)')
	p1, = plt.plot([-1, -1, -1, -1], color = 'orange', linestyle = '-', label = 'LEGUS')
	plt.legend(loc = 'upper right', title = galname + r' (10 Myr < $\tau \leq$ 100 Myr)' + '\n Number = {:.0f} / {:.0f}'.format(len(ydata_lim_1_float), len(ydata_lim_2_float)))
	p9, = plt.plot([complimits_1[1], complimits_1[1]], [1E-10, 1E10], 'k--')
	p9, = plt.plot([complimits_2[1], complimits_2[1]], [1E-10, 1E10], 'k--')
	plt.xlabel(r'M [M$_\odot$]')
	plt.ylabel(r'N (> M)')
	plt.xscale('log', nonposx = 'clip')
	plt.yscale('log', nonposy = 'clip')
	plt.axis(range_mass + [0.1, 10000.])
	ax1.xaxis.set_major_formatter(log10_labels_format)
	ax1.yaxis.set_major_formatter(log10_labels_format)
	plt.xlabel(r'log (M/M$_\odot$)')
	plt.ylabel(r'log N (> M)')
	plt.savefig('./FiguresSummary/' + galname + '_Comp1_A2_' + typename + '_CDF_Fit_Ex.png')
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_Comp1_A3_' + typename + '_CDF_Fit')
	print('>>>')
	fig = plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111)
	xdata_lim_1_float, ydata_lim_1_float = makecdffun(gal_array_1_age3[:,4], complimits_1[2])
	xdata_lim_2_float, ydata_lim_2_float = makecdffun(gal_array_2_age3[:,4], complimits_2[2])
	p1, = plt.plot([-1, -1, -1, -1], color = 'steelblue', linestyle = '-', label = 'Chandar et al. (2015)')
	p1, = plt.plot([-1, -1, -1, -1], color = 'orange', linestyle = '-', label = 'LEGUS')
	plt.legend(loc = 'upper right', title = galname + r' (100 Myr < $\tau \leq$ 400 Myr)' + '\n Number = {:.0f} / {:.0f}'.format(len(ydata_lim_1_float), len(ydata_lim_2_float)))
	p9, = plt.plot([complimits_1[2], complimits_1[2]], [1E-10, 1E10], 'k--')
	p9, = plt.plot([complimits_2[2], complimits_2[2]], [1E-10, 1E10], 'k--')
	plt.xlabel(r'M [M$_\odot$]')
	plt.ylabel(r'N (> M)')
	plt.xscale('log', nonposx = 'clip')
	plt.yscale('log', nonposy = 'clip')
	plt.axis(range_mass + [0.1, 10000.])
	ax1.xaxis.set_major_formatter(log10_labels_format)
	ax1.yaxis.set_major_formatter(log10_labels_format)
	plt.xlabel(r'log (M/M$_\odot$)')
	plt.ylabel(r'log N (> M)')
	plt.savefig('./FiguresSummary/' + galname + '_Comp1_A3_' + typename + '_CDF_Fit_Ex.png')
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_Comp2_A1_' + typename + '_Histogram_Equal_dM')
	print('>>>')
	fig = plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111)
	plt.step(binsequal_age1_1, np.append(nequal_dM_age1_1[0], nequal_dM_age1_1), color = 'k', alpha = 0.5)
	plt.plot(binsequal_fit_age1_1, nequal_fit_dM_age1_1, color = 'steelblue', marker = '^', linestyle = 'None')
	plt.step(binsequal_age1_2, np.append(nequal_dM_age1_2[0], nequal_dM_age1_2), color = 'k', alpha = 0.5)
	plt.plot(binsequal_fit_age1_2, nequal_fit_dM_age1_2, color = 'orange', marker = '^', linestyle = 'None')
	plt.legend(loc = 'upper right', title = galnameout + age1_label + '\n Number per bin = {:.0f}'.format(np.average(nequal_age1_1)))
	p9, = plt.plot([complimits_1[1], complimits_1[1]], [1E-10, 1E10], 'k--')
	p9, = plt.plot([complimits_2[1], complimits_2[1]], [1E-10, 1E10], 'k--')
	plt.xlabel(r'M [M$_\odot$]')
	plt.ylabel(r'dN/dM')
	plt.xscale('log', nonposx = 'clip')
	plt.yscale('log', nonposy = 'clip')
	plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(nequal_dM_age1_1), np.power(10, 1.3) * np.nanmax(nequal_dM_age1_1)])
	ax1.xaxis.set_major_formatter(log10_labels_format)
	ax1.yaxis.set_major_formatter(log10_labels_format)
	plt.xlabel(r'log (M/M$_\odot$)')
	plt.ylabel(r'log (dN/dM)')
	plt.savefig('./FiguresSummary/' + galname + '_Comp2_A1_' + typename + '_Histogram_Equal_dM_Ex.png')
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_Comp2_A2_' + typename + '_Histogram_Equal_dM')
	print('>>>')
	fig = plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111)
	plt.step(binsequal_age2_1, np.append(nequal_dM_age2_1[0], nequal_dM_age2_1), color = 'k', alpha = 0.5)
	plt.plot(binsequal_fit_age2_1, nequal_fit_dM_age2_1, color = 'steelblue', marker = '^', linestyle = 'None')
	plt.step(binsequal_age2_2, np.append(nequal_dM_age2_2[0], nequal_dM_age2_2), color = 'k', alpha = 0.5)
	plt.plot(binsequal_fit_age2_2, nequal_fit_dM_age2_2, color = 'orange', marker = '^', linestyle = 'None')
	plt.legend(loc = 'upper right', title = galnameout + r' (10 Myr < $\tau \leq$ 100 Myr)' + '\n Number per bin = {:.0f}'.format(np.average(nequal_age2_1)))
	p9, = plt.plot([complimits_1[1], complimits_1[1]], [1E-10, 1E10], 'k--')
	p9, = plt.plot([complimits_2[1], complimits_2[1]], [1E-10, 1E10], 'k--')
	plt.xlabel(r'M [M$_\odot$]')
	plt.ylabel(r'dN/dM')
	plt.xscale('log', nonposx = 'clip')
	plt.yscale('log', nonposy = 'clip')
	plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(nequal_dM_age2_1), np.power(10, 1.3) * np.nanmax(nequal_dM_age2_1)])
	ax1.xaxis.set_major_formatter(log10_labels_format)
	ax1.yaxis.set_major_formatter(log10_labels_format)
	plt.xlabel(r'log (M/M$_\odot$)')
	plt.ylabel(r'log (dN/dM)')
	plt.savefig('./FiguresSummary/' + galname + '_Comp2_A2_' + typename + '_Histogram_Equal_dM_Ex.png')
	plt.close()
	
	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_Comp2_A3_' + typename + '_Histogram_Equal_dM')
	print('>>>')
	fig = plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111)
	plt.step(binsequal_age3_1, np.append(nequal_dM_age3_1[0], nequal_dM_age3_1), color = 'k', alpha = 0.5)
	plt.plot(binsequal_fit_age3_1, nequal_fit_dM_age3_1, color = 'steelblue', marker = '^', linestyle = 'None')
	plt.step(binsequal_age3_2, np.append(nequal_dM_age3_2[0], nequal_dM_age3_2), color = 'k', alpha = 0.5)
	plt.plot(binsequal_fit_age3_2, nequal_fit_dM_age3_2, color = 'orange', marker = '^', linestyle = 'None')
	plt.legend(loc = 'upper right', title = galnameout + r' ($\tau \leq$ 400 Myr)' + '\n Number per bin = {:.0f}'.format(np.average(nequal_age3_1)))
	p9, = plt.plot([complimits_1[1], complimits_1[1]], [1E-10, 1E10], 'k--')
	p9, = plt.plot([complimits_2[1], complimits_2[1]], [1E-10, 1E10], 'k--')
	plt.xlabel(r'M [M$_\odot$]')
	plt.ylabel(r'dN/dM')
	plt.xscale('log', nonposx = 'clip')
	plt.yscale('log', nonposy = 'clip')
	plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(nequal_dM_age3_1), np.power(10, 1.3) * np.nanmax(nequal_dM_age3_1)])
	ax1.xaxis.set_major_formatter(log10_labels_format)
	ax1.yaxis.set_major_formatter(log10_labels_format)
	plt.xlabel(r'log (M/M$_\odot$)')
	plt.ylabel(r'log (dN/dM)')
	plt.savefig('./FiguresSummary/' + galname + '_Comp2_A3_' + typename + '_Histogram_Equal_dM_Ex.png')
	plt.close()

def masteranalysisfunctionSCGMC(galname, gal_array, gal_array_gmc, range_mass, complimits, complimits_gmc):

	'''
	Function: Perform analysis using a combination of cluster and GMC mass distributions
	'''

	typename = 'SCGMC'
	galnameout = galnameoutfun(galname)

	print('>>>')
	print('>>> Start - Analysis (SC + GMC)')
	print('>>>')

	# Define bins for the following code
	mass_bins_log = np.power(10, np.linspace(2, 9, num = 29))
	mass_bins_log_test1 = np.power(10, np.linspace(2, 9, num = 43))
	mass_bins_log_test2 = np.power(10, np.linspace(2, 9, num = 15))
	age_bins_log = np.power(10, np.linspace(4, 11, num = 29))
	mass_bins_log_plot = np.power(10, np.linspace(np.log10(np.min(range_mass)), np.log10(np.max(range_mass))))

	# Create classic 3 age bins
	print('Input Completeness Limits: {:.1e}, {:.1e}, {:.1e}'.format(complimits[0], complimits[1], complimits[2]))
	print('1 = (0, 10] Myr + input completeness limit')
	gal_array_age1 = gal_array[gal_array[:,6] <= 10.01*1E6]
	gal_array_age1_masslimit = gal_array_age1[gal_array_age1[:,4] > complimits[0]]
	age1_label = r' ($\tau \leq$ 10 Myr)'
	print('2 = (10, 100] Myr + input completeness limit')
	gal_array_age2_tmp = gal_array[gal_array[:,6] > 10.01*1E6]
	gal_array_age2 = gal_array_age2_tmp[gal_array_age2_tmp[:,6] <= 100.01*1E6]
	gal_array_age2_masslimit = gal_array_age2[gal_array_age2[:,4] > complimits[1]]
	age2_label = r' (10 < $\tau \leq$ 100 Myr)'
	print('3 = (100, 400] Myr + input completeness limit')
	gal_array_age3_tmp = gal_array[gal_array[:,6] > 100.01*1E6]
	gal_array_age3 = gal_array_age3_tmp[gal_array_age3_tmp[:,6] <= 400.01*1E6]
	gal_array_age3_masslimit = gal_array_age3[gal_array_age3[:,4] > complimits[2]]
	age3_label = r' (100 < $\tau \leq$ 400 Myr)'
	print('Number by Age in Bins: ({}/{}/{}), Total = {}'.format(len(gal_array_age1), len(gal_array_age2), len(gal_array_age3), len(gal_array_age1) + len(gal_array_age2) + len(gal_array_age3)))
	print('Fraction Above Mass Completeness Limits: {} / {}, {} / {}, {} / {}, Total = {} / {}'.format(len(gal_array_age1_masslimit), len(gal_array_age1), len(gal_array_age2_masslimit), len(gal_array_age2), len(gal_array_age3_masslimit), len(gal_array_age3), len(gal_array_age1_masslimit) + len(gal_array_age2_masslimit) + len(gal_array_age3_masslimit), len(gal_array_age1) + len(gal_array_age2) + len(gal_array_age3)))
	gal_array_masslimit = np.concatenate([gal_array_age1_masslimit, gal_array_age2_masslimit, gal_array_age3_masslimit])

	# Create mass bins
	gal_array_young = gal_array[gal_array[:,6] <= 400.01*1E6]
	gal_array_mass1 = gal_array_young[gal_array_young[:,4] <= 1E4]
	gal_array_mass2_tmp = gal_array_young[gal_array_young[:,4] > 1E4]
	gal_array_mass2 = gal_array_mass2_tmp[gal_array_mass2_tmp[:,4] <= 1E6]
	gal_array_mass3 = gal_array_young[gal_array_young[:,4] > 1E6]
	gal_array_mass4_tmp = gal_array_young[gal_array_young[:,4] > 3.*1E5]
	gal_array_mass4 = gal_array_mass4_tmp[gal_array_mass4_tmp[:,4] <= 1E6]
	gal_array_len = len(gal_array_mass1) + len(gal_array_mass2) + len(gal_array_mass3)
	print('Number by Mass < 400 Myr: ({}/{}/{}) --> Total = {}'.format(len(gal_array_mass1), len(gal_array_mass2), len(gal_array_mass3), gal_array_len))

	# Find a new completeness limit to be 5% of the maximum GMC mass
	percentile_gal_value = np.nanmax(gal_array_gmc[:,4]) * 0.05
	print('5 percent of maximum mass: {:.3e} or log M = {:.2f}'.format(percentile_gal_value, np.log10(percentile_gal_value)))
	gal_array_gmc_masslimit_percentile = gal_array_gmc[gal_array_gmc[:,4] > percentile_gal_value]
	gal_array_gmc_masslimit_survey = gal_array_gmc[gal_array_gmc[:,4] > complimits_gmc]

	# Creating Equal Width Bins
	print('>>>')
	print('>>> Creating Equal Width Bins')
	print('>>>')
	print('- All GMCs Above Mass Limit:')
	n, bins, bins_width, bins_centre, n_fit, bins_fit, n_dM, n_fit_dM, n_dlogM, n_fit_dlogM, ncum, ncum_fit, n_fit_err, n_fit_dM_err, n_fit_dlogM_err = makearrayhist(gal_array_gmc, mass_bins_log, complimits_gmc)
	print('- All GMCs Above Mass Limit (new):')
	nnew, binsnew, binsnew_width, binsnew_centre, nnew_fit, binsnew_fit, nnew_dM, nnew_fit_dM, nnew_dlogM, nnew_fit_dlogM, nnewcum, nnewcum_fit, nnew_fit_err, nnew_fit_dM_err, nnew_fit_dlogM_err = makearrayhist(gal_array_gmc, mass_bins_log, percentile_gal_value)
	print('- Clusters in Age Bin 1:')
	n_age1, bins_age1, bins_width_age1, bins_centre_age1, n_fit_age1, bins_fit_age1, n_dM_age1, n_fit_dM_age1, n_dlogM_age1, n_fit_dlogM_age1, ncum_age1, ncum_fit_age1, n_fit_age1_err, n_fit_age1_dM_err, n_fit_age1_dlogM_err = makearrayhist(gal_array_age1, mass_bins_log, complimits[0])
	print('- Clusters in Age Bin 2:')
	n_age2, bins_age2, bins_width_age2, bins_centre_age2, n_fit_age2, bins_fit_age2, n_dM_age2, n_fit_dM_age2, n_dlogM_age2, n_fit_dlogM_age2, ncum_age2, ncum_fit_age2, n_fit_age2_err, n_fit_age2_dM_err, n_fit_age2_dlogM_err = makearrayhist(gal_array_age2, mass_bins_log, complimits[1])
	print('- Clusters in Age Bin 3:')
	n_age3, bins_age3, bins_width_age3, bins_centre_age3, n_fit_age3, bins_fit_age3, n_dM_age3, n_fit_dM_age3, n_dlogM_age3, n_fit_dlogM_age3, ncum_age3, ncum_fit_age3, n_fit_age3_err, n_fit_age3_dM_err, n_fit_age3_dlogM_err = makearrayhist(gal_array_age3, mass_bins_log, complimits[2])

	# Creating Equal Number Bins
	print('>>>')
	print('>>> Creating Equal Number Bins')
	print('>>>')
	print('- All GMCs Above Mass Limit:')
	nequal, binsequal, binsequal_width, binsequal_centre, nequal_fit, binsequal_fit, nequal_dM, nequal_fit_dM, nequal_dlogM, nequal_fit_dlogM, nequalcum, nequalcum_fit, nequal_fit_err, nequal_fit_dM_err, nequal_fit_dlogM_err = makearrayhistequal(gal_array_gmc, complimits_gmc, 0)
	print('- All GMCs Above Mass Limit (new):')
	nnewequal, binsnewequal, binsnewequal_width, binsnewequal_centre, nnewequal_fit, binsnewequal_fit, nnewequal_dM, nnewequal_fit_dM, nnewequal_dlogM, nnewequal_fit_dlogM, nnewequalcum, nnewequalcum_fit, nnewequal_fit_err, nnewequal_fit_dM_err, nnewequal_fit_dlogM_err = makearrayhistequal(gal_array_gmc, percentile_gal_value, 0)
	print('- Clusters in Age Bin 1:')
	nequal_age1, binsequal_age1, binsequal_width_age1, binsequal_centre_age1, nequal_fit_age1, binsequal_fit_age1, nequal_dM_age1, nequal_fit_dM_age1, nequal_dlogM_age1, nequal_fit_dlogM_age1, nequalcum_age1, nequalcum_fit_age1, nequal_fit_age1_err, nequal_fit_age1_dM_err, nequal_fit_age1_dlogM_err = makearrayhistequal(gal_array_age1, complimits[0], 0)
	print('- Clusters in Age Bin 2:')
	nequal_age2, binsequal_age2, binsequal_width_age2, binsequal_centre_age2, nequal_fit_age2, binsequal_fit_age2, nequal_dM_age2, nequal_fit_dM_age2, nequal_dlogM_age2, nequal_fit_dlogM_age2, nequalcum_age2, nequalcum_fit_age2, nequal_fit_age2_err, nequal_fit_age2_dM_err, nequal_fit_age2_dlogM_err = makearrayhistequal(gal_array_age2, complimits[1], 0)
	print('- Clusters in Age Bin 3:')
	nequal_age3, binsequal_age3, binsequal_width_age3, binsequal_centre_age3, nequal_fit_age3, binsequal_fit_age3, nequal_dM_age3, nequal_fit_dM_age3, nequal_dlogM_age3, nequal_fit_dlogM_age3, nequalcum_age3, nequalcum_fit_age3, nequal_fit_age3_err, nequal_fit_age3_dM_err, nequal_fit_age3_dlogM_err = makearrayhistequal(gal_array_age3, complimits[2], 0)

	# Creating DSS Plots
	print('>>>')
	print('>>> Creating DSS Plots')
	print('>>>')
	RA, DEC, PA, LONGAXIS, SHORTAXIS, REGPATH, POLYPATH = returnposinfo(galnameout)
	### --==--==--==-- ###
	plt.rcParams['xtick.labelsize'] = 24
	plt.rcParams['ytick.labelsize'] = 24
	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_01_' + typename + 'Position_Age')
	print('>>>')
	gc = aplpy.FITSFigure(DSS_folder + galnameout + '_DSS.fits')
	gc.show_grayscale()
	gc.set_title(galnameout)
	if galnameout == 'Antennae':
		gc.recenter(RA, DEC + 0.01, 0.055)
	elif galnameout == 'NGC3627':
		gc.recenter(RA, DEC + (0.15 * LONGAXIS), LONGAXIS * plotstretchfactor)
	elif galnameout == 'M31':
		gc.recenter(RA, DEC + (0.25 * LONGAXIS), LONGAXIS * plotstretchfactor)
	elif galnameout == 'LMC':
		gc.recenter(RA, DEC + (0.25 * LONGAXIS), LONGAXIS * plotstretchfactor * 0.67)
	else:
		gc.recenter(RA, DEC, LONGAXIS * plotstretchfactor)
	if len(gal_array_age1_masslimit[:,2]) > 0:
		gc.show_markers(np.asarray(gal_array_age1_masslimit[:,2], dtype = np.float32), np.asarray(gal_array_age1_masslimit[:,3], dtype = np.float32), marker = '+', c = 'b', s = 4, alpha = 0.50, label = r'$\tau \leq$ 10 Myr')
	if len(gal_array_age2_masslimit[:,2]) > 0:
		gc.show_markers(np.asarray(gal_array_age2_masslimit[:,2], dtype = np.float32), np.asarray(gal_array_age2_masslimit[:,3], dtype = np.float32), marker = '+', c = 'g', s = 4, alpha = 0.50, label = r'10 < $\tau \leq$ 100 Myr')
	if len(gal_array_age3_masslimit[:,2]) > 0:
		gc.show_markers(np.asarray(gal_array_age3_masslimit[:,2], dtype = np.float32), np.asarray(gal_array_age3_masslimit[:,3], dtype = np.float32), marker = '+', c = 'r', s = 4, alpha = 0.50, label = r'100 < $\tau \leq$ 400 Myr')
	gc.save('./Figures' + typename + '/' + galname + '_01_' + typename + '_Position_Age_1.png')
	gc.show_markers(np.asarray(gal_array_gmc_masslimit_percentile[:,2], dtype = np.float32), np.asarray(gal_array_gmc_masslimit_percentile[:,3], dtype = np.float32), marker = 'o', c = 'cyan', s = 4, alpha = 1.0, label = 'GMC (> 5%)')
	plt.legend(loc = 'upper right', fontsize = 16)
	if galnameout != 'LMC':
		gc.show_polygons(POLYPATH, lw = 2, alpha = 1.0, color = 'brown')
	gc.save('./Figures' + typename + '/' + galname + '_01_' + typename + '_Position_Age_2.png')
	gc.show_markers(np.asarray(gal_array_gmc_masslimit_survey[:,2], dtype = np.float32), np.asarray(gal_array_gmc_masslimit_survey[:,3], dtype = np.float32), marker = 'o', c = 'purple', s = 4, alpha = 0.5, label = 'GMC (> comp limit)')
	gc.show_markers(np.asarray(gal_array_gmc_masslimit_percentile[:,2], dtype = np.float32), np.asarray(gal_array_gmc_masslimit_percentile[:,3], dtype = np.float32), marker = 'o', c = 'cyan', s = 4, alpha = 1.0)
	plt.legend(loc = 'upper right', fontsize = 16)
	gc.save('./Figures' + typename + '/' + galname + '_01_' + typename + '_Position_Age_3.png')
	gc.close()
	### --==--==--==-- ###
	plt.rcParams['xtick.labelsize'] = 42
	plt.rcParams['ytick.labelsize'] = 42
	plot_x = np.linspace(-10, 10)

	# Creating DSS Plots
	print('>>>')
	print('>>> Creating DSS Plots')
	print('>>>')
	RA, DEC, PA, LONGAXIS, SHORTAXIS, REGPATH, POLYPATH = returnposinfo(galnameout)
	### --==--==--==-- ###
	plt.rcParams['xtick.labelsize'] = 24
	plt.rcParams['ytick.labelsize'] = 24
	### --==--==--==-- ###
	gc = aplpy.FITSFigure(DSS_folder + galnameout + '_DSS.fits')
	gc.show_grayscale()
	gc.set_title(galnameout)
	if galnameout == 'Antennae':
		gc.recenter(RA, DEC + 0.01, 0.055)
	elif galnameout == 'NGC3627':
		gc.recenter(RA, DEC + (0.15 * LONGAXIS), LONGAXIS * plotstretchfactor)
	elif galnameout == 'M31':
		gc.recenter(RA, DEC + (0.25 * LONGAXIS), LONGAXIS * plotstretchfactor)
	elif galnameout == 'LMC':
		gc.recenter(RA, DEC + (0.25 * LONGAXIS), LONGAXIS * plotstretchfactor * 0.67)
	else:
		gc.recenter(RA, DEC, LONGAXIS * plotstretchfactor)
	gc.show_markers(np.asarray(gal_array_gmc[:,2], dtype = np.float32), np.asarray(gal_array_gmc[:,3], dtype = np.float32), marker = 'o', c = 'cyan', s = 4, alpha = 0.9, label = 'GMC')
	if len(gal_array_age1_masslimit[:,2]) > 0:
		gc.show_markers(np.asarray(gal_array_age1_masslimit[:,2], dtype = np.float32), np.asarray(gal_array_age1_masslimit[:,3], dtype = np.float32), marker = '+', c = 'b', s = 4, alpha = 0.9, label = r'Clusters')
	# if len(gal_array_age2_masslimit[:,2]) > 0:
	# 	gc.show_markers(np.asarray(gal_array_age2_masslimit[:,2], dtype = np.float32), np.asarray(gal_array_age2_masslimit[:,3], dtype = np.float32), marker = '+', c = 'g', s = 4, alpha = 0.50, label = r'10 < $\tau \leq$ 100 Myr')
	# if len(gal_array_age3_masslimit[:,2]) > 0:
	# 	gc.show_markers(np.asarray(gal_array_age3_masslimit[:,2], dtype = np.float32), np.asarray(gal_array_age3_masslimit[:,3], dtype = np.float32), marker = '+', c = 'r', s = 4, alpha = 0.50, label = r'100 < $\tau \leq$ 400 Myr')
	plt.legend(loc = 'upper right', fontsize = 26)
	if galnameout != 'LMC':
		gc.show_polygons(POLYPATH, lw = 2, alpha = 1.0, color = 'brown')
	gc.save('./Figures' + typename + '/' + galname + '_01_' + typename + '_Position_Age_4.png')
	gc.close()
	### --==--==--==-- ###
	plt.rcParams['xtick.labelsize'] = 42
	plt.rcParams['ytick.labelsize'] = 42
	plot_x = np.linspace(-10, 10)


	testalphalist = ['NGC3256', 'Antennae', 'NGC3256_cut', 'Antennae_cut']

	# Set up test alpha cases for certain galaxies
	if galname in testalphalist:

		# Set up special arrays
		gal_array_gmc_spec = gal_array_gmc.copy()
		for i in range(0, len(gal_array_gmc_spec)):
			gal_array_gmc_spec[i, 4] = gal_array_gmc_spec[i, 4] / (4.35 / 0.8)

		# Find a new completeness limit to be 5% of the maximum GMC mass
		percentile_gal_value_spec = np.nanmax(gal_array_gmc_spec[:,4]) * 0.05
		complimits_gmc_spec = complimits_gmc / (4.35 / 0.8)
		gal_array_gmc_spec_masslimit_percentile = gal_array_gmc_spec[gal_array_gmc_spec[:,4] > percentile_gal_value_spec]
		gal_array_gmc_spec_masslimit_survey = gal_array_gmc_spec[gal_array_gmc_spec[:,4] > complimits_gmc_spec]
		n_spec, bins_spec, bins_width_spec, bins_centre_spec, n_fit_spec, bins_fit_spec, n_dM_spec, n_fit_dM_spec, n_dlogM_spec, n_fit_dlogM_spec, ncum_spec, ncum_fit_spec, n_fit_err_spec, n_fit_dM_err_spec, n_fit_dlogM_err_spec = makearrayhist(gal_array_gmc_spec, mass_bins_log, complimits_gmc_spec)
		nnew_spec, binsnew_spec, binsnew_width_spec, binsnew_centre_spec, nnew_fit_spec, binsnew_fit_spec, nnew_dM_spec, nnew_fit_dM_spec, nnew_dlogM_spec, nnew_fit_dlogM_spec, nnewcum_spec, nnewcum_fit_spec, nnew_fit_err_spec, nnew_fit_dM_err_spec, nnew_fit_dlogM_err_spec = makearrayhist(gal_array_gmc_spec, mass_bins_log, percentile_gal_value_spec)

	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_02A1_' + typename + '_Histogram_dM')
	print('>>>')
	fig = plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111)
	plt.plot(bins_fit, n_fit_dM, 'bs', label = r'GMCs')
	if len(n_dM_age1) > 0:
		plt.plot(bins_fit_age1, n_fit_dM_age1, color = 'orange', marker = '^', linestyle = 'None', label = r'$\tau \leq$ 10 Myr')
	if len(n_dM_age2) > 0:
		plt.plot(bins_fit_age2, n_fit_dM_age2, 'gs', label = r'10 < $\tau \leq$ 100 Myr')
	if len(n_dM_age3) > 0:
		plt.plot(bins_fit_age3, n_fit_dM_age3, 'ro', label = r'100 < $\tau \leq$ 400 Myr')
	p9, = plt.plot([percentile_gal_value, percentile_gal_value], [1E-10, 1E10], 'k--')
	p9, = plt.plot([complimits_gmc, complimits_gmc], [1E-10, 1E10], 'k:')
	plt.legend(title = '{}: SC + GMC'.format(galnameout), loc = 'upper right')
	plt.xscale('log', nonposx = 'clip')
	plt.minorticks_off()
	plt.yscale('log', nonposy = 'clip')
	plt.minorticks_off()
	plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit_dM), np.power(10, 1.3) * np.nanmax(n_fit_dM_age1)])
	ax1.xaxis.set_major_formatter(log10_labels_format)
	ax1.yaxis.set_major_formatter(log10_labels_format)
	plt.xlabel(r'log (M/M$_\odot$)')
	plt.ylabel(r'log (dN/dM)')
	plt.savefig('./Figures' + typename + '/' + galname + '_02A1_' + typename + '_Histogram_dM_Ex.png')
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_02A2_' + typename + '_Histogram_dM')
	print('>>>')
	fig = plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111)
	plt.plot(binsnew_fit, nnew_fit_dM, 'bs', label = r'GMCs')
	if len(nequal_dM_age1) > 0:
		plt.plot(bins_fit_age1, n_fit_dM_age1, color = 'orange', marker = '^', linestyle = 'None', label = r'$\tau \leq$ 10 Myr')
	if len(n_dM_age2) > 0:
		plt.plot(bins_fit_age2, n_fit_dM_age2, 'gs', label = r'10 < $\tau \leq$ 100 Myr')
	if len(n_dM_age3) > 0:
		plt.plot(bins_fit_age3, n_fit_dM_age3, 'ro', label = r'100 < $\tau$ $\leq$ 400 Myr')
	p9, = plt.plot([percentile_gal_value, percentile_gal_value], [1E-10, 1E10], 'k--')
	p9, = plt.plot([complimits_gmc, complimits_gmc], [1E-10, 1E10], 'k:')
	plt.legend(title = '{}: SC + GMC'.format(galnameout), loc = 'upper right')
	plt.xscale('log', nonposx = 'clip')
	plt.minorticks_off()
	plt.yscale('log', nonposy = 'clip')
	plt.minorticks_off()
	plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit_dM), np.power(10, 1.3) * np.nanmax(n_fit_dM_age1)])
	ax1.xaxis.set_major_formatter(log10_labels_format)
	ax1.yaxis.set_major_formatter(log10_labels_format)
	plt.xlabel(r'log (M/M$_\odot$)')
	plt.ylabel(r'log (dN/dM)')
	plt.savefig('./Figures' + typename + '/' + galname + '_02A2_' + typename + '_Histogram_dM_Ex.png')
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_02B1_' + typename + '_Histogram_dM')
	print('>>>')
	fig = plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111)
	plt.plot(bins_fit, n_fit_dM, 'bs', label = r'GMCs')
	if len(nequal_dM_age1) > 0:
		plt.plot(bins_fit_age1, n_fit_dM_age1, color = 'orange', marker = '^', linestyle = 'None', label = r'$\tau \leq$ 10 Myr')
	popt_fit1, popt_fit2 = curve_fit1slope(bins_fit_age1, n_fit_dM_age1, bins_fit, n_fit_dM, range_mass, complimits[0], complimits_gmc, 2)
	if galname in testalphalist:
		plt.plot(bins_fit_spec, n_fit_dM_spec, 'rs')
		popt_fit1_spec, popt_fit2_spec = curve_fit1slope(bins_fit_age1, n_fit_dM_age1, bins_fit_spec, n_fit_dM_spec, range_mass, complimits[0], complimits_gmc_spec, 3)
	p9, = plt.plot([percentile_gal_value, percentile_gal_value], [1E-10, 1E10], 'k--')
	p9, = plt.plot([complimits_gmc, complimits_gmc], [1E-10, 1E10], 'k:')
	plt.legend(title = '{}: SC + GMC'.format(galnameout), loc = 'upper right')
	plt.xscale('log', nonposx = 'clip')
	plt.minorticks_off()
	plt.yscale('log', nonposy = 'clip')
	plt.minorticks_off()
	plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit_dM), np.power(10, 1.3) * np.nanmax(n_fit_dM_age1)])
	ax1.xaxis.set_major_formatter(log10_labels_format)
	ax1.yaxis.set_major_formatter(log10_labels_format)
	plt.xlabel(r'log (M/M$_\odot$)')
	plt.ylabel(r'log (dN/dM)')
	plt.savefig('./Figures' + typename + '/' + galname + '_02B1_' + typename + '_Histogram_dM_Ex.png')
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_02B2_A1_' + typename + '_Histogram_dM')
	print('>>>')
	fig = plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111)
	plt.plot(binsnew_fit, nnew_fit_dM, 'bs', label = r'GMCs')
	if len(nequal_dM_age1) > 0:
		plt.plot(bins_fit_age1, n_fit_dM_age1, color = 'orange', marker = '^', linestyle = 'None', label = r'$\tau \leq$ 10 Myr')
	popt_fit1, popt_fit2 = curve_fit1slope(bins_fit_age1, n_fit_dM_age1, binsnew_fit, nnew_fit_dM, range_mass, complimits[0], percentile_gal_value, 2)
	if galname in testalphalist:
		plt.plot(binsnew_fit_spec, nnew_fit_dM_spec, 'rs')
		popt_fit1_spec, popt_fit2_spec = curve_fit1slope(bins_fit_age1, n_fit_dM_age1, bins_fit_spec, n_fit_dM_spec, range_mass, complimits[0], complimits_gmc_spec, 3)
	p9, = plt.plot([percentile_gal_value, percentile_gal_value], [1E-10, 1E10], 'k--')
	# p9, = plt.plot([complimits_gmc, complimits_gmc], [1E-10, 1E10], 'k:')
	plt.legend(title = '{}: SC + GMC'.format(galnameout), loc = 'upper right')
	plt.xscale('log', nonposx = 'clip')
	plt.minorticks_off()
	plt.yscale('log', nonposy = 'clip')
	plt.minorticks_off()
	plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit_dM), np.power(10, 1.3) * np.nanmax(n_fit_dM_age1)])
	ax1.xaxis.set_major_formatter(log10_labels_format)
	ax1.yaxis.set_major_formatter(log10_labels_format)
	plt.xlabel(r'log (M/M$_\odot$)')
	plt.ylabel(r'log (dN/dM)')
	plt.savefig('./Figures' + typename + '/' + galname + '_02B2_A1_' + typename + '_Histogram_dM_Ex.png')
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_02B2_A2_' + typename + '_Histogram_dM')
	print('>>>')
	fig = plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111)
	plt.plot(binsnew_fit, nnew_fit_dM, 'bs', label = r'GMCs')
	if len(nequal_dM_age2) > 0:
		plt.plot(bins_fit_age2, n_fit_dM_age2, color = 'green', marker = 's', linestyle = 'None', label = r'10 < $\tau \leq$ 100 Myr')
	popt_fit1, popt_fit2 = curve_fit1slope(bins_fit_age2, n_fit_dM_age2, binsnew_fit, nnew_fit_dM, range_mass, complimits[1], percentile_gal_value, 2)
	p9, = plt.plot([percentile_gal_value, percentile_gal_value], [1E-10, 1E10], 'k--')
	p9, = plt.plot([complimits_gmc, complimits_gmc], [1E-10, 1E10], 'k:')
	plt.legend(title = '{}: SC + GMC'.format(galnameout), loc = 'upper right')
	plt.xscale('log', nonposx = 'clip')
	plt.minorticks_off()
	plt.yscale('log', nonposy = 'clip')
	plt.minorticks_off()
	plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit_dM), np.power(10, 1.3) * np.nanmax(n_fit_dM_age1)])
	ax1.xaxis.set_major_formatter(log10_labels_format)
	ax1.yaxis.set_major_formatter(log10_labels_format)
	plt.xlabel(r'log (M/M$_\odot$)')
	plt.ylabel(r'log (dN/dM)')
	plt.savefig('./Figures' + typename + '/' + galname + '_02B2_A2_' + typename + '_Histogram_dM_Ex.png')
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_02B2_A3_' + typename + '_Histogram_dM')
	print('>>>')
	fig = plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111)
	plt.plot(binsnew_fit, nnew_fit_dM, 'bs', label = r'GMCs')
	if len(nequal_dM_age3) > 0:
		plt.plot(bins_fit_age3, n_fit_dM_age3, color = 'red', marker = 'o', linestyle = 'None', label = r'100 < $\tau \leq$ 400 Myr')
	popt_fit1, popt_fit2 = curve_fit1slope(bins_fit_age3, n_fit_dM_age3, binsnew_fit, nnew_fit_dM, range_mass, complimits[2], percentile_gal_value, 2)
	p9, = plt.plot([percentile_gal_value, percentile_gal_value], [1E-10, 1E10], 'k--')
	p9, = plt.plot([complimits_gmc, complimits_gmc], [1E-10, 1E10], 'k:')
	plt.legend(title = '{}: SC + GMC'.format(galnameout), loc = 'upper right')
	plt.xscale('log', nonposx = 'clip')
	plt.yscale('log', nonposy = 'clip')
	plt.minorticks_off()
	plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(n_fit_dM), np.power(10, 1.3) * np.nanmax(n_fit_dM_age1)])
	ax1.xaxis.set_major_formatter(log10_labels_format)
	ax1.yaxis.set_major_formatter(log10_labels_format)
	plt.xlabel(r'log (M/M$_\odot$)')
	plt.ylabel(r'log (dN/dM)')
	plt.savefig('./Figures' + typename + '/' + galname + '_02B2_A3_' + typename + '_Histogram_dM_Ex.png')
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_03A_' + typename + '_CDF_Fit')
	print('>>>')
	fig = plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111)
	gal_array_gmc_xdata_float, gal_array_gmc_ydata_float, gal_array_gmc_xdata_float_lim, gal_array_gmc_ydata_float_lim = makecdffunnoplot(gal_array_gmc[:,4], complimits_gmc)
	plt.step(gal_array_gmc_xdata_float, gal_array_gmc_ydata_float, color = 'grey', linestyle = '--')
	plt.step(gal_array_gmc_xdata_float_lim, gal_array_gmc_ydata_float_lim, label = r'GMCs', color = 'b')
	if len(gal_array_age1) > 0:
		gal_array_age1_xdata_float, gal_array_age1_ydata_float, gal_array_age1_xdata_float_lim, gal_array_age1_ydata_float_lim = makecdffunnoplot(gal_array_age1[:,4], complimits[0])
		plt.step(gal_array_age1_xdata_float, gal_array_age1_ydata_float, color = 'grey', linestyle = '--')
		plt.step(gal_array_age1_xdata_float_lim, gal_array_age1_ydata_float_lim, label = r'$\tau \leq$ 10 Myr', color = 'orange')
	if len(gal_array_age2) > 0:
		gal_array_age2_xdata_float, gal_array_age2_ydata_float, gal_array_age2_xdata_float_lim, gal_array_age2_ydata_float_lim = makecdffunnoplot(gal_array_age2[:,4], complimits[1])
		plt.step(gal_array_age2_xdata_float, gal_array_age2_ydata_float, color = 'grey', linestyle = '--')
		plt.step(gal_array_age2_xdata_float_lim, gal_array_age2_ydata_float_lim, label = age2_label, color = 'g')
	if len(gal_array_age3) > 0:
		gal_array_age3_xdata_float, gal_array_age3_ydata_float, gal_array_age3_xdata_float_lim, gal_array_age3_ydata_float_lim = makecdffunnoplot(gal_array_age3[:,4], complimits[2])
		plt.step(gal_array_age3_xdata_float, gal_array_age3_ydata_float, color = 'grey', linestyle = '--')
		plt.step(gal_array_age3_xdata_float_lim, gal_array_age3_ydata_float_lim, label = age3_label, color = 'r')
	plt.legend(title = '{}: SC + GMC'.format(galnameout), loc = 'upper right')
	plt.xscale('log', nonposx = 'clip')
	plt.yscale('log', nonposy = 'clip')
	plt.axis(range_mass + [np.power(10, -0.5), np.power(10, 3.5)])
	ax1.xaxis.set_major_formatter(log10_labels_format)
	ax1.yaxis.set_major_formatter(log10_labels_format)
	plt.xlabel(r'log (M/M$_\odot$)')
	plt.ylabel(r'log N (> M)')
	plt.savefig('./Figures' + typename + '/' + galname + '_03A_' + typename + '_CDF_Fit_Ex.png')
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_03B_' + typename + '_CDF_Fit_Model')
	print('>>>')
	fig = plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111)
	gal_array_gmc_xdata_float, gal_array_gmc_ydata_float, gal_array_gmc_xdata_float_lim, gal_array_gmc_ydata_float_lim = makecdffunnoplot(gal_array_gmc[:,4], complimits_gmc)
	plt.step(gal_array_gmc_xdata_float, gal_array_gmc_ydata_float, color = 'grey', linestyle = '--')
	plt.step(gal_array_gmc_xdata_float_lim, gal_array_gmc_ydata_float_lim, label = r'GMCs', color = 'b')
	plt.step(gal_array_gmc_xdata_float * 0.04, gal_array_gmc_ydata_float, color = 'grey', linestyle = '--')
	plt.step(gal_array_gmc_xdata_float_lim * 0.04, gal_array_gmc_ydata_float_lim, label = r'GMCs $\times$ 0.04', color = 'b', linestyle = '--')
	plt.step(gal_array_gmc_xdata_float[gal_array_gmc_xdata_float < np.nanmin(gal_array_gmc_xdata_float_lim)] * 0.01, gal_array_gmc_ydata_float[gal_array_gmc_xdata_float < np.nanmin(gal_array_gmc_xdata_float_lim)], color = 'grey', linestyle = '--')
	plt.step(gal_array_gmc_xdata_float_lim * 0.01, gal_array_gmc_ydata_float_lim, label = r'GMCs $\times$ 0.01', color = 'b', linestyle = ':')
	plt.step(gal_array_gmc_xdata_float[gal_array_gmc_xdata_float < np.nanmin(gal_array_gmc_xdata_float_lim)] * 0.001, gal_array_gmc_ydata_float[gal_array_gmc_xdata_float < np.nanmin(gal_array_gmc_xdata_float_lim)], color = 'grey', linestyle = '--')
	plt.step(gal_array_gmc_xdata_float_lim * 0.001, gal_array_gmc_ydata_float_lim, label = r'GMCs $\times$ 0.001', color = 'b', linestyle = '-.')
	if len(gal_array_age1) > 0:
		gal_array_age1_xdata_float, gal_array_age1_ydata_float, gal_array_age1_xdata_float_lim, gal_array_age1_ydata_float_lim = makecdffunnoplot(gal_array_age1[:,4], complimits[0])
		plt.step(gal_array_age1_xdata_float, gal_array_age1_ydata_float, color = 'grey', linestyle = '--')
		plt.step(gal_array_age1_xdata_float_lim, gal_array_age1_ydata_float_lim, label = r'$\tau \leq$ 10 Myr', color = 'orange')
	plt.legend(title = '{}: SC + GMC'.format(galnameout), loc = 'upper right')
	plt.xscale('log', nonposx = 'clip')
	plt.yscale('log', nonposy = 'clip')
	plt.axis(range_mass + [np.power(10, -0.5), np.power(10, 3.5)])
	ax1.xaxis.set_major_formatter(log10_labels_format)
	ax1.yaxis.set_major_formatter(log10_labels_format)
	plt.xlabel(r'log (M/M$_\odot$)')
	plt.ylabel(r'log N (> M)')
	plt.savefig('./Figures' + typename + '/' + galname + '_03B_' + typename + '_CDF_Fit_Model_Ex.png')
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_04A1_' + typename + '_Histogram_Equal_dM')
	print('>>>')
	fig = plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111)
	plt.plot(binsequal_fit, nequal_fit_dM, 'bs', label = r'GMCs')
	if len(nequal_dM_age1) > 0:
		plt.plot(binsequal_fit_age1, nequal_fit_dM_age1, color = 'orange', marker = '^', linestyle = 'None', label = r'$\tau \leq$ 10 Myr')
	if len(nequal_dM_age2) > 0:
		plt.plot(binsequal_fit_age2, nequal_fit_dM_age2, 'gs', label = age2_label)
	if len(nequal_dM_age3) > 0:
		plt.plot(binsequal_fit_age3, nequal_fit_dM_age3, 'ro', label = age3_label)
	p9, = plt.plot([percentile_gal_value, percentile_gal_value], [1E-10, 1E10], 'k--')
	p9, = plt.plot([complimits_gmc, complimits_gmc], [1E-10, 1E10], 'k:')
	plt.legend(title = '{}: SC + GMC'.format(galnameout), loc = 'upper right')
	plt.xscale('log', nonposx = 'clip')
	plt.minorticks_off()
	plt.yscale('log', nonposy = 'clip')
	plt.minorticks_off()
	plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(nequal_fit_dM), np.power(10, 1.3) * np.nanmax(nequal_fit_dM_age1)])
	ax1.xaxis.set_major_formatter(log10_labels_format)
	ax1.yaxis.set_major_formatter(log10_labels_format)
	plt.xlabel(r'log (M/M$_\odot$)')
	plt.ylabel(r'log (dN/dM)')
	plt.savefig('./Figures' + typename + '/' + galname + '_04A1_' + typename + '_Histogram_Equal_dM_Ex.png')
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_04A2_' + typename + '_Histogram_Equal_dM')
	print('>>>')
	fig = plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111)
	plt.plot(binsnewequal_fit, nnewequal_fit_dM, 'bs', label = r'GMCs')
	if len(nequal_dM_age1) > 0:
		plt.plot(binsequal_fit_age1, nequal_fit_dM_age1, color = 'orange', marker = '^', linestyle = 'None', label = r'$\tau \leq$ 10 Myr')
	if len(nequal_dM_age2) > 0:
		plt.plot(binsequal_fit_age2, nequal_fit_dM_age2, 'gs', label = age2_label)
	if len(nequal_dM_age3) > 0:
		plt.plot(binsequal_fit_age3, nequal_fit_dM_age3, 'ro', label = age3_label)
	p9, = plt.plot([percentile_gal_value, percentile_gal_value], [1E-10, 1E10], 'k--')
	p9, = plt.plot([complimits_gmc, complimits_gmc], [1E-10, 1E10], 'k:')
	plt.legend(title = '{}: SC + GMC'.format(galnameout), loc = 'upper right')
	plt.xscale('log', nonposx = 'clip')
	plt.minorticks_off()
	plt.yscale('log', nonposy = 'clip')
	plt.minorticks_off()
	plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(nequal_fit_dM), np.power(10, 1.3) * np.nanmax(nequal_fit_dM_age1)])
	ax1.xaxis.set_major_formatter(log10_labels_format)
	ax1.yaxis.set_major_formatter(log10_labels_format)
	plt.xlabel(r'log (M/M$_\odot$)')
	plt.ylabel(r'log (dN/dM)')
	plt.savefig('./Figures' + typename + '/' + galname + '_04A2_' + typename + '_Histogram_Equal_dM_Ex.png')
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_04B1_' + typename + '_Histogram_Equal_dM')
	print('>>>')
	fig = plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111)
	plt.plot(binsequal_fit, nequal_fit_dM, 'bs', label = r'GMCs')
	if len(nequal_dM_age1) > 0:
		plt.plot(binsequal_fit_age1, nequal_fit_dM_age1, color = 'orange', marker = '^', linestyle = 'None', label = r'$\tau \leq$ 10 Myr')
	popt_fit1, popt_fit2 = curve_fit1slope(binsequal_fit_age1, nequal_fit_dM_age1, binsequal_fit, nequal_fit_dM, range_mass, complimits[0], np.nanmax(complimits_gmc), 2)
	p9, = plt.plot([percentile_gal_value, percentile_gal_value], [1E-10, 1E10], 'k--')
	p9, = plt.plot([complimits_gmc, complimits_gmc], [1E-10, 1E10], 'k:')
	plt.legend(title = '{}: SC + GMC'.format(galnameout), loc = 'upper right')
	plt.xscale('log', nonposx = 'clip')
	plt.minorticks_off()
	plt.yscale('log', nonposy = 'clip')
	plt.minorticks_off()
	plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(nequal_fit_dM), np.power(10, 1.3) * np.nanmax(nequal_fit_dM_age1)])
	ax1.xaxis.set_major_formatter(log10_labels_format)
	ax1.yaxis.set_major_formatter(log10_labels_format)
	plt.xlabel(r'log (M/M$_\odot$)')
	plt.ylabel(r'log (dN/dM)')
	plt.savefig('./Figures' + typename + '/' + galname + '_04B1_' + typename + '_Histogram_Equal_dM_Ex.png')
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> ' + galname + '_04B2_' + typename + '_Histogram_Equal_dM')
	print('>>>')
	fig = plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111)
	plt.plot(binsnewequal_fit, nnewequal_fit_dM, 'bs', label = r'GMCs')
	if len(nequal_dM_age1) > 0:
		plt.plot(binsequal_fit_age1, nequal_fit_dM_age1, color = 'orange', marker = '^', linestyle = 'None', label = r'$\tau \leq$ 10 Myr')
	popt_fit1, popt_fit2 = curve_fit1slope(binsequal_fit_age1, nequal_fit_dM_age1, binsnewequal_fit, nnewequal_fit_dM, range_mass, complimits[0], percentile_gal_value, 2)
	p9, = plt.plot([percentile_gal_value, percentile_gal_value], [1E-10, 1E10], 'k--')
	p9, = plt.plot([complimits_gmc, complimits_gmc], [1E-10, 1E10], 'k:')
	plt.legend(title = '{}: SC + GMC'.format(galnameout), loc = 'upper right')
	plt.xscale('log', nonposx = 'clip')
	plt.minorticks_off()
	plt.yscale('log', nonposy = 'clip')
	plt.minorticks_off()
	plt.axis(range_mass + [np.power(10, -0.5) * np.nanmin(nequal_fit_dM), np.power(10, 1.3) * np.nanmax(nequal_fit_dM_age1)])
	ax1.xaxis.set_major_formatter(log10_labels_format)
	ax1.yaxis.set_major_formatter(log10_labels_format)
	plt.xlabel(r'log (M/M$_\odot$)')
	plt.ylabel(r'log (dN/dM)')
	plt.savefig('./Figures' + typename + '/' + galname + '_04B2_' + typename + '_Histogram_Equal_dM_Ex.png')
	plt.close()

	return 0

#------------------------------------------------------------------------------
###
# (11) Code Snippets (Analysis I - Base Functions)
###

def runscmostmass(gal_array, plot_marker, plot_label):

	'''
	Function: Plot the most massive cluster vs total mass for three age ranges
	'''

	# Create different age bins
	gal_array_age1 = gal_array[gal_array[:,6] <= 10*1E6]
	gal_array_age2_tmp = gal_array[gal_array[:,6] > 10*1E6]
	gal_array_age2 = gal_array_age2_tmp[gal_array_age2_tmp[:,6] <= 100*1E6]
	gal_array_age3_tmp = gal_array[gal_array[:,6] > 100*1E6]
	gal_array_age3 = gal_array_age3_tmp[gal_array_age3_tmp[:,6] <= 400*1E6]

	A1_mass = gal_array_age1[:,4]
	A2_mass = gal_array_age2[:,4]
	A3_mass = gal_array_age3[:,4]
	A1_most_mass = np.nanmax(A1_mass)
	A2_most_mass = np.nanmax(A2_mass)
	A3_most_mass = np.nanmax(A3_mass)
	A1_filter = filter(lambda x: x >= 1E4, A1_mass)
	A1_tot_mass = np.sum(A1_filter)
	A2_filter = filter(lambda x: x >= 1E4, A2_mass)
	A2_tot_mass = np.sum(A2_filter)
	A3_filter = filter(lambda x: x >= 1E4, A3_mass)
	A3_tot_mass = np.sum(A3_filter)
	plt.plot(A1_tot_mass, A1_most_mass, markerfacecolor = plot_marker, markeredgecolor = 'black', marker = '^', linestyle = 'None', markersize = 16)
	plt.plot(A2_tot_mass, A2_most_mass, markerfacecolor = plot_marker, markeredgecolor = 'black', marker = 's', linestyle = 'None', markersize = 16)
	plt.plot(A3_tot_mass, A3_most_mass, markerfacecolor = plot_marker, markeredgecolor = 'black', marker = 'o', linestyle = 'None', label = plot_label, markersize = 16)
	if plot_label in ['NGC 3256']:
		plt.arrow(A1_tot_mass, A1_most_mass, A1_tot_mass * 0.75, 0, head_width = A1_most_mass * 0.25, head_length = A1_tot_mass * 0.25, fc = 'k', ec = 'k')
	if plot_label in ['Antennae', 'NGC 3256']:
		plt.arrow(A2_tot_mass, A2_most_mass, A2_tot_mass * 0.75, 0, head_width = A2_most_mass * 0.25, head_length = A2_tot_mass * 0.25, fc = 'k', ec = 'k')
		plt.arrow(A3_tot_mass, A3_most_mass, A3_tot_mass * 0.75, 0, head_width = A3_most_mass * 0.25, head_length = A3_tot_mass * 0.25, fc = 'k', ec = 'k')
	print(plot_label)
	print(len(A1_filter), len(A1_mass), len(A2_filter), len(A2_mass), len(A3_filter), len(A3_mass))
	print(A1_most_mass, A2_most_mass, A3_most_mass, A1_tot_mass, A2_tot_mass, A3_tot_mass)

	return 0

def rungmcmostmass(gal_array, plot_marker, plot_label):

	'''
	Function: Plot the most massive GMC vs total mass 
	'''

	gmc_threshold = 1E5
	gmc_mass = gal_array[:,4]
	gmc_most_mass = np.nanmax(gmc_mass)

	gmc_filter = filter(lambda x: x >= gmc_threshold, gmc_mass)
	gmc_tot_mass = np.sum(gmc_filter)

	plt.plot(gmc_tot_mass, gmc_most_mass, markerfacecolor = plot_marker, markeredgecolor = 'black', marker = '^', linestyle = 'None', markersize = 16)

	return 0

def agecuts_outputarrays(gal_array, galname, complimits, ageflag):

	'''
	Function: Output cluster arrays with chosen cuts (age/mass)
	'''

	if ageflag == 1:
		print(galname)
		print('Ageflag: 1 = (0, 10] Myr + input completeness limit A1')
		gal_array_age1 = gal_array[gal_array[:,6] <= 10.01*1E6]
		gal_array_masslimit = gal_array_age1[gal_array_age1[:,4] > complimits[0]]
		age_label = r' ($\tau \leq$ 10 Myr)'
		gal_array_out = gal_array_age1
	elif ageflag == 2:
		print('Ageflag: 2 = (10, 100] Myr + input completeness limit A2')
		gal_array_age2_tmp = gal_array[gal_array[:,6] > 10.01*1E6]
		gal_array_age2 = gal_array_age2_tmp[gal_array_age2_tmp[:,6] <= 100.01*1E6]
		gal_array_masslimit = gal_array_age2[gal_array_age2[:,4] > complimits[1]]
		age_label = r' (10 < $\tau \leq$ 100 Myr)'
		gal_array_out = gal_array_age2
	elif ageflag == 3:
		print('Ageflag: 3 = (100, 400] Myr + input completeness limit A3')
		gal_array_age3_tmp = gal_array[gal_array[:,6] > 100.01*1E6]
		gal_array_age3 = gal_array_age3_tmp[gal_array_age3_tmp[:,6] <= 400.01*1E6]
		gal_array_masslimit = gal_array_age3[gal_array_age3[:,4] > complimits[2]]
		age_label = r' (100 < $\tau \leq$ 400 Myr)'
		gal_array_out = gal_array_age3
	elif ageflag == 5:
		print('Ageflag: 5 = [1, 200] Myr + input completeness limit A3')
		gal_array_age5_tmp = gal_array[gal_array[:,6] > 0.99*1E6]
		gal_array_age5 = gal_array_age5_tmp[gal_array_age5_tmp[:,6] <= 200.01*1E6]
		gal_array_masslimit = gal_array_age5[gal_array_age5[:,4] > complimits[2]]
		age_label = r' (1 <= $\tau \leq$ 200 Myr)'
		gal_array_out = gal_array_age5
	elif ageflag == 0:
		print('Ageflag: 0 = A1 + A2 + A3 + completeness limits')
		gal_array_age1 = gal_array[gal_array[:,6] <= 10.01*1E6]
		gal_array_age1_masslimit = gal_array_age1[gal_array_age1[:,4] > complimits[0]]
		gal_array_age2_tmp = gal_array[gal_array[:,6] > 10.01*1E6]
		gal_array_age2 = gal_array_age2_tmp[gal_array_age2_tmp[:,6] <= 100.01*1E6]
		gal_array_age2_masslimit = gal_array_age2[gal_array_age2[:,4] > complimits[1]]
		gal_array_age3_tmp = gal_array[gal_array[:,6] > 100.01*1E6]
		gal_array_age3 = gal_array_age3_tmp[gal_array_age3_tmp[:,6] <= 400.01*1E6]
		gal_array_age3_masslimit = gal_array_age3[gal_array_age3[:,4] > complimits[2]]
		gal_array_masslimit = np.concatenate([gal_array_age1_masslimit, gal_array_age2_masslimit, gal_array_age3_masslimit])
		age_label = r''
		gal_array_out = np.concatenate([gal_array_age1, gal_array_age2, gal_array_age3])
	elif ageflag == -1:
		print('Ageflag: -1 = All + Max(Complimits)')
		gal_array_masslimit = gal_array[gal_array[:,4] > np.nanmax(complimits)]
		age_label = r''
		gal_array_out = gal_array
	else:
		print('Ageflag: All + Max(Complimits)')
		gal_array_masslimit = gal_array[gal_array[:,4] > np.nanmax(complimits)]
		age_label = r''
		gal_array_out = gal_array

	return gal_array_masslimit, age_label, gal_array_out

def plotgal_likelihood(gal_array, galname, complimits, plotrow, axes, flag):
	
	'''
	Function: Plot 3 likelihood Plot for 3 Age Bins in One Galaxy
	'''

	print('Starting {}'.format(galname))

	# Create classic 3 age bins
	print('Input Completeness Limits: {:.1e}, {:.1e}, {:.1e}'.format(complimits[0], complimits[1], complimits[2]))
	gal_array_age1_masslimit, age1_label, gal_array_age1_out = agecuts_outputarrays(gal_array, galname, complimits, 1)
	gal_array_age2_masslimit, age2_label, gal_array_age2_out = agecuts_outputarrays(gal_array, galname, complimits, 2)
	gal_array_age3_masslimit, age3_label, gal_array_age3_out = agecuts_outputarrays(gal_array, galname, complimits, 3)


	# array1, galname, agename, plotaxes, errorflag, sigma
	if flag == 1:

		if plotrow == -1:
			likelihoodplot_out_age1 = likelihoodplot(gal_array_age1_masslimit, galname, '_A1', axes[0], 1, 0.3, 0, 'Test')
			likelihoodplot_out_age2 = likelihoodplot(gal_array_age2_masslimit, galname, '_A2', axes[1], 1, 0.3, 0, 'Test')
			likelihoodplot_out_age3 = likelihoodplot(gal_array_age3_masslimit, galname, '_A3', axes[2], 1, 0.3, 0, 'Test')
		else:
			likelihoodplot_out_age1 = likelihoodplot(gal_array_age1_masslimit, galname, '_A1', axes[plotrow, 0], 1, 0.3, 0, 'Test')
			likelihoodplot_out_age2 = likelihoodplot(gal_array_age2_masslimit, galname, '_A2', axes[plotrow, 1], 1, 0.3, 0, 'Test')
			likelihoodplot_out_age3 = likelihoodplot(gal_array_age3_masslimit, galname, '_A3', axes[plotrow, 2], 1, 0.3, 0, 'Test')

	else:

		if plotrow == -1:
			likelihoodplot_out_age1 = likelihoodplot(gal_array_age1_masslimit, galname, '_A1', axes[0], 0, 0, 0, 'Test')
			likelihoodplot_out_age2 = likelihoodplot(gal_array_age2_masslimit, galname, '_A2', axes[1], 0, 0, 0, 'Test')
			likelihoodplot_out_age3 = likelihoodplot(gal_array_age3_masslimit, galname, '_A3', axes[2], 0, 0, 0, 'Test')
		else:
			likelihoodplot_out_age1 = likelihoodplot(gal_array_age1_masslimit, galname, '_A1', axes[plotrow, 0], 0, 0, 0, 'Test')
			likelihoodplot_out_age2 = likelihoodplot(gal_array_age2_masslimit, galname, '_A2', axes[plotrow, 1], 0, 0, 0, 'Test')
			likelihoodplot_out_age3 = likelihoodplot(gal_array_age3_masslimit, galname, '_A3', axes[plotrow, 2], 0, 0, 0, 'Test')

	return 0

def plotaxes_likelihood(gal_array, galname, complimits, plotaxes, ageflag, loc, sigma, output, outputfile, cutoff_mass_mult = 100, flag_simple = 0):

	'''
	Function: Make One Likelihood Plot at Chosen Location
	'''

	print('Starting {} - Likelihood (plotaxes)'.format(galname))

	if ageflag == 'GMC_P':
		print('>>> {}: 5% Threshold'.format(galname))
		complimits_val = np.nanmax(gal_array[:,4]) * 0.05
		gal_array_masslimit, age_label, gal_array_out = agecuts_outputarrays(gal_array, galname, complimits_val, ageflag)
	else:
		gal_array_masslimit, age_label, gal_array_out = agecuts_outputarrays(gal_array, galname, complimits, ageflag)

	if ageflag == 1:
		if len(gal_array_masslimit) > 0:
			likelihoodplot(gal_array_masslimit, galname, '_A1', plotaxes, loc, sigma, output, outputfile, cutoff_mass_mult, flag_simple = flag_simple)
	elif ageflag == 2:
		if len(gal_array_masslimit) > 0:
			likelihoodplot(gal_array_masslimit, galname, '_A2', plotaxes, loc, sigma, output, outputfile, cutoff_mass_mult, flag_simple = flag_simple)
	elif ageflag == 3:
		if len(gal_array_masslimit) > 0:
			likelihoodplot(gal_array_masslimit, galname, '_A3', plotaxes, loc, sigma, output, outputfile, cutoff_mass_mult, flag_simple = flag_simple)
	elif ageflag == 5:
		if len(gal_array_masslimit) > 0:
			likelihoodplot(gal_array_masslimit, galname, '_A5', plotaxes, loc, sigma, output, outputfile, cutoff_mass_mult, flag_simple = flag_simple)
	elif ageflag == 0:
		if len(gal_array_masslimit) > 0:
			likelihoodplot(gal_array_masslimit, galname, '_AL', plotaxes, loc, sigma, output, outputfile, cutoff_mass_mult, flag_simple = flag_simple)
	elif ageflag == -1 or ageflag in ['XB_comb']:
		if len(gal_array_masslimit) > 0:
			likelihoodplot(gal_array_masslimit, galname, 'XB_comb', plotaxes, loc, sigma, output, outputfile, cutoff_mass_mult, flag_simple = flag_simple)
	elif ageflag in ['C']:
		if len(gal_array_masslimit) > 0:
			likelihoodplot(gal_array_masslimit, galname, 'C', plotaxes, loc, sigma, output, outputfile, cutoff_mass_mult, flag_simple = flag_simple)
	elif ageflag in ['GMC', 'GMC_P']:
		likelihoodplot(gal_array_masslimit, galname, 'GMC', plotaxes, loc, sigma, output, outputfile, cutoff_mass_mult, flag_simple = flag_simple)
	else:
		if len(gal_array_masslimit) > 0:
			likelihoodplot(gal_array_masslimit, galname, '', plotaxes, loc, sigma, output, outputfile, cutoff_mass_mult, flag_simple = flag_simple)

def likelihoodplot(array, galname, agename, plotaxes, loc, sigma, output_flag, outputfile, cutoff_mass_mult = 100, flag_simple = 0):

	'''
	Function: Make One Likelihood Plot at Chosen Location
	'''

	# Select mass array
	array4 = array[:,4]

	# Test if the array contains data
	if len(array4) == 0:

		return 0

	# Reset variables
	res_loglikesimplepowerlaw = 0
	res_loglikeschechter = 0
	res_loglikeschechterwithslope2 = 0

	# Set age names
	ageout = ageflagconvert(agename)

	# Set error flag
	if abs(sigma) > 0:
		errorflag = 1
	else:
		errorflag = 0

	# Output to terminal
	print('>>> - <-> - <<<')
	print('Running Likelihood Routine on {} + {}:'.format(galname, ageout))
	print('>>> - <=> - <<<')

	# Set plot range and values
	if agename in ['GMC', 'GMC_P']:
		M0_low = 4.5
		M0_high = 9.0
		M_grid = np.power(10, np.linspace(M0_low, M0_high, 91))
		plot_likexticks = [5.0, 6.0, 7.0, 8.0]
		plot_likeyticks = [0.0, 1.0, 2.0, 3.0]
	elif agename in ['XB_comb']:
		M0_low = 3.0
		M0_high = 8.0
		M_grid = np.power(10, np.linspace(M0_low, M0_high, 101))
		plot_likexticks = [4.0, 5.0, 6.0, 7.0]
		plot_likeyticks = [0.0, 1.0, 2.0, 3.0]
	elif agename in ['XB', 'XB_P']:
		M0_low = 4.5
		M0_high = 8.5
		M_grid = np.power(10, np.linspace(M0_low, M0_high, 81))
		plot_likexticks = [5.0, 6.0, 7.0, 8.0]
		plot_likeyticks = [0.0, 1.0, 2.0, 3.0]
	elif agename in ['C', 'C_P']:
		M0_low = 1.0
		M0_high = 6.0	
		M_grid = np.power(10, np.linspace(M0_low, M0_high, 101))
		plot_likexticks = [2.0, 3.0, 4.0, 5.0]
		plot_likeyticks = [0.0, 1.0, 2.0, 3.0]
	else:
		M0_low = 3.0
		M0_high = 7.5
		M_grid = np.power(10, np.linspace(M0_low, M0_high, 91))
		plot_likexticks = [4.0, 5.0, 6.0, 7.0]
		plot_likeyticks = [0.0, 1.0, 2.0, 3.0]

	# Define common variables	
	Beta_low = 0
	Beta_high = 3
	plot_likerange = [M0_low, M0_high] + [Beta_low, Beta_high]
	G_grid = np.linspace(Beta_low, Beta_high, 61)
	Value_grid = np.zeros((len(G_grid), len(M_grid)))
	boxprops = dict(boxstyle = 'round', facecolor = 'None')

	# Output important parameters and calculate the analytical estimator for the slope
	print('Num in array: {}'.format(len(array4)))
	print('Max/Min in array: {:.2f} - {:.2f}'.format(np.log10(np.nanmin(array4)), np.log10(np.nanmax(array4))))
	sumval = 0
	for i in range(0, len(array4)):
		sumval = sumval + np.log(array4[i]/np.nanmin(array4))
	alpha_est = 1 + len(array4) / sumval
	print('Beta Analytic - Estimator: {:.2f}'.format(alpha_est))

	# Set cutoff masses to be a multiple of the maximum mass in the catalogue
	max_mass = np.nanmax(array4)
	cutoff_mass = np.nanmax(array4) * cutoff_mass_mult
	print('Cutoff Mass Multiple Value: {}'.format(cutoff_mass_mult))
	print('Cutoff Mass (Integration Bound): {:.2f}'.format(np.log10(cutoff_mass)))

	# Define likelihood functions
	if errorflag == 0:

		# Define likelihood function for a Schechter function
		def loglikeschechter(params):

			M0 = params[0]
			Beta = params[1]

			LL = 0
			probsum = 0

			# Switch: Default Non-Log Transformed Method
			if True:
				normfun = lambda x: np.power((x / M0), -Beta) * np.exp(-(x / M0))
				norm = integrate.quad(normfun, np.nanmin(array4), cutoff_mass)[0]
				for i in range(0, len(array4)):
					numerator = np.power((array4[i] / M0), -Beta) * np.exp(-(array4[i] / M0))
					probsum = probsum + np.log(numerator / norm)

			# Switch: Log Transformed Method
			if False:
				logM0 = np.log10(M0)
				denominator = lambda x: np.power(np.power(10, (x - logM0)), -(Beta - 1)) * np.exp(-(np.power(10, x - logM0)))
				denominator_int = integrate.quad(denominator, np.log10(np.nanmin(array4)), np.log10(cutoff_mass))[0]
				for i in range(0, len(array4)):
					logM = np.log10(array4[i])
					numerator_int = np.power(np.power(10, (logM - logM0)), -(Beta - 1)) * np.exp(-(np.power(10, logM - logM0)))
					probsum = probsum + np.log(numerator_int / denominator_int)

			# Output Diagnostics
			# print('Test - Output no Error: {:.2e}, {:.2f}, L:{:.2f}'.format(M0, Beta, -probsum))

			LL = LL + probsum

			return -(LL)

		def loglikesimplepowerlaw(params):

			Beta = params

			LL = 0
			probsum = 0
			M0 = np.nanmin(array4)
			normfun = lambda x: np.power((x / M0), -Beta)
			norm = integrate.quad(normfun, np.nanmin(array4), cutoff_mass)[0]
			for i in range(0, len(array4)):
				probsum = probsum + np.log(np.power((array4[i] / M0), -Beta) / norm)		
			LL = LL + probsum

			return -(LL)

		def loglikeschechterwithslope2(params):

			M0 = params

			LL = 0
			probsum = 0
			normfun = lambda x: np.power((x / M0), -2) * np.exp(-(x / M0))
			norm = integrate.quad(normfun, np.nanmin(array4), cutoff_mass)[0]
			for i in range(0, len(array4)):
				probsum = probsum + np.log(np.power((array4[i] / M0), -2) * np.exp(-(array4[i] / M0)) / norm)
			LL = LL + probsum

			return -(LL)

	else:

		def loglikeschechter(params):

			M0 = params[0]
			Beta = params[1]

			LL = 0
			probsum = 0

			# Switch: Log-Normal Distribution ^^
			if True:

				# Define mass/mass-diff/function arrays
				m_array = np.power(10, np.linspace(np.log10(np.nanmin(array4)), np.log10(cutoff_mass), num_interp))
				m_diff_array = np.diff(m_array)
				function_array = outputmassfunction(sigma, Beta, M0, m_array, 2, 0)

				# Run loop to calculate value of the denominator
				denominator_int = 0
				for i in range(0, len(function_array) - 1):
					sumval = m_diff_array[i] * 0.5 * (function_array[i] + function_array[i+1])
					denominator_int = denominator_int + sumval

				# Run interpolation routine
				m_array_interp = np.power(10, np.linspace(np.log10(np.nanmin(array4)) - 0.05, np.log10(np.nanmax(array4)) + 0.05, num_interp))
				function_array_interep = outputmassfunction(sigma, Beta, M0, m_array_interp, 2, 0)
				function_interp = interpolate.interp1d(m_array_interp, function_array_interep)

				# Calculate numerator using same output mass function
				for i in range(0, len(array4)):
					numerator_int = function_interp(array4[i])
					probsum = probsum + np.log(numerator_int / denominator_int)

			# Output Diagnostics
			# print('Test - Output with Error: {:.2e}, {:.2f}, L:{:.2f}'.format(M0, Beta, -probsum))

			LL = LL + probsum

			return -(LL)

		def loglikesimplepowerlaw(params):

			Beta = params

			LL = 0
			probsum = 0
			M0 = np.nanmin(array4)
			normfun = lambda x: np.power((x / M0), -Beta)
			norm = integrate.quad(normfun, np.nanmin(array4), cutoff_mass)[0]

			for i in range(0, len(array4)):

				logmassval = np.log10(array4[i])
				gaussnorm = lambda x: np.exp(-0.5 * ((x - logmassval) / sigma) ** 2)
				gaussnormval = integrate.quad(gaussnorm, logmassval - (sigma * 5), logmassval + (sigma * 5))[0]
				fun = lambda x: np.log10(np.power((np.power(10, x) / M0), -Beta)) * np.exp(-0.5 * ((x - logmassval) / sigma) ** 2) / gaussnormval
				funval = np.power(10, integrate.quad(fun, logmassval - (sigma * 5), logmassval + (sigma * 5))[0])				
				probsum = probsum + np.log(funval / norm)

			LL = LL + probsum

			return -(LL)

		def loglikeschechterwithslope2(params):

			M0 = params

			LL = 0
			probsum = 0
			normfun = lambda x: np.power((x / M0), -2) * np.exp(-(x / M0))
			norm = integrate.quad(normfun, np.nanmin(array4), cutoff_mass)[0]
			
			for i in range(0, len(array4)):
				
				logmassval = np.log10(array4[i])
				gaussnorm = lambda x: np.exp(-0.5 * ((x - logmassval) / sigma) ** 2)
				gaussnormval = integrate.quad(gaussnorm, logmassval - (sigma * 5), logmassval + (sigma * 5))[0]
				fun = lambda x: np.log10(np.power((np.power(10, x) / M0), -2) * np.exp(-(np.power(10, x) / M0))) * np.exp(-0.5 * ((x - logmassval) / sigma) ** 2) / gaussnormval
				funval = np.power(10, integrate.quad(fun, logmassval - (sigma * 5), logmassval + (sigma * 5))[0])				
				probsum = probsum + np.log(funval / norm)
			
			LL = LL + probsum

			return -(LL)

	# Run maximum likelihood routine
	try:
		res_loglikeschechter = optimize.minimize(loglikeschechter, (max_mass, 2), method = 'Nelder-Mead', options = {'maxiter':1000})
	except ZeroDivisionError:
		res_loglikeschechter = optimize.minimize(loglikeschechter, (max_mass, 2), method = 'Nelder-Mead', options = {'maxiter':1})
		print('Unable to find schehter function due to error')
	res_loglikesimplepowerlaw = optimize.minimize(loglikesimplepowerlaw, [2], method = 'Nelder-Mead')
	res_loglikeschechterwithslope2 = optimize.minimize(loglikeschechterwithslope2, [max_mass], method = 'Nelder-Mead')

	# Print results
	print('>>> Schechter: Likelihood Results')
	print('Max Likelihood Results (Nelder-Meld): {}, {:.3f}, {:.2f}, {:.2f}'.format(res_loglikeschechter.message, res_loglikeschechter.fun, res_loglikeschechter.x[1], np.log10(res_loglikeschechter.x[0])))
	print(' --- Power Law: Likelihood Results (M* = infinity)')
	print('Max Likelihood Results (Nelder-Meld): {}, {:.3f}, {:.2f}'.format(res_loglikesimplepowerlaw.message, res_loglikesimplepowerlaw.fun, res_loglikesimplepowerlaw.x[0]))
	print(' --- Schechter: Likelihood Results (Beta = 2)')
	print('Max Likelihood Results (Nelder-Meld): {}, {:.3f}, {:.2f}'.format(res_loglikeschechterwithslope2.message, res_loglikeschechterwithslope2.fun, np.log10(res_loglikeschechterwithslope2.x[0])))

	# Perform a log-ratio test
	import scipy as scipy
	if res_loglikeschechter.fun < res_loglikesimplepowerlaw.fun:
		df = 2 - 1
		x = 2 * (res_loglikesimplepowerlaw.fun - res_loglikeschechter.fun)
		print('P-value Log-Ratio: {:.2e}'.format(stats.distributions.chi2.sf(x, df)))
	else:
		print('Power Law appears to be a better fit.')

	# Perform an AIC test
	akaike_powerlaw = 4 + (2 * res_loglikesimplepowerlaw.fun)
	akaike_schechter = 6 + (2 * res_loglikeschechter.fun)
	print('AIC (lower is better): {:.0f} (PL) vs {:.0f} (Schechter)'.format(akaike_powerlaw, akaike_schechter))
	print('AIC Test Value: p = {:.0e}'.format(np.exp(-(abs(akaike_powerlaw - akaike_schechter) / 2))))

	# Determine likelihood function value in grid
	for i in range(0, len(G_grid)):
		for ii in range(0, len(M_grid)):
			Value_grid[i, ii] = 1E20
			testval = loglikeschechter([M_grid[ii], G_grid[i]])
			if math.isinf(testval) or math.isnan(testval):
				Value_grid[i, ii] = 1E20
			elif testval < 1E-5:
				Value_grid[i, ii] = 1E20
			elif testval < res_loglikeschechter.fun: # Easy way of removing where the function is not converging
				Value_grid[i, ii] = 1E20
			else:
				Value_grid[i, ii] = testval

	# Use maximum likelihood value (if found) or the minimum value in grid
	# and res_loglikeschechter.fun < np.nanmin(Value_grid[Value_grid > 0])
	if np.isfinite(res_loglikeschechter.fun) and res_loglikeschechter.success == True:
		likemin = res_loglikeschechter.fun
		flag_success = 1
		print ('>>> Used Nelder-Meld Results')

	else:
		likemin = np.nanmin(Value_grid[Value_grid > 0])
		flag_success = 0
		print ('>>> Used Value Grid Minimum Value')

	# Find where min value is
	likegridout = np.nanmin(Value_grid[Value_grid > 0])
	likegrididx = np.where(Value_grid == likegridout)
	likegridG = np.float(G_grid[likegrididx[0]])
	likegridM = np.float(M_grid[likegrididx[1]])

	# Print min/max values in grid
	print('Min/Max in Grid: {:.3e}, {:.3e}'.format(likemin, np.nanmax(Value_grid[Value_grid > 0])))
	print('Min in Grid, located at: {:.3f}, {:.2f}, {:.2f}'.format(likemin, likegridG, np.log10(likegridM)))
	print('Comparison - Max Likelihood Results (Nelder-Meld): {:.3f}, {:.2f}, {:.2f}, Note: Flag {} at nit {}'.format(res_loglikeschechter.fun, res_loglikeschechter.x[1], np.log10(res_loglikeschechter.x[0]), res_loglikeschechter.success, res_loglikeschechter.nit))
	
	######################################################

	## Create extra outputs
	if output_flag == 1:

		# Output file
		np.savetxt('./GridLike/' + outputfile + '.txt', Value_grid, delimiter = ',')

		# Plot 0 - Entire range
		fig = plt.figure(figsize = (13.5, 13.5))
		ax1 = fig.add_subplot(111)
		plt.contourf(np.log10(M_grid), G_grid, Value_grid, np.linspace(np.nanmin(Value_grid[Value_grid > 0]), np.nanmin([np.nanmax(Value_grid[Value_grid > 0]), 20000]), 50))
		plt.plot(np.log10(np.nanmax(array4)), 0.5, 'g^', markersize = 12)
		plt.xlabel(r'log M$_*$ [M$_\odot$]')
		plt.ylabel(r'- $\beta$')
		plt.axis(plot_likerange)
		plt.xticks([4.0, 5.0, 6.0, 7.0])
		plt.yticks([0.0, 1.0, 2.0, 3.0])
		plt.colorbar()
		plt.text(4.0, 2.75, 'N = {}'.format(len(array4)), fontsize = 32, verticalalignment = 'top', bbox = boxprops)
		plt.savefig('./GridLike/' + outputfile + '_1.png')
		plt.close()

		fig = plt.figure(figsize = (13.5, 13.5))
		ax1 = fig.add_subplot(111)
		plt.contourf(np.log10(M_grid), G_grid, Value_grid, np.linspace(np.nanmin(Value_grid[Value_grid > 0]), np.nanmin(Value_grid[Value_grid > 0]) + 200, 50))
		plt.plot(np.log10(np.nanmax(array4)), 0.5, 'g^', markersize = 12)
		plt.xlabel(r'log M$_*$ [M$_\odot$]')
		plt.ylabel(r'- $\beta$')
		plt.axis(plot_likerange)
		plt.xticks([4.0, 5.0, 6.0, 7.0])
		plt.yticks([0.0, 1.0, 2.0, 3.0])
		plt.colorbar()
		plt.text(4.0, 2.75, 'N = {}'.format(len(array4)), fontsize = 32, verticalalignment = 'top', bbox = boxprops)
		plt.savefig('./GridLike/' + outputfile + '_2.png')
		plt.close()

	######################################################

	# Set standard sigma values
	like1sigma = likemin + 1.15
	like2sigma = likemin + 3.09
	like3sigma = likemin + 5.91
	like4sigma = likemin + 9.67
	like5sigma = likemin + 14.37

	# Print results
	print('>>> Errors: Schechter Fit')
	Value_grid_xmin = np.nan_to_num(np.amin(Value_grid, axis = 0))
	Value_grid_ymin = np.nan_to_num(np.amin(Value_grid, axis = 1))
	like_M_idx, like_M_val = find_nearest(M_grid, res_loglikeschechter.x[0])
	like_G_idx, like_G_val = find_nearest(G_grid, res_loglikeschechter.x[1])

	# Check closest grid value
	print('--- Check: Closest Grid Val = M ({:.2f} vs NM - {:.2f}) and G({:.2f} vs NM - {:.2f})'.format(np.log10(like_M_val), np.log10(res_loglikeschechter.x[0]), like_G_val, res_loglikeschechter.x[1]))
	print('--- 1 sigma with value from Grid:')
	like1sigmaValueGrid = np.nanmin(Value_grid[Value_grid > 0]) + 1.15
	like1sigma_x_idx1, like1sigma_x_idx2 = find_nearest2guided(Value_grid_xmin, like1sigmaValueGrid, like_M_idx)
	like1sigma_y_idx1, like1sigma_y_idx2 = find_nearest2guided(Value_grid_ymin, like1sigmaValueGrid, like_G_idx)
	print('--- {:.3f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(like1sigmaValueGrid, np.log10(M_grid[like1sigma_x_idx1]), np.log10(M_grid[like1sigma_x_idx2]), G_grid[like1sigma_y_idx1], G_grid[like1sigma_y_idx2]))
	akaike_powerlaw = 4 + (2 * res_loglikesimplepowerlaw.fun)
	akaike_schechter = 6 + (2 * np.nanmin(Value_grid[Value_grid > 0]))
	print('AIC (lower is better): {:.0f} (PL) vs {:.0f} (Schechter - In GRID)'.format(akaike_powerlaw, akaike_schechter))
	print('AIC Test Value: p = {:.0e}'.format(np.exp(-(abs(akaike_powerlaw - akaike_schechter) / 2))))

	# Print results
	like1sigma_x_idx1, like1sigma_x_idx2 = find_nearest2guided(Value_grid_xmin, like1sigma, like_M_idx)
	like1sigma_y_idx1, like1sigma_y_idx2 = find_nearest2guided(Value_grid_ymin, like1sigma, like_G_idx)
	print('1 sigma: {:.3f}, {:.2f}, {:.2f}'.format(like1sigma, np.log10(M_grid[like1sigma_x_idx1]), np.log10(M_grid[like1sigma_x_idx2])))
	print('1 sigma: {:.3f}, {:.2f}, {:.2f}'.format(like1sigma, G_grid[like1sigma_y_idx1], G_grid[like1sigma_y_idx2]))
	###
	like2sigma_x_idx1, like2sigma_x_idx2 = find_nearest2guided(Value_grid_xmin, like2sigma, like_M_idx)
	like2sigma_y_idx1, like2sigma_y_idx2 = find_nearest2guided(Value_grid_ymin, like2sigma, like_G_idx)
	print('2 sigma: {:.3f}, {:.2f}, {:.2f}'.format(like2sigma, np.log10(M_grid[like2sigma_x_idx1]), np.log10(M_grid[like2sigma_x_idx2])))
	print('2 sigma: {:.3f}, {:.2f}, {:.2f}'.format(like2sigma, G_grid[like2sigma_y_idx1], G_grid[like2sigma_y_idx2]))
	###
	like3sigma_x_idx1, like3sigma_x_idx2 = find_nearest2guided(Value_grid_xmin, like3sigma, like_M_idx)
	like3sigma_y_idx1, like3sigma_y_idx2 = find_nearest2guided(Value_grid_ymin, like3sigma, like_G_idx)
	print('3 sigma: {:.3f}, {:.2f}, {:.2f}'.format(like3sigma, np.log10(M_grid[like3sigma_x_idx1]), np.log10(M_grid[like3sigma_x_idx2])))
	print('3 sigma: {:.3f}, {:.2f}, {:.2f}'.format(like3sigma, G_grid[like3sigma_y_idx1], G_grid[like3sigma_y_idx2]))
	###
	like4sigma_x_idx1, like4sigma_x_idx2 = find_nearest2guided(Value_grid_xmin, like4sigma, like_M_idx)
	like4sigma_y_idx1, like4sigma_y_idx2 = find_nearest2guided(Value_grid_ymin, like4sigma, like_G_idx)
	print('4 sigma: {:.3f}, {:.2f}, {:.2f}'.format(like4sigma, np.log10(M_grid[like4sigma_x_idx1]), np.log10(M_grid[like4sigma_x_idx2])))
	print('4 sigma: {:.3f}, {:.2f}, {:.2f}'.format(like4sigma, G_grid[like4sigma_y_idx1], G_grid[like4sigma_y_idx2]))
	###
	like5sigma_x_idx1, like5sigma_x_idx2 = find_nearest2guided(Value_grid_xmin, like5sigma, like_M_idx)
	like5sigma_y_idx1, like5sigma_y_idx2 = find_nearest2guided(Value_grid_ymin, like5sigma, like_G_idx)
	print('5 sigma: {:.3f}, {:.2f}, {:.2f}'.format(like5sigma, np.log10(M_grid[like5sigma_x_idx1]), np.log10(M_grid[like5sigma_x_idx2])))
	print('5 sigma: {:.3f}, {:.2f}, {:.2f}'.format(like5sigma, G_grid[like5sigma_y_idx1], G_grid[like5sigma_y_idx2]))
	
	###>>>
	if output_flag == 1:		
		# Open and output Schechter fit results to file
		f = open('./GridLike/' + outputfile + '_out.txt', 'w')
		f.write('Schechter Results (Beta, M*)\n')
		f.write('{:.2f}, {:.2f}\n'.format(res_loglikeschechter.x[1], np.log10(res_loglikeschechter.x[0])))
		f.write('1 sigma: {:.2f}, {:.2f}\n'.format(np.log10(M_grid[like1sigma_x_idx1]), np.log10(M_grid[like1sigma_x_idx2])))
		f.write('1 sigma: {:.2f}, {:.2f}\n'.format(G_grid[like1sigma_y_idx1], G_grid[like1sigma_y_idx2]))
	###>>>>

	# Results for power law fit
	print('Errors: Power Law')
	###
	like1sigmasimplepowerlaw = res_loglikesimplepowerlaw.fun + 0.50
	like2sigmasimplepowerlaw = res_loglikesimplepowerlaw.fun + 2.00
	simplepowerlaw_grid = np.zeros(len(G_grid))
	for i in range(0, len(G_grid)):
		testval = loglikesimplepowerlaw(G_grid[i])
		if math.isinf(testval):
			simplepowerlaw_grid[i] = 1E20
		elif testval <= 1E-5:
			simplepowerlaw_grid[i] = 1E20
		else:
			simplepowerlaw_grid[i] = testval
	like_G_idx, like_G_val = find_nearest(G_grid, res_loglikesimplepowerlaw.x[0])
	###
	like1sigma_idx1, like1sigma_idx2 = find_nearest2guided(simplepowerlaw_grid, like1sigmasimplepowerlaw, like_G_idx)
	print('1 sigma: {:.3f}, {:.2f}, {:.2f}'.format(like1sigmasimplepowerlaw, G_grid[like1sigma_idx1], G_grid[like1sigma_idx2]))
	###
	like2sigma_idx1, like2sigma_idx2 = find_nearest2guided(simplepowerlaw_grid, like2sigmasimplepowerlaw, like_G_idx)
	print('2 sigma: {:.3f}, {:.2f}, {:.2f}'.format(like2sigmasimplepowerlaw, G_grid[like2sigma_idx1], G_grid[like2sigma_idx2]))
	
	###>>>>
	if output_flag == 1:
		# Output power law fit results to file and close file
		f.write('Fitting Results (Beta)\n')
		f.write('{:.2f}\n'.format(np.log10(res_loglikesimplepowerlaw.x[0])))
		f.write('1 sigma: {:.2f}, {:.2f}\n'.format(G_grid[like1sigma_idx1], G_grid[like1sigma_idx2]))
	###>>>>

	# Results for schechter function fit with slope of 2
	print('Errors: Schechter with Beta = 2')
	###
	like1sigmaschechterwithslope2 = res_loglikeschechterwithslope2.fun + 0.50
	like2sigmaschechterwithslope2 = res_loglikeschechterwithslope2.fun + 2.00
	schechterwithslope2_grid = np.zeros(len(M_grid))
	for i in range(0, len(G_grid)):
		testval = loglikeschechterwithslope2(M_grid[i])
		if math.isinf(testval):
			schechterwithslope2_grid[i] = 1E20
		elif testval <= 1E-5:
			schechterwithslope2_grid[i] = 1E20
		else:
			schechterwithslope2_grid[i] = testval
	like_M_idx, like_M_val = find_nearest(M_grid, res_loglikeschechterwithslope2.x[0])
	###
	like1sigma_idx1, like1sigma_idx2 = find_nearest2guided(schechterwithslope2_grid, like1sigmaschechterwithslope2, like_M_idx)
	print('1 sigma: {:.3f}, {:.2f}, {:.2f}'.format(like1sigmaschechterwithslope2, np.log10(M_grid[like1sigma_idx1]), np.log10(M_grid[like1sigma_idx2])))
	###
	like2sigma_idx1, like2sigma_idx2 = find_nearest2guided(schechterwithslope2_grid, like2sigmaschechterwithslope2, like_M_idx)
	print('2 sigma: {:.3f}, {:.2f}, {:.2f}'.format(like2sigmaschechterwithslope2, np.log10(M_grid[like2sigma_idx1]), np.log10(M_grid[like2sigma_idx2])))
	
	###>>>>
	if output_flag == 1:
		# Output power law fit results to file and close file
		f.write('Fitting Results (M*)\n')
		f.write('{:.2f}\n'.format(np.log10(res_loglikeschechterwithslope2.x[0])))
		f.write('1 sigma: {:.2f}, {:.2f}\n'.format(np.log10(M_grid[like1sigma_x_idx1]), np.log10(M_grid[like1sigma_x_idx2])))
		f.close()
	###>>>>

	######################################################

	## Make contour plots and label
	plotaxes.contourf(np.log10(M_grid), G_grid, Value_grid, [likemin, like1sigma, like2sigma, like3sigma], colors = ['black', 'darkgrey', 'lightgrey', 'gainsboro'])
	plotaxes.plot(np.log10(np.nanmax(array4)), 0.5, 'g^', markersize = 30)
	
	#
	locval = [3.75, 2.75]
	if loc != 0:
	 	locval = loc
	elif agename in ['GMC', 'GMC_P']:
		locval = [6.50, 1.00]
	elif agename in ['XB_comb']:
		locval = [4.50, 2.50]
	print('Final locval value: {}'.format(locval))

	# flag_simple = 1
	galnameout = galnameoutfun(galname)

	# Set legend labels
	if agename in ['XB_comb'] or flag_simple == 1:
		plotaxes.text(locval[0], locval[1], galnameout, fontsize = 32, verticalalignment = 'top', bbox = boxprops)
	else:
		plotaxes.text(locval[0], locval[1], galnameout + '\n     N = {}'.format(len(array4)), fontsize = 32, verticalalignment = 'top', bbox = boxprops)
	
	# Plot dotted lines
	if flag_success == 1:
		plotaxes.plot([np.log10(res_loglikeschechter.x[0]), np.log10(res_loglikeschechter.x[0])], [-1, 5], 'k--')
		plotaxes.plot([0.0, 10.0], [res_loglikeschechter.x[1], res_loglikeschechter.x[1]], 'k--')
	else:
		plotaxes.plot([np.log10(likegridM), np.log10(likegridM)], [-1, 5], 'k--')
		plotaxes.plot([0.0, 10.0], [likegridG, likegridG], 'k--')

	## Output important information and close output file
	likelihoodplot_out = [np.log10(M_grid[like2sigma_x_idx1]), np.log10(M_grid[like2sigma_x_idx2]), G_grid[like2sigma_y_idx1], G_grid[like2sigma_y_idx2]]

	return likelihoodplot_out

def plotaxes_agemass(gal_array, galname, complimits, plotaxes, flag, clustermarker = 'ro'):
	
	'''
	Function: Make one age-mass plot at chosen location
	'''

	print('Plotting {} - Age-Mass (plotaxes)'.format(galname))

	plotaxes.plot(gal_array[:,6], gal_array[:,4], clustermarker, markersize = 4, alpha = 0.4)
	
	# Plot
	if flag == 1:
		plotaxes.plot([np.power(10, 5.8), np.power(10, 7)], [complimits[0], complimits[0]], 'k--')
		plotaxes.plot([np.power(10, 7), np.power(10, 8)], [complimits[1], complimits[1]], 'k--')
		plotaxes.plot([np.power(10, 8), 4 * np.power(10, 8)], [complimits[2], complimits[2]], 'k--')
		plotaxes.plot([np.power(10, 7), np.power(10, 7)], [complimits[0], 1.5 * np.nanmax(gal_array[:,4])], 'k--')
		plotaxes.plot([np.power(10, 8), np.power(10, 8)], [complimits[1], 1.5 * np.nanmax(gal_array[:,4])], 'k--')
		plotaxes.plot([4 * np.power(10, 8), 4 * np.power(10, 8)], [complimits[2], 1.5 * np.nanmax(gal_array[:,4])], 'k--')
	# Plot
	elif flag == 2:
		plotaxes.plot([np.power(10, 5.8), 2 * np.power(10, 8)], [complimits[2], complimits[2]], 'k--')
		plotaxes.plot([2 * np.power(10, 8), 2 * np.power(10, 8)], [complimits[2], 1.5 * np.nanmax(gal_array[:,4])], 'k--')
	# Plot only 100 - 400 Myr
	elif flag == 3:
		plotaxes.plot([np.power(10, 8), np.power(10, 8)], [complimits[2], 1.5 * np.nanmax(gal_array[:,4])], 'k--')
		plotaxes.plot([np.power(10, 8), 4 * np.power(10, 8)], [complimits[2], complimits[2]], 'k--')
		plotaxes.plot([4 * np.power(10, 8), 4 * np.power(10, 8)], [complimits[2], 1.5 * np.nanmax(gal_array[:,4])], 'k--')

	# plotaxes.legend(title = galname, loc = 'upper left', fontsize = 80)
	plotaxes.annotate(galname, xy = (0.1, 0.80), xycoords='axes fraction', fontsize = 20)
	return 0

def plotaxes_histogram(gal_array, galname, complimits, plotaxes, ageflag, outflag):

	'''
	Function: Make one histogram plot at chosen location
	'''

	print('Starting {} - Histogram (plotaxes)'.format(galname))

	mass_bins_log = np.power(10, np.linspace(2, 9, num = 29))
	array_plot = np.power(10, np.linspace(np.log10(np.min(mass_bins_log)), np.log10(np.max(mass_bins_log))))

	gal_array_masslimit, age_label, gal_array_out = agecuts_outputarrays(gal_array, galname, complimits, ageflag)

	if ageflag == 1:
		if len(gal_array_masslimit) > 0:
			n_age1, bins_age1, bins_width_age1, bins_centre_age1, n_fit_age1, bins_fit_age1, n_dM_age1, n_fit_dM_age1, n_dlogM_age1, n_fit_dlogM_age1, ncum_age1, ncum_fit_age1, n_fit_age1_err, n_fit_dM_age1_err, n_fit_dlogM_age1_err = makearrayhist(gal_array_masslimit, mass_bins_log, complimits[0])
			histogramplot(galname, age_label, bins_age1, n_dM_age1, bins_fit_age1, n_fit_dM_age1, n_fit_dM_age1_err, complimits[0], array_plot, plotaxes, len(gal_array_masslimit), outflag)
	elif ageflag == 2:
		if len(gal_array_masslimit) > 0:
			n_age2, bins_age2, bins_width_age2, bins_centre_age2, n_fit_age2, bins_fit_age2, n_dM_age2, n_fit_dM_age2, n_dlogM_age2, n_fit_dlogM_age2, ncum_age2, ncum_fit_age2, n_fit_age2_err, n_fit_dM_age2_err, n_fit_dlogM_age2_err = makearrayhist(gal_array_masslimit, mass_bins_log, complimits[1])
			histogramplot(galname, age_label, bins_age2, n_dM_age2, bins_fit_age2, n_fit_dM_age2, n_fit_dM_age2_err, complimits[1], array_plot, plotaxes, len(gal_array_masslimit), outflag)
	elif ageflag == 3:
		if len(gal_array_masslimit) > 0:
			n_age3, bins_age3, bins_width_age3, bins_centre_age3, n_fit_age3, bins_fit_age3, n_dM_age3, n_fit_dM_age3, n_dlogM_age3, n_fit_dlogM_age3, ncum_age3, ncum_fit_age3, n_fit_age3_err, n_fit_dM_age3_err, n_fit_dlogM_age3_err = makearrayhist(gal_array_masslimit, mass_bins_log, complimits[2])
			histogramplot(galname, age_label, bins_age3, n_dM_age3, bins_fit_age3, n_fit_dM_age3, n_fit_dM_age3_err, complimits[2], array_plot, plotaxes, len(gal_array_masslimit), outflag)
	elif ageflag == 5:
		if len(gal_array_masslimit) > 0:
			n_age5, bins_age5, bins_width_age5, bins_centre_age5, n_fit_age5, bins_fit_age5, n_dM_age5, n_fit_dM_age5, n_dlogM_age5, n_fit_dlogM_age5, ncum_age5, ncum_fit_age5, n_fit_age5_err, n_fit_dM_age5_err, n_fit_dlogM_age5_err = makearrayhist(gal_array_masslimit, mass_bins_log, complimits[2])
			histogramplot(galname, age_label, bins_age5, n_dM_age5, bins_fit_age5, n_fit_dM_age5, n_fit_dM_age5_err, complimits[2], array_plot, plotaxes, len(gal_array_masslimit), outflag)
	elif ageflag == 0:
		n, bins, bins_width, bins_centre, n_fit, bins_fit, n_dM, n_fit_dM, n_dlogM, n_fit_dlogM, ncum, ncum_fit, n_fit_err, n_fit_dM_err, n_fit_dlogM_err = makearrayhist(gal_array_masslimit, mass_bins_log, np.nanmax(complimits))
		histogramplot(galname, age_label, bins, n_dM, bins_fit, n_fit_dM, n_fit_dM_err, np.nanmax(complimits), array_plot, plotaxes, len(gal_array_masslimit), outflag)
	elif ageflag == -1:
		n, bins, bins_width, bins_centre, n_fit, bins_fit, n_dM, n_fit_dM, n_dlogM, n_fit_dlogM, ncum, ncum_fit, n_fit_err, n_fit_dM_err, n_fit_dlogM_err = makearrayhist(gal_array_masslimit, mass_bins_log, np.nanmax(complimits))
		histogramplot(galname, age_label, bins, n_dM, bins_fit, n_fit_dM, n_fit_dM_err, np.nanmax(complimits), array_plot, plotaxes, len(gal_array_masslimit), outflag)
	else:
		n, bins, bins_width, bins_centre, n_fit, bins_fit, n_dM, n_fit_dM, n_dlogM, n_fit_dlogM, ncum, ncum_fit, n_fit_err, n_fit_dM_err, n_fit_dlogM_err = makearrayhist(gal_array_masslimit, mass_bins_log, np.nanmax(complimits))
		histogramplot(galname, age_label, bins, n_dM, bins_fit, n_fit_dM, n_fit_dM_err, np.nanmax(complimits), array_plot, plotaxes, len(gal_array_masslimit), outflag)

def plotaxes_equalhistogram(gal_array, galname, complimits, plotaxes, ageflag, outflag, numgal_bin_in = 5, outputbinstofile = False):

	'''
	Function: Make one histogram plot at chosen location
	'''

	print('Starting {} - Histogram (plotaxes)'.format(galname))

	mass_bins_log = np.power(10, np.linspace(2, 9, num = 29))
	array_plot = np.power(10, np.linspace(np.log10(np.min(mass_bins_log)), np.log10(np.max(mass_bins_log))))

	gal_array_masslimit, age_label, gal_array_out = agecuts_outputarrays(gal_array, galname, complimits, ageflag)

	# Could be renamed to add 'equal' terminology
	if ageflag == 1:
		if len(gal_array_masslimit) > 0:
			n_age1, bins_age1, bins_width_age1, bins_centre_age1, n_fit_age1, bins_fit_age1, n_dM_age1, n_fit_dM_age1, n_dlogM_age1, n_fit_dlogM_age1, ncum_age1, ncum_fit_age1, n_fit_age1_err, n_fit_dM_age1_err, n_fit_dlogM_age1_err = makearrayhistequal(gal_array_masslimit, complimits[0], -1, numgal_bin_in = numgal_bin_in)
			histogramplot(galname, age_label, bins_age1, n_dM_age1, bins_fit_age1, n_fit_dM_age1, n_fit_dM_age1_err, complimits[0], array_plot, plotaxes, len(gal_array_masslimit), outflag)
	elif ageflag == 2:
		if len(gal_array_masslimit) > 0:
			n_age2, bins_age2, bins_width_age2, bins_centre_age2, n_fit_age2, bins_fit_age2, n_dM_age2, n_fit_dM_age2, n_dlogM_age2, n_fit_dlogM_age2, ncum_age2, ncum_fit_age2, n_fit_age2_err, n_fit_dM_age2_err, n_fit_dlogM_age2_err = makearrayhistequal(gal_array_masslimit, complimits[1], -1, numgal_bin_in = numgal_bin_in)
			histogramplot(galname, age_label, bins_age2, n_dM_age2, bins_fit_age2, n_fit_dM_age2, n_fit_dM_age2_err, complimits[1], array_plot, plotaxes, len(gal_array_masslimit), outflag)
	elif ageflag == 3:
		if len(gal_array_masslimit) > 0:
			n_age3, bins_age3, bins_width_age3, bins_centre_age3, n_fit_age3, bins_fit_age3, n_dM_age3, n_fit_dM_age3, n_dlogM_age3, n_fit_dlogM_age3, ncum_age3, ncum_fit_age3, n_fit_age3_err, n_fit_dM_age3_err, n_fit_dlogM_age3_err = makearrayhistequal(gal_array_masslimit, complimits[2], -1, numgal_bin_in = numgal_bin_in)
			histogramplot(galname, age_label, bins_age3, n_dM_age3, bins_fit_age3, n_fit_dM_age3, n_fit_dM_age3_err, complimits[2], array_plot, plotaxes, len(gal_array_masslimit), outflag)
			# Print Bins to File
			if outputbinstofile == True:
				f1 = open('./Logs/ZOutput_' + galname +  '_Bins.txt', 'w')
				for i in range(0, len(n_fit_dM_age3)):
					print('{}, {}, {}'.format(bins_fit_age3[i], n_fit_dM_age3[i], n_fit_dM_age3_err[i]))
				f1.close()
	elif ageflag == 5:
		if len(gal_array_masslimit) > 0:
			n_age5, bins_age5, bins_width_age5, bins_centre_age5, n_fit_age5, bins_fit_age5, n_dM_age5, n_fit_dM_age5, n_dlogM_age5, n_fit_dlogM_age5, ncum_age5, ncum_fit_age5, n_fit_age5_err, n_fit_dM_age5_err, n_fit_dlogM_age5_err = makearrayhistequal(gal_array_masslimit, complimits[2], -1, numgal_bin_in = numgal_bin_in)
			histogramplot(galname, age_label, bins_age5, n_dM_age5, bins_fit_age5, n_fit_dM_age5, n_fit_dM_age5_err, complimits[2], array_plot, plotaxes, len(gal_array_masslimit), outflag)
	elif ageflag == 0:
		n, bins, bins_width, bins_centre, n_fit, bins_fit, n_dM, n_fit_dM, n_dlogM, n_fit_dlogM, ncum, ncum_fit, n_fit_err, n_fit_dM_err, n_fit_dlogM_err = makearrayhistequal(gal_array_masslimit, np.nanmax(complimits), -1, numgal_bin_in = numgal_bin_in)
		histogramplot(galname, age_label, bins, n_dM, bins_fit, n_fit_dM, n_fit_dM_err, np.nanmax(complimits), array_plot, plotaxes, len(gal_array_masslimit), outflag)
	elif ageflag == -1:
		n, bins, bins_width, bins_centre, n_fit, bins_fit, n_dM, n_fit_dM, n_dlogM, n_fit_dlogM, ncum, ncum_fit, n_fit_err, n_fit_dM_err, n_fit_dlogM_err = makearrayhistequal(gal_array_masslimit, np.nanmax(complimits), -1, numgal_bin_in = numgal_bin_in)
		histogramplot(galname, age_label, bins, n_dM, bins_fit, n_fit_dM, n_fit_dM_err, np.nanmax(complimits), array_plot, plotaxes, len(gal_array_masslimit), outflag)
	else:
		n, bins, bins_width, bins_centre, n_fit, bins_fit, n_dM, n_fit_dM, n_dlogM, n_fit_dlogM, ncum, ncum_fit, n_fit_err, n_fit_dM_err, n_fit_dlogM_err = makearrayhistequal(gal_array_masslimit, np.nanmax(complimits), -1, numgal_bin_in = numgal_bin_in)
		histogramplot(galname, age_label, bins, n_dM, bins_fit, n_fit_dM, n_fit_dM_err, np.nanmax(complimits), array_plot, plotaxes, len(gal_array_masslimit), outflag)

def histogramplot(galname, age_label, bins, n_dM, bins_fit, n_fit_dM, n_fit_dM_err, complimits, array_plot, plotaxes, len_sortedarray, outflag):

	'''
	Function: Plot histogram (subfunction for plotaxes_histogram)
	'''

	plotaxes.step(bins, np.append(n_dM[0], n_dM), color = 'k', alpha = 0.5)
	plotaxes.errorbar(bins_fit, n_fit_dM, marker = '^', linestyle = 'None', markerfacecolor = 'g', yerr = n_fit_dM_err)
	popt_truncatedpowerlaw, pcov_truncatedpowerlaw, popt_simplepowerlaw, pcov_simplepowerlaw, powerlawchisq, popt_schechter, pcov_schechter, schechterchisq = curve_fit3(bins_fit, n_fit_dM, [1E2, 1E8], np.nanmax(complimits), 1, 0, n_fit_dM_err)
	if outflag == 1:
		plotaxes.plot(array_plot, schechter(array_plot, *popt_schechter), 'b-.', label = r'Sch. - log M$_0$ = {:.2f}, $\gamma$ = {:.2f}'.format(np.log10(popt_schechter[1]), popt_schechter[2]) + '\n' + r'			  ($\chi^2_r$ = {:.2f})'.format(schechterchisq))
		plotaxes.plot(array_plot, simplepowerlaw(array_plot, *popt_simplepowerlaw), 'r--', label = r'PL - log M$_0$ = {:.2f}, $\gamma$ = {:.2f}'.format(np.log10(popt_simplepowerlaw[0]), popt_simplepowerlaw[1]) +  '\n' + r'			  ($\chi^2_r$ = {:.2f})'.format(powerlawchisq))
	elif outflag == 2:
		plotaxes.plot(array_plot, simplepowerlaw(array_plot, *popt_simplepowerlaw), 'r--', label = r'$-\beta$ = {:.2f} $\pm$ {:.2f}'.format(-popt_simplepowerlaw[1], np.sqrt(abs(pcov_simplepowerlaw[1][1]))))
	else:
		plotaxes.plot(array_plot, simplepowerlaw(array_plot, *popt_simplepowerlaw), 'r--', label = r'PL - $\gamma$ = {:.2f}'.format(popt_simplepowerlaw[1]))
	plotaxes.plot([np.nanmax(complimits), np.nanmax(complimits)], [1E-10, 1E10], 'k--')
	if outflag != 2:
		plotaxes.legend(loc = 'upper right', title = galname + age_label + '\n' + '            N = {}'.format(len_sortedarray))
	else:
		plotaxes.legend(loc = 'upper right', title = galname)

	return 0

def plotaxes_mspecfit(gal_array, galname, complimits, plotaxes, ageflag, errorflag, sigma, plotflag = 1, flagcol = 1):

	'''
	Function: Make One MSpecFit Plot at Chosen Location
	'''

	print('Starting {} - MSPECFIT (plotaxes) with Error Flag {}'.format(galname, errorflag))

	gal_array_masslimit, age_label, gal_array_out = agecuts_outputarrays(gal_array, galname, complimits, ageflag)

	if flagcol == 2:
		mspecfitplot(gal_array[:,9], gal_array[:,9], 20, plotaxes, galname, '', errorflag, flagcol = flagcol, plotflag = plotflag)
	elif ageflag == 1:
		if len(gal_array_masslimit) > 0:
			mspecfitplot(gal_array_out[:,4], gal_array_masslimit[:,4], complimits[0], plotaxes, galname, '_A1', errorflag, plotflag = plotflag)
	elif ageflag == 2:
		if len(gal_array_masslimit) > 0:
			mspecfitplot(gal_array_out[:,4], gal_array_masslimit[:,4], complimits[1], plotaxes, galname, '_A2', errorflag, plotflag = plotflag)
	elif ageflag == 3:
		if len(gal_array_masslimit) > 0:
			mspecfitplot(gal_array_out[:,4], gal_array_masslimit[:,4], complimits[2], plotaxes, galname, '_A3', errorflag, plotflag = plotflag)
	elif ageflag == 5:
		if len(gal_array_masslimit) > 0:
			mspecfitplot(gal_array_out[:,4], gal_array_masslimit[:,4], complimits[2], plotaxes, galname, '_A5', errorflag, plotflag = plotflag)
	elif ageflag == 0:
		if len(gal_array_masslimit) > 0:
			mspecfitplot(gal_array_out[:,4], gal_array_masslimit[:,4], np.nanmax(complimits), plotaxes, galname, '_AL', errorflag, plotflag = plotflag)
	elif ageflag == -1:
		if len(gal_array_masslimit) > 0:
			mspecfitplot(gal_array_out[:,4], gal_array_masslimit[:,4], np.nanmax(complimits), plotaxes, galname, '', errorflag, plotflag = plotflag)
	else:
		if len(gal_array_masslimit) > 0:
			mspecfitplot(gal_array_out[:,4], gal_array_masslimit[:,4], np.nanmax(complimits), plotaxes, galname, '', errorflag, plotflag = plotflag)

def mspecfitplot(array1, array1_masslimit, complimits_val, plotaxes, galname, agename, errorflag, plotflag = 1, flagcol = 1):

	'''
	Function: Plot MSpecFit (subfunction for plotaxes_mmspecfit)
	'''

	if agename == '_A1':
		ageout = r'$\tau \leq$ 10 Myr'
	elif agename == '_A2':
		ageout = r'10 < $\tau \leq$ 100 Myr'
	elif agename == '_A3':
		ageout = r'100 < $\tau \leq$ 400 Myr'
	elif agename == '_A3d':
		ageout = r'$\tau \leq$ 200 Myr'
	elif agename == '_A5':
		ageout = r'1 <= $\tau \leq$ 200 Myr'
	elif agename == '_A7':
		ageout = r'$\tau \leq$ 400 Myr'
	elif agename == '_A7a':
		ageout = r'30 <= $\tau \leq$ 400 Myr'
	elif agename == '_A7b':
		ageout = r'50 <= $\tau \leq$ 400 Myr'
	elif agename == '_A7c':
		ageout = r'80 <= $\tau \leq$ 400 Myr'
	elif agename == '_A8':
		ageout = r'100 <= $\tau \leq$ 200 Myr'
	else:
		ageout = agename

	print('>>> {}: Stated Completeness Limit'.format(galname))

	boxprops = dict(boxstyle = 'round', facecolor = 'white') # , alpha = 0.5

	if flagcol == 1:

		# Sort data array, create x and y arrays
		sorted_data = np.sort(array1)
		xdata = np.concatenate([sorted_data[::-1]])
		ydata = np.arange(sorted_data.size) + 1

	else:

		# Sort data array, create x and y arrays
		sorted_data = np.sort(array1)
		xdata = np.concatenate([sorted_data[::-1]])
		ydata = np.arange(sorted_data.size) + 1
		
	# Filter array to those > complimit, output results
	filt_array = np.where(sorted_data >= complimits_val)
	data_lim = sorted_data[filt_array]
	sorted_data_lim = np.sort(data_lim)

	# Output results
	xdata_lim = np.concatenate([sorted_data_lim[::-1]])
	ydata_lim = np.arange(sorted_data_lim.size) + 1
	xdata_lim_float = xdata_lim.astype(float)
	ydata_lim_float = ydata_lim.astype(float)
	
	print('[{}, {}]'.format(np.nanmin(xdata_lim_float), np.nanmax(xdata_lim_float)))
	print('[{}, {}]'.format(np.nanmax(ydata_lim_float), np.nanmin(ydata_lim_float)))
	
	if galname == 'All':
		plotaxes.step(xdata, ydata, 'k')
	elif galname == 'HXMB':
		plotaxes.step(xdata, ydata, 'b')
	elif galname == 'IMXB':
		plotaxes.step(xdata, ydata, 'g')
	elif galname == 'LMXB':
		plotaxes.step(xdata, ydata, 'r')
	else:
		plotaxes.step(xdata, ydata)

	plotaxes.plot([complimits_val, complimits_val], [1E-10, 1E10], 'k--')

	prevdir = os.getcwd()
	from idlpy import IDL

	if flagcol == 2:
		num_iter = 1
	else:
		num_iter = 100

	# Set outputs based on flags
	# 1: PL - Error + X Iter, TPL - Error + X Iter, Print Results
	# 0: PL - No Error + X Iter, TPL - No Error + X Iter
	# -1: PL - No Error + X Iter, TPL - DO NOT RUN
	if errorflag == 1:
		print('IDL - Error + {} iterations'.format(num_iter))
		fit_pl = IDL.mspecfit(np.array(array1_masslimit, dtype = np.float32), np.power(10, errorflag)*np.array(array1_masslimit, dtype = np.float32), notrunc = 'notrunc', bootiter = num_iter)
	elif errorflag == 0:
		print('IDL - No Error + {} iterations'.format(num_iter))
		fit_pl = IDL.mspecfit(np.array(array1_masslimit, dtype = np.float32), 1E-6*np.ones(len(array1_masslimit), dtype = np.float32), notrunc = 'notrunc', bootiter = num_iter)
	elif errorflag == -1:
		print('IDL - No Error + No iterations')
		fit_pl = IDL.mspecfit(np.array(array1_masslimit, dtype = np.float32), 1E-6*np.ones(len(array1_masslimit), dtype = np.float32), notrunc = 'notrunc', bootiter = num_iter)

	# Output
	fit_pl_out = [fit_pl[1], fit_pl[2] + 1]
	error_pl_out = [fit_pl[4], fit_pl[5]]
	print('PL:', fit_pl)
	print('Fit: N = {:.2f} +/- {:.2f}, Log(X) = {:.2f} +/- {:.2f}, Slope = {:.2f} +/- {:.2f}'.format(fit_pl[0], fit_pl[3], np.log10(fit_pl[1]), 0.434 * (fit_pl[4] / fit_pl[1]), fit_pl[2], fit_pl[5]))

	if plotflag != 3:
		# Set output based on flags
		if errorflag == 1:
			fit = IDL.mspecfit(np.array(array1_masslimit, dtype = np.float32), np.power(10, errorflag)*np.array(array1_masslimit, dtype = np.float32), bootiter = num_iter)
		elif errorflag == 0:
			fit = IDL.mspecfit(np.array(array1_masslimit, dtype = np.float32), 1E-6*np.ones(len(array1_masslimit), dtype = np.float32), bootiter = num_iter)
		elif errorflag == -1:
			# Do not run
			fit_pl = IDL.mspecfit(np.array(array1_masslimit, dtype = np.float32), 1E-6*np.ones(len(array1_masslimit), dtype = np.float32), bootiter = num_iter)

		fit_out = [fit[0], fit[1], fit[2] + 1]
		error_out = [fit[3], fit[4], fit[5]]
		print('TPL:', fit)
		print('Fit: N = {:.2f} +/- {:.2f}, Log(X) = {:.2f} +/- {:.2f}, Slope = {:.2f} +/- {:.2f}'.format(fit[0], fit[3], np.log10(fit[1]), 0.434 * (fit[4] / fit[1]), fit[2], fit[5]))
	
	os.chdir(prevdir)

	mass_bins_log_plot = np.power(10, np.linspace(2, 8))
	if plotflag == 3:
		plotaxes.plot(mass_bins_log_plot, simplepowerlaw(mass_bins_log_plot, *fit_pl_out), 'k-.', label = r'{} - $-\beta$ = {:.2f} $\pm$ {:.2f}'.format(galname, -fit_pl[2], fit_pl[5]))
	elif plotflag == 2:
		plotaxes.plot(mass_bins_log_plot, simplepowerlaw(mass_bins_log_plot, *fit_pl_out), 'k-.', label = r'PL ($-\beta$ = {:.2f} $\pm$ {:.2f})'.format(-fit_pl[2], fit_pl[5]))
		plotaxes.plot(mass_bins_log_plot, truncatedpowerlaw(mass_bins_log_plot, *fit_out), 'k:', label = r'TPL (N$_c$ = {:.2f} $\pm$ {:.2f})'.format(fit[0], fit[3]))
	elif errorflag > 0:
		plotaxes.plot(mass_bins_log_plot, simplepowerlaw(mass_bins_log_plot, *fit_pl_out), 'k-.', label = r'PL ($-\beta$ = {:.1f} $\pm$ {:.1f})'.format(-fit_pl[2], fit_pl[5]))
		plotaxes.plot(mass_bins_log_plot, truncatedpowerlaw(mass_bins_log_plot, *fit_out), 'k:', label = r'TPL (N$_c$ = {:.1f} $\pm$ {:.1f})'.format(fit[0], fit[3]))
	else:
		# plotaxes.plot(mass_bins_log_plot, simplepowerlaw(mass_bins_log_plot, *fit_pl_out), 'k-.', label = r'$\beta$ = -{:.1f} $\pm$ {:.1f}'.format(-fit_pl[2], fit_pl[5]))
		plotaxes.plot(mass_bins_log_plot, simplepowerlaw(mass_bins_log_plot, *fit_pl_out), 'k-.', label = r'PL ($-\beta$ = {:.1f} $\pm$ {:.1f})'.format(-fit_pl[2], fit_pl[5]))
		plotaxes.plot(mass_bins_log_plot, truncatedpowerlaw(mass_bins_log_plot, *fit_out), 'k:', label = r'TPL (N$_c$ = {:.1f} $\pm$ {:.1f})'.format(fit[0], fit[3]))
		
	if plotflag != 3:
		plotaxes.legend(loc = 'upper right', title = galname, framealpha = 1.0)
	else:
		plotaxes.legend(loc = 'upper right', framealpha = 1.0)

	return xdata_lim_float, ydata_lim_float

def plotaxes_twohistogram(gal_array, galname, complimits, complimits_gmc, gal_array_gmc, plotaxes, ageflag, outflag):

	'''
	Function: Plot the cluster and GMC
	'''

	print('Starting {}'.format(galname))

	mass_bins_log = np.power(10, np.linspace(2, 9, num = 29))
	array_plot = np.power(10, np.linspace(np.log10(np.min(mass_bins_log)), np.log10(np.max(mass_bins_log)) + 1))

	def histogram_twoplot(galname, age_label, bins, n_dM, bins_fit, n_fit_dM, n_fit_dM_err, complimits, bins_gmc, n_dM_gmc, bins_fit_gmc, n_fit_dM_gmc, n_fit_dM_err_gmc, complimits_gmc, array_plot, plotaxes):

		plotaxes.errorbar(bins_fit, n_fit_dM, marker = '^', linestyle = 'None', markerfacecolor = 'g', yerr = n_fit_dM_err, markersize = 20, label = r'Clusters')
		plotaxes.errorbar(bins_fit_gmc, n_fit_dM_gmc, marker = 's', linestyle = 'None', markerfacecolor = 'r', yerr = n_fit_dM_err_gmc, markersize = 20, label = r'GMCs')
		popt_truncatedpowerlaw, pcov_truncatedpowerlaw, popt_simplepowerlaw, pcov_simplepowerlaw, powerlawchisq, popt_schechter, pcov_schechter, schechterchisq = curve_fit3(bins_fit, n_fit_dM, [1E2, 1E8], np.nanmax(complimits), 1, 0, n_fit_dM_err)
		popt_truncatedpowerlaw_gmc, pcov_truncatedpowerlaw_gmc, popt_simplepowerlaw_gmc, pcov_simplepowerlaw_gmc, powerlawchisq_gmc, popt_schechter_gmc, pcov_schechter_gmc, schechterchisq_gmc = curve_fit3(bins_fit_gmc, n_fit_dM_gmc, [1E2, 1E8], np.nanmax(complimits_gmc), 1, 0, n_fit_dM_err_gmc)
		if outflag == 1:
			plotaxes.plot(array_plot, schechter(array_plot, *popt_schechter), 'b-.', label = r'Sch. - log M$_0$ = {:.2f}, $\gamma$ = {:.2f}'.format(np.log10(popt_schechter[1]), popt_schechter[2]) + '\n' + r'           ($\chi^2_r$ = {:.2f})'.format(schechterchisq))
			plotaxes.plot(array_plot, simplepowerlaw(array_plot, *popt_simplepowerlaw), 'r--', label = r'PL - log M$_0$ = {:.2f}, $\gamma$ = {:.2f}'.format(np.log10(popt_simplepowerlaw[0]), popt_simplepowerlaw[1]) +  '\n' + r'            ($\chi^2_r$ = {:.2f})'.format(powerlawchisq))
		else:
			a = 1

		legend = plotaxes.legend(loc = 'upper right', title = galname, fontsize = 30)
		plt.setp(legend.get_title(), fontsize = 30)

		return 0

	percentile_gal_value = np.nanmax(gal_array_gmc[:,4]) * 0.05
	print('GMC - 5 percent of maximum mass: {:.3e} or log M = {:.2f}'.format(percentile_gal_value, np.log10(percentile_gal_value)))
	gal_array_gmc_masslimit_percentile = gal_array_gmc[gal_array_gmc[:,4] > percentile_gal_value]
	gal_array_gmc_masslimit_survey = gal_array_gmc[gal_array_gmc[:,4] > complimits_gmc]
	n, bins, bins_width, bins_centre, n_fit, bins_fit, n_dM, n_fit_dM, n_dlogM, n_fit_dlogM, ncum, ncum_fit, n_fit_err, n_fit_dM_err, n_fit_dlogM_err = makearrayhist(gal_array_gmc, mass_bins_log, complimits_gmc)
	nnew, binsnew, binsnew_width, binsnew_centre, nnew_fit, binsnew_fit, nnew_dM, nnew_fit_dM, nnew_dlogM, nnew_fit_dlogM, nnewcum, nnewcum_fit, nnew_fit_err, nnew_fit_dM_err, nnew_fit_dlogM_err = makearrayhist(gal_array_gmc, mass_bins_log, percentile_gal_value)

	if ageflag == 1:
		print(galname)
		print('1 = (0, 10] Myr + input completeness limit A1')
		gal_array_age1 = gal_array[gal_array[:,6] <= 10.01*1E6]
		gal_array_age1_masslimit = gal_array_age1[gal_array_age1[:,4] > complimits[0]]
		age1_label = r' ($\tau \leq$ 10 Myr)'
		if len(gal_array_age1_masslimit) > 0:
			n_age1, bins_age1, bins_width_age1, bins_centre_age1, n_fit_age1, bins_fit_age1, n_dM_age1, n_fit_dM_age1, n_dlogM_age1, n_fit_dlogM_age1, ncum_age1, ncum_fit_age1, n_fit_age1_err, n_fit_dM_age1_err, n_fit_dlogM_age1_err = makearrayhist(gal_array_age1_masslimit, mass_bins_log, complimits[0])
			histogram_twoplot(galname, age1_label, bins_age1, n_dM_age1, bins_fit_age1, n_fit_dM_age1, n_fit_dM_age1_err, complimits[0], binsnew, nnew_dM, binsnew_fit, nnew_fit_dM, nnew_fit_dM_err, percentile_gal_value, array_plot, plotaxes)

	if ageflag == 2:
		print('2 = (10, 100] Myr + input completeness limit A2')
		gal_array_age2_tmp = gal_array[gal_array[:,6] > 10.01*1E6]
		gal_array_age2 = gal_array_age2_tmp[gal_array_age2_tmp[:,6] <= 100.01*1E6]
		gal_array_age2_masslimit = gal_array_age2[gal_array_age2[:,4] > complimits[1]]
		age2_label = r' (10 < $\tau \leq$ 100 Myr)'
		if len(gal_array_age2_masslimit) > 0:
			n_age2, bins_age2, bins_width_age2, bins_centre_age2, n_fit_age2, bins_fit_age2, n_dM_age2, n_fit_dM_age2, n_dlogM_age2, n_fit_dlogM_age2, ncum_age2, ncum_fit_age2, n_fit_age2_err, n_fit_dM_age2_err, n_fit_dlogM_age2_err = makearrayhist(gal_array_age2_masslimit, mass_bins_log, complimits[1])
			histogram_twoplot(galname, age2_label, bins_age2, n_dM_age2, bins_fit_age2, n_fit_dM_age2, n_fit_dM_age2_err, complimits[1], binsnew, nnew_dM, binsnew_fit, nnew_fit_dM, nnew_fit_dM_err, percentile_gal_value, array_plot, plotaxes)

	if ageflag == 3:
		print('3 = (100, 400] Myr + input completeness limit A3')
		gal_array_age3_tmp = gal_array[gal_array[:,6] > 100.01*1E6]
		gal_array_age3 = gal_array_age3_tmp[gal_array_age3_tmp[:,6] <= 400.01*1E6]
		gal_array_age3_masslimit = gal_array_age3[gal_array_age3[:,4] > complimits[2]]
		age3_label = r' (100 < $\tau \leq$ 400 Myr)'
		if len(gal_array_age3_masslimit) > 0:
			n_age3, bins_age3, bins_width_age3, bins_centre_age3, n_fit_age3, bins_fit_age3, n_dM_age3, n_fit_dM_age3, n_dlogM_age3, n_fit_dlogM_age3, ncum_age3, ncum_fit_age3, n_fit_age3_err, n_fit_dM_age3_err, n_fit_dlogM_age3_err = makearrayhist(gal_array_age3_masslimit, mass_bins_log, complimits[2])
			histogram_twoplot(galname, age3_label, bins_age3, n_dM_age3, bins_fit_age3, n_fit_dM_age3, n_fit_dM_age3_err, complimits[2], binsnew, nnew_dM, binsnew_fit, nnew_fit_dM, nnew_fit_dM_err, percentile_gal_value, array_plot, plotaxes)

	if ageflag == 5:
		print('5 = [1, 200] Myr + input completeness limit A3')
		gal_array_age5 = gal_array[gal_array[:,6] <= 200.01*1E6]
		gal_array_age5_masslimit = gal_array_age5[gal_array_age5[:,4] > complimits[2]]
		age5_label = r' (1 <= $\tau \leq$ 200 Myr)'
		if len(gal_array_age5_masslimit) > 0:
			n_age5, bins_age5, bins_width_age5, bins_centre_age5, n_fit_age5, bins_fit_age5, n_dM_age5, n_fit_dM_age5, n_dlogM_age5, n_fit_dlogM_age5, ncum_age5, ncum_fit_age5, n_fit_age5_err, n_fit_dM_age5_err, n_fit_dlogM_age5_err = makearrayhist(gal_array_age5_masslimit, mass_bins_log, complimits[2])
			histogram_twoplot(galname, age5_label, bins_age5, n_dM_age5, bins_fit_age5, n_fit_dM_age5, n_fit_dM_age5_err, complimits[2], binsnew, nnew_dM, binsnew_fit, nnew_fit_dM, nnew_fit_dM_err, percentile_gal_value, array_plot, plotaxes)

def plotaxes_threehistogram(gal_array, galname, complimits, ageflag, outflag):

	'''
	Function: Plot cluster dataset - oneplot
	'''

	print('Starting {}'.format(galname))

	mass_bins_log = np.power(10, np.linspace(2, 9, num = 29))
	array_plot = np.power(10, np.linspace(np.log10(np.min(mass_bins_log)), np.log10(np.max(mass_bins_log))))

	def histogram_threeplot(galname, age_label, bins, n_dM, bins_fit, n_fit_dM, n_fit_dM_err, complimits, array_plot, marker_symbol):

		if age_label == r'10 < $\tau \leq$ 100 Myr':
			plt.errorbar(bins_fit, n_fit_dM * 0.1, marker = marker_symbol, linestyle = 'None', markerfacecolor = 'g', markersize = 10, yerr = n_fit_dM_err * 0.1, label = age_label)
		elif age_label == r'100 < $\tau \leq$ 400 Myr':
			plt.errorbar(bins_fit, n_fit_dM * 0.01, marker = marker_symbol, linestyle = 'None', markerfacecolor = 'g', markersize = 10, yerr = n_fit_dM_err * 0.01, label = age_label)
		else:
			plt.errorbar(bins_fit, n_fit_dM, marker = marker_symbol, linestyle = 'None', markerfacecolor = 'g', markersize = 10, yerr = n_fit_dM_err, label = age_label)
		popt_truncatedpowerlaw, pcov_truncatedpowerlaw, popt_simplepowerlaw, pcov_simplepowerlaw, powerlawchisq, popt_schechter, pcov_schechter, schechterchisq = curve_fit3(bins_fit, n_fit_dM, [1E2, 1E8], np.nanmax(complimits), 1, 0, n_fit_dM_err)
		array_plot = np.power(10, np.linspace(np.log10(np.min(bins_fit)), np.log10(np.max(bins_fit))))
		if outflag == 1:
			print(popt_simplepowerlaw)
			if age_label == r'10 < $\tau \leq$ 100 Myr':
				plt.plot(array_plot, 0.1 * simplepowerlaw(array_plot, *popt_simplepowerlaw), 'k--')
			elif age_label == r'100 < $\tau \leq$ 400 Myr':
				plt.plot(array_plot, 0.01 * simplepowerlaw(array_plot, *popt_simplepowerlaw), 'k--')
			else:
				plt.plot(array_plot, simplepowerlaw(array_plot, *popt_simplepowerlaw), 'k--')
		else:
			plt.plot(array_plot, simplepowerlaw(array_plot, *popt_simplepowerlaw), 'r--', label = r'Power Law')
		# plt.plot([np.nanmax(complimits), np.nanmax(complimits)], [1E-10, 1E10], 'k--')
		# plt.legend(loc = 'upper right', title = galname + age_label)

		return 0

	if ageflag == 1:
		print(galname)
		print('1 = (0, 10] Myr + input completeness limit A1')
		gal_array_age1 = gal_array[gal_array[:,6] <= 10.01*1E6]
		gal_array_age1_masslimit = gal_array_age1[gal_array_age1[:,4] > complimits[0]]
		age1_label = r'$\tau \leq$ 10 Myr'
		if len(gal_array_age1_masslimit) > 0:
			n_age1, bins_age1, bins_width_age1, bins_centre_age1, n_fit_age1, bins_fit_age1, n_dM_age1, n_fit_dM_age1, n_dlogM_age1, n_fit_dlogM_age1, ncum_age1, ncum_fit_age1, n_fit_age1_err, n_fit_dM_age1_err, n_fit_dlogM_age1_err = makearrayhist(gal_array_age1_masslimit, mass_bins_log, complimits[0])
			histogram_threeplot(galname, age1_label, bins_age1, n_dM_age1, bins_fit_age1, n_fit_dM_age1, n_fit_dM_age1_err, complimits[0], array_plot, 'o')

	if ageflag == 2:
		print('2 = (10, 100] Myr + input completeness limit A2')
		gal_array_age2_tmp = gal_array[gal_array[:,6] > 10.01*1E6]
		gal_array_age2 = gal_array_age2_tmp[gal_array_age2_tmp[:,6] <= 100.01*1E6]
		gal_array_age2_masslimit = gal_array_age2[gal_array_age2[:,4] > complimits[1]]
		age2_label = r'10 < $\tau \leq$ 100 Myr'
		if len(gal_array_age2_masslimit) > 0:
			n_age2, bins_age2, bins_width_age2, bins_centre_age2, n_fit_age2, bins_fit_age2, n_dM_age2, n_fit_dM_age2, n_dlogM_age2, n_fit_dlogM_age2, ncum_age2, ncum_fit_age2, n_fit_age2_err, n_fit_dM_age2_err, n_fit_dlogM_age2_err = makearrayhist(gal_array_age2_masslimit, mass_bins_log, complimits[1])
			histogram_threeplot(galname, age2_label, bins_age2, n_dM_age2, bins_fit_age2, n_fit_dM_age2, n_fit_dM_age2_err, complimits[1], array_plot, '^')

	if ageflag == 3:
		print('3 = (100, 400] Myr + input completeness limit A3')
		gal_array_age3_tmp = gal_array[gal_array[:,6] > 100.01*1E6]
		gal_array_age3 = gal_array_age3_tmp[gal_array_age3_tmp[:,6] <= 400.01*1E6]
		gal_array_age3_masslimit = gal_array_age3[gal_array_age3[:,4] > complimits[2]]
		age3_label = r'100 < $\tau \leq$ 400 Myr'
		if len(gal_array_age3_masslimit) > 0:
			n_age3, bins_age3, bins_width_age3, bins_centre_age3, n_fit_age3, bins_fit_age3, n_dM_age3, n_fit_dM_age3, n_dlogM_age3, n_fit_dlogM_age3, ncum_age3, ncum_fit_age3, n_fit_age3_err, n_fit_dM_age3_err, n_fit_dlogM_age3_err = makearrayhist(gal_array_age3_masslimit, mass_bins_log, complimits[2])
			histogram_threeplot(galname, age3_label, bins_age3, n_dM_age3, bins_fit_age3, n_fit_dM_age3, n_fit_dM_age3_err, complimits[2], array_plot, 's')

	if ageflag == 5:
		print('5 = [1, 200] Myr + input completeness limit A3')
		gal_array_age5 = gal_array[gal_array[:,6] <= 200.01*1E6]
		gal_array_age5_masslimit = gal_array_age5[gal_array_age5[:,4] > complimits[2]]
		age5_label = r' (1 <= $\tau \leq$ 200 Myr)'
		if len(gal_array_age5_masslimit) > 0:
			n_age5, bins_age5, bins_width_age5, bins_centre_age5, n_fit_age5, bins_fit_age5, n_dM_age5, n_fit_dM_age5, n_dlogM_age5, n_fit_dlogM_age5, ncum_age5, ncum_fit_age5, n_fit_age5_err, n_fit_dM_age5_err, n_fit_dlogM_age5_err = makearrayhist(gal_array_age5_masslimit, mass_bins_log, complimits[2])
			histogram_threeplot(galname, age5_label, bins_age5, n_dM_age5, bins_fit_age5, n_fit_dM_age5, n_fit_dM_age5_err, complimits[2], array_plot, '^')

	if ageflag == 0:
		gal_array_age1 = gal_array[gal_array[:,6] <= 10.01*1E6]
		gal_array_age1_masslimit = gal_array_age1[gal_array_age1[:,4] > complimits[0]]
		gal_array_age2_tmp = gal_array[gal_array[:,6] > 10.01*1E6]
		gal_array_age2 = gal_array_age2_tmp[gal_array_age2_tmp[:,6] <= 100.01*1E6]
		gal_array_age2_masslimit = gal_array_age2[gal_array_age2[:,4] > complimits[1]]
		gal_array_age3_tmp = gal_array[gal_array[:,6] > 100.01*1E6]
		gal_array_age3 = gal_array_age3_tmp[gal_array_age3_tmp[:,6] <= 400.01*1E6]
		gal_array_age3_masslimit = gal_array_age3[gal_array_age3[:,4] > complimits[2]]
		gal_array_masslimit = np.concatenate([gal_array_age1_masslimit, gal_array_age2_masslimit, gal_array_age3_masslimit])
		age5_label = r' ($\tau \leq$ 400 Myr)'
		n, bins, bins_width, bins_centre, n_fit, bins_fit, n_dM, n_fit_dM, n_dlogM, n_fit_dlogM, ncum, ncum_fit, n_fit_err, n_fit_dM_err, n_fit_dlogM_err = makearrayhist(gal_array_masslimit, mass_bins_log, np.nanmax(complimits))
		histogram_threeplot(galname, age_label, bins, n_dM, bins_fit, n_fit_dM, n_fit_dM_err, np.nanmax(complimits), array_plot,
			'^')

def calculateslopeandcorragebins(x, y1, y2, y3):


	'''
	Function: Calculate slope and plot (x + x + x) vs (y1 + y2 + y3)
	'''

	###
	plot_x = np.linspace(-20, 20)
	###
	popt, pcov = optimize.curve_fit(linefunction, x + x + x, y1 + y2 + y3, maxfev = 100000)
	rho, pval = stats.spearmanr(x + x + x, y1 + y2 + y3)
	print('Spearman: {:.2f}, {:.2e}'.format(rho, pval))
	print('Fitting Routine Results: {:.2f}, {:.2f}'.format(popt[0], popt[1]))
	plt.plot(plot_x, linefunction(plot_x, *popt), 'k-', label = r'm = {:.2f} $\pm$ {:.2f}'.format(popt[0], np.sqrt(abs(pcov[0][0]))) + '\n ' + r'$\rho$ = {:.2f}, p = {:.1e}'.format(rho, pval))

	return 0

def calculateslopeandcorragebinsx(x1, x2, x3, y):

	'''
	Function: Calculate slope and plot (x1 + x2 + x3) vs (y + y + y)
	'''

	###
	plot_x = np.linspace(-20, 20)
	###
	popt, pcov = optimize.curve_fit(linefunction, x1 + x2 + x3, y + y + y, maxfev = 100000)
	rho, pval = stats.spearmanr(x1 + x2 + x3, y + y + y)
	print('Spearman: {:.2f}, {:.2e}'.format(rho, pval))
	print('Fitting Routine Results: {:.2f}, {:.2f}'.format(popt[0], popt[1]))
	plt.plot(plot_x, linefunction(plot_x, *popt), 'k-', label = r'm = {:.2f} $\pm$ {:.2f}'.format(popt[0], np.sqrt(abs(pcov[0][0]))) + '\n ' + r'$\rho$ = {:.2f}, p = {:.1e}'.format(rho, pval))

	return 0

def calculateslopeandcorragebin(x, y):

	'''
	Function:
	'''

	###
	plot_x = np.linspace(-20, 20)
	###
	popt1, pcov1 = optimize.curve_fit(linefunction, x, y, maxfev = 100000)
	rho1, pval1 = stats.spearmanr(x, y)
	print('Spearman: {:.2f}, {:.2e}'.format(rho1, pval1))
	print('Fitting Routine Results: {:.2f}, {:.2f}'.format(popt1[0], popt1[1]))
	plt.plot(plot_x, linefunction(plot_x, *popt1), 'k-', label = r'm = {:.2f} $\pm$ {:.2f}'.format(popt1[0], np.sqrt(abs(pcov1[0][0]))) + '\n ' + r'$\rho$ = {:.2f}, p = {:.1e}'.format(rho1, pval1))

	return 0

def calculateslopeandcorragebin_log(x, y):

	'''
	Function:
	'''

	###
	plot_x = np.power(10, np.linspace(-20, 20))
	###
	log10_x = np.log10(x)
	log10_y = np.log10(y)
	###
	popt1, pcov1 = optimize.curve_fit(linefunction, log10_x, log10_y, maxfev = 100000)
	rho1, pval1 = stats.spearmanr(log10_x, log10_y)
	print('Spearman: {:.2f}, {:.2e}'.format(rho1, pval1))
	print('Fitting Routine Results (A1): {:.2f}, {:.2f}'.format(popt1[0], popt1[1]))
	plt.plot(plot_x, np.power(10, linefunction(np.log10(plot_x), *popt1)), 'k-', label = r'm = {:.2f} $\pm$ {:.2f}'.format(popt1[0], np.sqrt(abs(pcov1[0][0]))) + '\n ' + r'$\rho$ = {:.2f}, p = {:.1e}'.format(rho1, pval1))

	return 0

# Function:
def calculateslopeandcorragebin_semilog(x, y):

	'''
	Function:
	'''

	###
	plot_x = np.power(10, np.linspace(-20, 20))
	###
	log10_x = np.log10(x)
	###
	popt1, pcov1 = optimize.curve_fit(linefunction, log10_x, y, maxfev = 100000)
	rho1, pval1 = stats.spearmanr(log10_x, y)
	print('Spearman: {:.2f}, {:.2e}'.format(rho1, pval1))
	print('Fitting Routine Results (A1): {:.2f}, {:.2f}'.format(popt1[0], popt1[1]))
	plt.plot(plot_x, linefunction(np.log10(plot_x), *popt1), 'k-', label = r'm = {:.2f} $\pm$ {:.2f}'.format(popt1[0], np.sqrt(abs(pcov1[0][0]))) + '\n ' + r'$\rho$ = {:.2f}, p = {:.1e}'.format(rho1, pval1))

	return 0

def returnmaxvalue(gal_array, complimits, galname):

	'''
	Function: Return max value in 3 age bins
	'''

	# Create 3 age bins
	gal_array_age1 = gal_array[gal_array[:,6] <= 10.01*1E6]
	gal_array_age1_masslimit = gal_array_age1[gal_array_age1[:,4] > complimits[0]]
	age1_label = r' ($\tau \leq$ 10 Myr)'
	gal_array_age2_tmp = gal_array[gal_array[:,6] > 10.01*1E6]
	gal_array_age2 = gal_array_age2_tmp[gal_array_age2_tmp[:,6] <= 100.01*1E6]
	gal_array_age2_masslimit = gal_array_age2[gal_array_age2[:,4] > complimits[1]]
	age2_label = r' (10 < $\tau \leq$ 100 Myr)'
	gal_array_age3 = gal_array[gal_array[:,6] <= 200.01*1E6]
	gal_array_age3_masslimit = gal_array_age3[gal_array_age3[:,4] > complimits[2]]
	age3_label = r' ($\tau \leq$ 200 Myr)'
	gal_array_age3b_tmp = gal_array[gal_array[:,6] > 100.01*1E6]
	gal_array_age3b = gal_array_age3b_tmp[gal_array_age3b_tmp[:,6] <= 400.01*1E6]
	gal_array_age3b_masslimit = gal_array_age3b[gal_array_age3b[:,4] > complimits[2]]
	age3b_label = r' (100 < $\tau \leq$ 400 Myr)'

	age1_max = np.log10(np.nanmax(gal_array_age1_masslimit[:,4]))
	age2_max = np.log10(np.nanmax(gal_array_age2_masslimit[:,4]))
	age3_max = np.log10(np.nanmax(gal_array_age3_masslimit[:,4]))

	print('{}: {:.2f}, {:.2f}, {:.2f}'.format(galname, age1_max, age2_max, age3_max))

	return [age1_max, age2_max, age3_max]

def plotaxes_xb_histogram(gal_array_masslimit, complimits, marker, color, label, flag_fit, plotaxes):

	'''
	Function: 
	'''

	mass_bins_log = np.power(10, np.linspace(2, 9, num = 29))
	n, bins, bins_width, bins_centre, n_fit, bins_fit, n_dM, n_fit_dM, n_dlogM, n_fit_dlogM, ncum, ncum_fit, n_fit_err, n_fit_dM_err, n_fit_dlogM_err = makearrayhist(gal_array_masslimit, mass_bins_log, np.nanmax(complimits))
	plotaxes.step(bins, np.append(n_dM[0], n_dM), color = color, alpha = 0.5)
	plotaxes.errorbar(bins_fit, n_fit_dM, marker = marker, linestyle = 'None', markerfacecolor = color, markeredgecolor = color, yerr = n_fit_dM_err, ecolor = color, label = label)

	if flag_fit > 0:
		# plt.plot([complimits, complimits], [1E-10, 1E10], 'k--')
		popt_truncatedpowerlaw, pcov_truncatedpowerlaw, popt_simplepowerlaw, pcov_simplepowerlaw, powerlawchisq, popt_schechter, pcov_schechter, schechterchisq = curve_fit3(bins_fit, n_fit_dM, [1E2, 1E8], np.nanmax(complimits), 1, 0, n_fit_dM_err)
		array_plot = np.power(10, np.linspace(np.log10(np.min(bins_fit)) - 1.0, np.log10(np.max(bins_fit)) + 1.0))
		if flag_fit == 1:
			plotaxes.plot(array_plot, schechter(array_plot, *popt_schechter), 'b-.')
		elif flag_fit == 2:
			plotaxes.plot(array_plot, simplepowerlaw(array_plot, *popt_simplepowerlaw), 'k-.')
		plotaxes.plot([np.nanmax(complimits), np.nanmax(complimits)], [1E-10, 1E10], 'k--')


	return 0

def overplot_histogram(array1, complimits_val, linetype, labelshape, labelcolour, textlabel, norm_val = 0, sortindex = -1, massindex = 4, equal = False, outputbinstofile = False, alphaval = 0.4):

	'''
	Function: 
	'''

	print('--- Plotting histogram for {} ---'.format(textlabel))

	if sortindex > 0:
		arraytemp = array1[array1[:,sortindex] == 1]
		array1 = arraytemp

	mass_bins_log = np.power(10, np.linspace(1, 6, num = 21))
	array_plot = np.power(10, np.linspace(np.log10(np.min(mass_bins_log)), np.log10(np.max(mass_bins_log)) + 1))

	if equal == False:
		n, bins, bins_width, bins_centre, n_fit, bins_fit, n_dM, n_fit_dM, n_dlogM, n_fit_dlogM, ncum, ncum_fit, n_fit_err, n_fit_dM_err, n_fit_dlogM_err = makearrayhist(array1, mass_bins_log, complimits_val, massindex = massindex)
		plt.errorbar(bins_fit, n_fit_dM * np.power(10, norm_val), marker = labelshape, linestyle = 'None', color = labelcolour, markeredgecolor = labelcolour, markerfacecolor = labelcolour, yerr = n_fit_dM_err * np.power(10, norm_val), markersize = 20, label = textlabel, alpha = alphaval)

	else:
		n, bins, bins_width, bins_centre, n_fit, bins_fit, n_dM, n_fit_dM, n_dlogM, n_fit_dlogM, ncum, ncum_fit, n_fit_err, n_fit_dM_err, n_fit_dlogM_err = makearrayhistequal(array1, complimits_val, -1, massindex = massindex, numgal_bin_in = 3)
		plt.errorbar(bins_fit, n_fit_dM * np.power(10, norm_val), marker = labelshape, linestyle = 'None', color = labelcolour, markeredgecolor = labelcolour, markerfacecolor = labelcolour, yerr = n_fit_dM_err * np.power(10, norm_val), markersize = 20, label = textlabel, alpha = alphaval)

	popt_truncatedpowerlaw, pcov_truncatedpowerlaw, popt_simplepowerlaw, pcov_simplepowerlaw, powerlawchisq, popt_schechter, pcov_schechter, schechterchisq = curve_fit3(bins_fit, n_fit_dM * np.power(10, norm_val), [1E2, 1E8], complimits_val, 1, 0, n_fit_dM_err * np.power(10, norm_val))

	# Print Bins to File
	if outputbinstofile == True:
		f1 = open('./Logs/ZOutput_' + textlabel +  '_Bins.txt', 'w')
		for i in range(0, len(bins_fit)):
			outputstring = '{:.3e}, {:.3e}, {:.3e}\n'.format(bins_fit[i], n_fit_dM[i], n_fit_dM_err[i])
			f1.write(outputstring)
		f1.close()

	return bins_fit, n_fit_dM * np.power(10, norm_val), n_fit_dM_err * np.power(10, norm_val)

def overplot_cumulative(gal_array_masslimit, complimits, color, linestyle, label, flag_fit):

	'''
	Function: 
	'''
	print('>>>')	
	print('>>> Plotting cumulative function for {}'.format(label))	

	# Sort data array, create x and y arrays
	gal_array_masslimit_in = gal_array_masslimit[:,4]
	sorted_data = np.sort(gal_array_masslimit_in)
	xdata = np.concatenate([sorted_data[::-1]])
	ydata = np.arange(sorted_data.size) + 1
	
	# Filter array to those > complimit, output results
	filt_array = np.where(sorted_data >= complimits)
	data_lim = sorted_data[filt_array]
	sorted_data_lim = np.sort(data_lim)

	# Output results
	xdata_lim = np.concatenate([sorted_data_lim[::-1]])
	ydata_lim = np.arange(sorted_data_lim.size) + 1
	xdata_lim_float = xdata_lim.astype(float)
	ydata_lim_float = ydata_lim.astype(float)

	# For Mineo dataset, scale down 30%
	if label == 'Mineo+14':
		ydata = ydata * 0.7
		ydata_lim_float = ydata_lim_float * 0.7

	# Print result to screen
	print('CDF Function Results {:.2e}:'.format(complimits))
	print('[Len: {}, Min: {:.2e}, Max:{:.2e}]'.format(len(xdata), np.nanmin(xdata), np.nanmax(xdata)))
	print('[Len: {}, Min: {:.2e}, Max:{:.2e}]'.format(len(xdata_lim_float), np.nanmin(xdata_lim_float), np.nanmax(xdata_lim_float)))

	plt.step(xdata, ydata, color = color, linestyle = linestyle, label = label)

	if flag_fit > 0:
		# plt.plot([complimits, complimits], [1E-10, 1E10], 'k--')
		popt_truncatedpowerlaw, pcov_truncatedpowerlaw, popt_simplepowerlaw, pcov_simplepowerlaw, powerlawchisq, popt_schechter, pcov_schechter, schechterchisq = curve_fit3(xdata_lim_float, ydata_lim_float, [1E2, 1E8], np.nanmax(complimits), 1, 0, np.sqrt(ydata_lim_float))
		# array_plot = np.power(10, np.linspace(np.log10(np.min(xdata_lim_float)) - 1.0, np.log10(np.max(xdata_lim_float)) + 1.0))
		array_plot = np.power(10, np.linspace(np.log10(complimits), np.log10(np.max(xdata_lim_float)) + 1.0))
		if flag_fit == 1:
			plt.plot(array_plot, schechter(array_plot, *popt_schechter), 'b-.')
		elif flag_fit == 2:
			if label == 'This Work':
				plt.plot(array_plot, simplepowerlaw(array_plot, *popt_simplepowerlaw), 'b-')
			else:
				plt.plot(array_plot, simplepowerlaw(array_plot, *popt_simplepowerlaw), 'b-.')
		plt.plot([np.nanmax(complimits), np.nanmax(complimits)], [1E-10, 1E10], 'k--')

def plotaxes_cumulative(gal_array_masslimit, complimits, color, linestyle, label, flag_fit, plotaxes):

	'''
	Function: 
	'''

	print('>>>')	
	print('>>> Plotting cumulative function for {}'.format(label))	
	
	# Sort data array, create x and y arrays
	gal_array_masslimit_in = gal_array_masslimit[:,4]
	sorted_data = np.sort(gal_array_masslimit_in)
	xdata = np.concatenate([sorted_data[::-1]])
	ydata = np.arange(sorted_data.size) + 1
	
	# Filter array to those > complimit, output results
	filt_array = np.where(sorted_data >= complimits)
	data_lim = sorted_data[filt_array]
	sorted_data_lim = np.sort(data_lim)

	# Output results
	xdata_lim = np.concatenate([sorted_data_lim[::-1]])
	ydata_lim = np.arange(sorted_data_lim.size) + 1
	xdata_lim_float = xdata_lim.astype(float)
	ydata_lim_float = ydata_lim.astype(float)
	
	# Print result to screen
	print('CDF Function Results {:.2e}:'.format(complimits))
	print('[Len: {}, Min: {:.2e}, Max:{:.2e}]'.format(len(xdata), np.nanmin(xdata), np.nanmax(xdata)))
	print('[Len: {}, Min: {:.2e}, Max:{:.2e}]'.format(len(xdata_lim_float), np.nanmin(xdata_lim_float), np.nanmax(xdata_lim_float)))

	plotaxes.step(xdata, ydata, color = color, linestyle = linestyle, label = label)

	if flag_fit > 0:
		# plt.plot([complimits, complimits], [1E-10, 1E10], 'k--')
		popt_truncatedpowerlaw, pcov_truncatedpowerlaw, popt_simplepowerlaw, pcov_simplepowerlaw, powerlawchisq, popt_schechter, pcov_schechter, schechterchisq = curve_fit3(xdata_lim_float, ydata_lim_float, [1E2, 1E8], np.nanmax(complimits), 1, 0, np.sqrt(ydata_lim_float))
		array_plot = np.power(10, np.linspace(np.log10(np.min(xdata_lim_float)) - 1.0, np.log10(np.max(xdata_lim_float)) + 1.0))
		if flag_fit == 1:
			plotaxes.plot(array_plot, schechter(array_plot, *popt_schechter), 'b-.')
		elif flag_fit == 2:
			plotaxes.plot(array_plot, simplepowerlaw(array_plot, *popt_simplepowerlaw), 'k-.')
		plotaxes.plot([np.nanmax(complimits), np.nanmax(complimits)], [1E-10, 1E10], 'k--')


	return 0

def calculateslopeandcorragebin_plotaxes(x, y, plotaxes):

	'''
	Function: 
	'''

	plot_x = np.linspace(-20, 20)
	###
	popt1, pcov1 = optimize.curve_fit(linefunction, x, y, maxfev = 100000)
	rho1, pval1 = stats.spearmanr(x, y)
	print('Spearman: {:.2f}, {:.2e}'.format(rho1, pval1))
	print('Fitting Routine Results: {:.2f}, {:.2f}'.format(popt1[0], popt1[1]))
	plotaxes.plot(plot_x, linefunction(plot_x, *popt1), 'k-', label = r'm = {:.2f} $\pm$ {:.2f}'.format(popt1[0], np.sqrt(abs(pcov1[0][0]))) + '\n ' + r'$\rho$ = {:.2f}, p = {:.1e}'.format(rho1, pval1))
	return 0

def find_nearest(array, value):

	'''
	Function: Find nearest value in array
	'''

	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	arraytrim = array[:idx]
	# print (idx, array[idx], len(arraytrim), np.nanmax(arraytrim))
	return idx, array[idx]

def outputanalyticresult(testarrayint, testval):

	'''
	Function: 
	'''

	analytic_idx, analytic_val = find_nearest(testarrayint, testval)
	analytic_maxmass = testarray[analytic_idx]
	print('{:.1f} - {:.2f}'.format(np.log10(testval), np.log10(analytic_maxmass)))

	return analytic_maxmass

def calculatemaxmass(mmaxlist):

	'''
	Function: 
	'''

	# Analytic Power Law Solution
	from pynverse import inversefunc
	massrel = (lambda x: x * np.log(x / 1E4))
	testarray = np.power(10, np.linspace(4, 7, num = 3000))
	testarrayint = []
	for i in range(0, len(testarray)):
		testarrayint.append(massrel(testarray[i]))

	pl_mmaxlist_out = []
	for i in range(0, len(mmaxlist)):
		temp_idx, tempidx_val = find_nearest(testarrayint, mmaxlist[i])
		temp_maxmass = testarray[temp_idx]
		print('{:.1f} - {:.2f}'.format(np.log10(mmaxlist[i]), np.log10(temp_maxmass)))
		pl_mmaxlist_out.append(temp_maxmass)

	# Analytic Schechter (1E5) Solution
	fun = lambda x: x * np.power(x / 1E5, -2) * np.exp(-(x / 1E5))
	fun2 = lambda x: np.power(x / 1E5, -2) * np.exp(-(x / 1E5))
	massrel = (lambda x: (integrate.quad(fun, 1E4, x)[0]) / np.nanmax([integrate.quad(fun2, x, np.inf)[0], -99999]))
	testarrayint = []
	for i in range(0, len(testarray)):
		testarrayint.append(massrel(testarray[i]))

	sch_mmaxlist_out = []
	for i in range(0, len(mmaxlist)):
		temp_idx, tempidx_val = find_nearest(testarrayint, mmaxlist[i])
		temp_maxmass = testarray[temp_idx]
		print('{:.1f} - {:.2f}'.format(np.log10(mmaxlist[i]), np.log10(temp_maxmass)))
		sch_mmaxlist_out.append(temp_maxmass)

	return pl_mmaxlist_out, sch_mmaxlist_out

def calculatetotalmassA5(gal_array, complimits):

	'''
	Function: 
	'''

	gal_array_age5_tmp = gal_array[gal_array[:,6] > 0.99*1E6]
	gal_array_age5 = gal_array_age5_tmp[gal_array_age5_tmp[:,6] <= 200.01*1E6]
	# gal_array_age5_masslimit = gal_array_age5[gal_array_age5[:,4] > complimits[2]]
	gal_array_age5_masslimit = gal_array_age5[gal_array_age5[:,4] >= 1E4]
	gal_array_age5_val = np.nanmax([1E4, np.sum(gal_array_age5_masslimit[:,4])])
	# print(gal_array_age5_val)
	return gal_array_age5_val

def returnarraywhereabovelimit(gal_array, array_len, ageflag, complimits):

	'''
	Function: Create array selecting only those in age range and above mass limit
	'''

	if ageflag == 1:
		gal_array_age1 = gal_array[gal_array[:,6] <= 10.01*1E6]
		gal_array_age1_masslimit = gal_array_age1[gal_array_age1[:,4] > complimits[0]]
		output = gal_array_age1_masslimit[:(array_len)]
	elif ageflag == 2:
		gal_array_age2_tmp = gal_array[gal_array[:,6] > 10.01*1E6]
		gal_array_age2 = gal_array_age2_tmp[gal_array_age2_tmp[:,6] <= 100.01*1E6]
		gal_array_age2_masslimit = gal_array_age2[gal_array_age2[:,4] > complimits[1]]
		output = gal_array_age2_masslimit[:(array_len)]
	else:
		gal_array_age3_tmp = gal_array[gal_array[:,6] > 100.01*1E6]
		gal_array_age3 = gal_array_age3_tmp[gal_array_age3_tmp[:,6] <= 400.01*1E6]
		gal_array_age3_masslimit = gal_array_age3[gal_array_age3[:,4] > complimits[2]]
		output = gal_array_age3_masslimit[:(array_len)]

	print('Finished outputing - length: {}'.format(len(output)))

	return output

#------------------------------------------------------------------------------
###
# (12) Code Snippets (Analysis II - Make Plots)
###

def rungalaxyP_output(galname, np_SCP_B_array, np_SCP_C_array, SCP_B_complimits, SCP_C_complimits):

	
	'''
	Function: Create PHANGS specific plots
	'''

	### --==--==--==-- ###
	print('>>>')
	print('>>> {}_SCP_AgeBins_00_AgeMass'.format(galname))
	print('>>>')
	fig, axes = plt.subplots(1, 2, sharex = 'all', sharey = 'all', figsize = (24, 12))
	fig.subplots_adjust(hspace = 0, wspace = 0, bottom = 0.15)
	plotaxes_agemass(np_SCP_B_array, '{} (Bayes)'.format(galname), SCP_B_complimits, axes[0], 1)
	plotaxes_agemass(np_SCP_C_array, r'($\chi^2$)', SCP_C_complimits, axes[1], 1)
	fig.text(0.5, 0.05, r'log (Age/yr)', ha = 'center', fontsize = 45)
	fig.text(0.05, 0.5, r'log (M/M$_\odot$)', va = 'center', rotation = 'vertical', fontsize = 45)
	plt.xscale('log', nonposx = 'clip')
	plt.yscale('log', nonposy = 'clip')
	axes[0].xaxis.set_major_formatter(log10_labels_format)
	axes[0].yaxis.set_major_formatter(log10_labels_format)
	axes[0].minorticks_off()
	plt.axis([1E5, 1E9] + [1E2, 1E8])
	plt.xticks([1E6, 1E7, 1E8])
	plt.yticks([1E3, 1E5, 1E7])
	plt.savefig('./FiguresSC/{}_SCP_AgeBins_00_AgeMass.png'.format(galname))
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> {}_SCP_AgeBins_01_Like'.format(galname))
	print('>>>')
	fig, axes = plt.subplots(2, 3, sharex = 'all', sharey = 'all', figsize = (36, 24))
	fig.subplots_adjust(hspace = 0, wspace = 0, bottom = 0.15)
	plotaxes_likelihood(np_SCP_B_array, 'Bayesian (< 10 Myr)', SCP_B_complimits, axes[0, 0], 1, 0, 0, 1, 'Test')
	plotaxes_likelihood(np_SCP_B_array, '(10 - 100 Myr)', SCP_B_complimits, axes[0, 1], 2, 0, 0, 1, 'Test')
	plotaxes_likelihood(np_SCP_B_array, '(100 - 400 Myr)', SCP_B_complimits, axes[0, 2], 3, [3.5, 2.75], 0, 1, 'Test')
	plotaxes_likelihood(np_SCP_C_array, r'$\chi^2$ (< 10 Myr)', SCP_C_complimits, axes[1, 0], 1, 0, 0, 1, 'Test')
	plotaxes_likelihood(np_SCP_C_array, '(10 - 100 Myr)', SCP_C_complimits, axes[1, 1], 2, [5.5, 1.0], 0, 1, 'Test')
	plotaxes_likelihood(np_SCP_C_array, '(100 - 400 Myr)', SCP_C_complimits, axes[1, 2], 3, [3.5, 2.75], 0, 1, 'Test')
	fig.text(0.5, 0.05, r'log (M$_*$/M$_\odot$)', ha = 'center', fontsize = 45)
	fig.text(0.05, 0.5, r'$-\beta$', va = 'center', rotation = 'vertical', fontsize = 45)
	plt.axis([3.0, 7.5] + [0, 3])
	plt.xticks([4.0, 5.0, 6.0, 7.0])
	plt.yticks([0.0, 1.0, 2.0])
	plt.savefig('./FiguresSC/{}_SCP_AgeBins_01_Like.png'.format(galname))
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> {}_SCP_AgeBins_02_MSF'.format(galname))
	print('>>>')
	fig, axes = plt.subplots(2, 3, sharex = 'all', sharey = 'all', figsize = (36, 24))
	fig.subplots_adjust(hspace = 0, wspace = 0, bottom = 0.15)
	plotaxes_mspecfit(np_SCP_B_array, 'Bayesian (< 10 Myr)', SCP_B_complimits, axes[0, 0], 1, 0, 0)
	plotaxes_mspecfit(np_SCP_B_array, '(10 - 100 Myr)', SCP_B_complimits, axes[0, 1], 2, 0, 0)
	plotaxes_mspecfit(np_SCP_B_array, '(100 - 400 Myr)', SCP_B_complimits, axes[0, 2], 3, 0, 0)
	plotaxes_mspecfit(np_SCP_C_array, r'$\chi^2$ (< 10 Myr)', SCP_C_complimits, axes[1, 0], 1, 0, 0)
	plotaxes_mspecfit(np_SCP_C_array, '(10 - 100 Myr)', SCP_C_complimits, axes[1, 1], 2, 0, 0)
	plotaxes_mspecfit(np_SCP_C_array, '(100 - 400 Myr)', SCP_C_complimits, axes[1, 2], 3, 0, 0)
	fig.text(0.5, 0.05, r'log (M/M$_\odot$)', ha = 'center', fontsize = 45)
	fig.text(0.05, 0.5, r'log N (> M)', va = 'center', rotation = 'vertical', fontsize = 45)
	plt.xscale('log', nonposx = 'clip')
	plt.yscale('log', nonposy = 'clip')
	axes[0, 0].xaxis.set_major_formatter(log10_labels_format)
	axes[0, 0].yaxis.set_major_formatter(log10_labels_format)
	axes[0, 0].minorticks_off()
	plt.axis([np.power(10, 3), np.power(10, 7.5)] + [np.power(10, -0.5), np.power(10, 3.5)])
	plt.xticks([1E4, 1E5, 1E6, 1E7])
	plt.yticks([1E0, 1E1, 1E2, 1E3])
	plt.savefig('./FiguresSC/{}_SCP_AgeBins_02_MSF.png'.format(galname))
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> {}_SCP_AgeBins_03_Hist'.format(galname))
	print('>>>')
	fig, axes = plt.subplots(2, 3, sharex = 'all', sharey = 'all', figsize = (36, 24))
	fig.subplots_adjust(hspace = 0, wspace = 0, bottom = 0.15)
	plotaxes_histogram(np_SCP_B_array, 'Bayesian (< 10 Myr)', SCP_B_complimits, axes[0, 0], 1, 2)
	plotaxes_histogram(np_SCP_B_array, '(10 - 100 Myr)', SCP_B_complimits, axes[0, 1], 2, 2)
	plotaxes_histogram(np_SCP_B_array, '(100 - 400 Myr)', SCP_B_complimits, axes[0, 2], 3, 2)
	plotaxes_histogram(np_SCP_C_array, r'$\chi^2$ (< 10 Myr)', SCP_C_complimits, axes[1, 0], 1, 2)
	plotaxes_histogram(np_SCP_C_array, '(10 - 100 Myr)', SCP_C_complimits, axes[1, 1], 2, 2)
	plotaxes_histogram(np_SCP_C_array, '(100 - 400 Myr)', SCP_C_complimits, axes[1, 2], 3, 2)
	fig.text(0.5, 0.05, r'log (M/M$_\odot$)', ha = 'center', fontsize = 45)
	fig.text(0.05, 0.5, r'log (dN/dM)', va = 'center', rotation = 'vertical', fontsize = 45)
	plt.xscale('log', nonposx = 'clip')
	plt.yscale('log', nonposy = 'clip')
	axes[0, 0].xaxis.set_major_formatter(log10_labels_format)
	axes[0, 0].yaxis.set_major_formatter(log10_labels_format)
	axes[0, 0].minorticks_off()
	plt.axis([np.power(10, 3), np.power(10, 7.5)] + [1E-7, 1E-1])
	plt.xticks([1E4, 1E5, 1E6, 1E7])
	plt.yticks([1E-6, 1E-4, 1E-2])
	plt.savefig('./FiguresSC/{}_SCP_AgeBins_03_Hist.png'.format(galname))
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> {}_SCP_AgeBins_04_EqualHist'.format(galname))
	print('>>>')
	fig, axes = plt.subplots(2, 3, sharex = 'all', sharey = 'all', figsize = (36, 24))
	fig.subplots_adjust(hspace = 0, wspace = 0, bottom = 0.15)
	plotaxes_equalhistogram(np_SCP_B_array, 'Bayesian (< 10 Myr)', SCP_B_complimits, axes[0, 0], 1, 2)
	plotaxes_equalhistogram(np_SCP_B_array, '(10 - 100 Myr)', SCP_B_complimits, axes[0, 1], 2, 2)
	plotaxes_equalhistogram(np_SCP_B_array, '(100 - 400 Myr)', SCP_B_complimits, axes[0, 2], 3, 2)
	plotaxes_equalhistogram(np_SCP_C_array, r'$\chi^2$ (< 10 Myr)', SCP_C_complimits, axes[1, 0], 1, 2)
	plotaxes_equalhistogram(np_SCP_C_array, '(10 - 100 Myr)', SCP_C_complimits, axes[1, 1], 2, 2)
	plotaxes_equalhistogram(np_SCP_C_array, '(100 - 400 Myr)', SCP_C_complimits, axes[1, 2], 3, 2)
	fig.text(0.5, 0.05, r'log (M/M$_\odot$)', ha = 'center', fontsize = 45)
	fig.text(0.05, 0.5, r'log (dN/dM)', va = 'center', rotation = 'vertical', fontsize = 45)
	plt.xscale('log', nonposx = 'clip')
	plt.yscale('log', nonposy = 'clip')
	axes[0, 0].xaxis.set_major_formatter(log10_labels_format)
	axes[0, 0].yaxis.set_major_formatter(log10_labels_format)
	axes[0, 0].minorticks_off()
	plt.axis([np.power(10, 3), np.power(10, 7.5)] + [1E-7, 1E-1])
	plt.xticks([1E4, 1E5, 1E6, 1E7])
	plt.yticks([1E-6, 1E-4, 1E-2])
	plt.savefig('./FiguresSC/{}_SCP_AgeBins_04_EqualHist.png'.format(galname))
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> {}_SCP_AgeBins_05A_EqualHist'.format(galname))
	print('>>>')
	fig, axes = plt.subplots(1, 2, sharex = 'all', sharey = 'all', figsize = (24, 12))
	fig.subplots_adjust(hspace = 0, wspace = 0, bottom = 0.15)
	plotaxes_equalhistogram(np_SCP_B_array, 'Bayesian (< 200 Myr)', [SCP_B_complimits[1], SCP_B_complimits[1], SCP_B_complimits[1]], axes[0], 5, 2)
	plotaxes_equalhistogram(np_SCP_C_array, r'$\chi^2$ (< 200 Myr)', [SCP_C_complimits[1], SCP_C_complimits[1], SCP_C_complimits[1]], axes[1], 5, 2)
	fig.text(0.5, 0.05, r'log (M/M$_\odot$)', ha = 'center', fontsize = 45)
	fig.text(0.05, 0.5, r'log (dN/dM)', va = 'center', rotation = 'vertical', fontsize = 45)
	plt.xscale('log', nonposx = 'clip')
	plt.yscale('log', nonposy = 'clip')
	axes[0].xaxis.set_major_formatter(log10_labels_format)
	axes[0].yaxis.set_major_formatter(log10_labels_format)
	axes[0].minorticks_off()
	plt.axis([np.power(10, 3), np.power(10, 7.5)] + [1E-7, 1E-1])
	plt.xticks([1E4, 1E5, 1E6, 1E7])
	plt.yticks([1E-6, 1E-4, 1E-2])
	plt.savefig('./FiguresSC/{}_SCP_AgeBins_05A_EqualHist.png'.format(galname))
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> {}_SCP_AgeBins_05B_MSF'.format(galname))
	print('>>>')
	fig, axes = plt.subplots(1, 2, sharex = 'all', sharey = 'all', figsize = (24, 12))
	fig.subplots_adjust(hspace = 0, wspace = 0, bottom = 0.15)
	plotaxes_mspecfit(np_SCP_B_array, 'Bayesian (< 200 Myr)', [SCP_B_complimits[1], SCP_B_complimits[1], SCP_B_complimits[1]], axes[0], 5, 0, 0)
	plotaxes_mspecfit(np_SCP_C_array, r'$\chi^2$ (< 200 Myr)', [SCP_C_complimits[1], SCP_C_complimits[1], SCP_C_complimits[1]], axes[1], 5, 0, 0)
	fig.text(0.5, 0.05, r'log (M/M$_\odot$)', ha = 'center', fontsize = 45)
	fig.text(0.05, 0.5, r'log N (> M)', va = 'center', rotation = 'vertical', fontsize = 45)
	plt.xscale('log', nonposx = 'clip')
	plt.yscale('log', nonposy = 'clip')
	axes[0].xaxis.set_major_formatter(log10_labels_format)
	axes[0].yaxis.set_major_formatter(log10_labels_format)
	axes[0].minorticks_off()
	plt.axis([np.power(10, 3), np.power(10, 7.5)] + [np.power(10, -0.5), np.power(10, 3.5)])
	plt.xticks([1E4, 1E5, 1E6, 1E7])
	plt.yticks([1E0, 1E1, 1E2, 1E3])
	plt.savefig('./FiguresSC/{}_SCP_AgeBins_05B_MSF.png'.format(galname))
	plt.close()

	return 0

def rungalaxyP_v1_1_output(galname, np_SCP_C_array, SCP_C_complimits):

	
	'''
	Function: Create PHANGS specific plots (V1.1)
	'''

	### --==--==--==-- ###
	print('>>>')
	print('>>> {}_SCP_AgeBins_00_AgeMass'.format(galname))
	print('>>>')
	fig, axes = plt.subplots(1, 1, sharex = 'all', sharey = 'all', figsize = (12, 12))
	fig.subplots_adjust(hspace = 0, wspace = 0, bottom = 0.15)
	plotaxes_agemass(np_SCP_C_array, '{}'.format(galname), SCP_C_complimits, axes, 1)
	fig.text(0.5, 0.05, r'log (Age/yr)', ha = 'center', fontsize = 45)
	fig.text(0.05, 0.5, r'log (M/M$_\odot$)', va = 'center', rotation = 'vertical', fontsize = 45)
	plt.xscale('log', nonposx = 'clip')
	plt.yscale('log', nonposy = 'clip')
	axes.xaxis.set_major_formatter(log10_labels_format)
	axes.yaxis.set_major_formatter(log10_labels_format)
	axes.minorticks_off()
	plt.axis([1E5, 1E9] + [1E2, 1E8])
	plt.xticks([1E6, 1E7, 1E8])
	plt.yticks([1E3, 1E5, 1E7])
	plt.savefig('./FiguresSC/{}_SCP_AgeBins_00_AgeMass.png'.format(galname))
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> {}_SCP_AgeBins_01_Like'.format(galname))
	print('>>>')
	fig, axes = plt.subplots(1, 3, sharex = 'all', sharey = 'all', figsize = (36, 12))
	fig.subplots_adjust(hspace = 0, wspace = 0, bottom = 0.15)
	plotaxes_likelihood(np_SCP_C_array, '{} (< 10 Myr)'.format(galname), SCP_C_complimits, axes[0], 1, 0, 0, 1, 'Test')
	plotaxes_likelihood(np_SCP_C_array, '(10 - 100 Myr)', SCP_C_complimits, axes[1], 2, 0, 0, 1, 'Test')
	plotaxes_likelihood(np_SCP_C_array, '(100 - 400 Myr)', SCP_C_complimits, axes[2], 3, [3.5, 2.75], 0, 1, 'Test')
	fig.text(0.5, 0.05, r'log (M$_*$/M$_\odot$)', ha = 'center', fontsize = 45)
	fig.text(0.05, 0.5, r'$-\beta$', va = 'center', rotation = 'vertical', fontsize = 45)
	plt.axis([3.0, 7.5] + [0, 3])
	plt.xticks([4.0, 5.0, 6.0, 7.0])
	plt.yticks([0.0, 1.0, 2.0])
	plt.savefig('./FiguresSC/{}_SCP_AgeBins_01_Like.png'.format(galname))
	plt.close()

	if False:

		### --==--==--==-- ###
		print('>>>')
		print('>>> {}_SCP_AgeBins_02_MSF'.format(galname))
		print('>>>')
		fig, axes = plt.subplots(1, 3, sharex = 'all', sharey = 'all', figsize = (36, 12))
		fig.subplots_adjust(hspace = 0, wspace = 0, bottom = 0.15)
		plotaxes_mspecfit(np_SCP_C_array, '{} (< 10 Myr)'.format(galname), SCP_C_complimits, axes[0], 1, 0, 0)
		plotaxes_mspecfit(np_SCP_C_array, '(10 - 100 Myr)', SCP_C_complimits, axes[1], 2, 0, 0)
		plotaxes_mspecfit(np_SCP_C_array, '(100 - 400 Myr)', SCP_C_complimits, axes[2], 3, 0, 0)
		fig.text(0.5, 0.05, r'log (M/M$_\odot$)', ha = 'center', fontsize = 45)
		fig.text(0.05, 0.5, r'log N (> M)', va = 'center', rotation = 'vertical', fontsize = 45)
		plt.xscale('log', nonposx = 'clip')
		plt.yscale('log', nonposy = 'clip')
		axes[0].xaxis.set_major_formatter(log10_labels_format)
		axes[0].yaxis.set_major_formatter(log10_labels_format)
		axes[0].minorticks_off()
		plt.axis([np.power(10, 3), np.power(10, 7.5)] + [np.power(10, -0.5), np.power(10, 3.5)])
		plt.xticks([1E4, 1E5, 1E6, 1E7])
		plt.yticks([1E0, 1E1, 1E2, 1E3])
		plt.savefig('./FiguresSC/{}_SCP_AgeBins_02_MSF.png'.format(galname))
		plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> {}_SCP_AgeBins_03_Hist'.format(galname))
	print('>>>')
	fig, axes = plt.subplots(1, 3, sharex = 'all', sharey = 'all', figsize = (36, 12))
	fig.subplots_adjust(hspace = 0, wspace = 0, bottom = 0.15)
	plotaxes_histogram(np_SCP_C_array, '{} (< 10 Myr)'.format(galname), SCP_C_complimits, axes[0], 1, 2)
	plotaxes_histogram(np_SCP_C_array, '(10 - 100 Myr)', SCP_C_complimits, axes[1], 2, 2)
	plotaxes_histogram(np_SCP_C_array, '(100 - 400 Myr)', SCP_C_complimits, axes[2], 3, 2)
	fig.text(0.5, 0.05, r'log (M/M$_\odot$)', ha = 'center', fontsize = 45)
	fig.text(0.05, 0.5, r'log (dN/dM)', va = 'center', rotation = 'vertical', fontsize = 45)
	plt.xscale('log', nonposx = 'clip')
	plt.yscale('log', nonposy = 'clip')
	axes[0].xaxis.set_major_formatter(log10_labels_format)
	axes[0].yaxis.set_major_formatter(log10_labels_format)
	axes[0].minorticks_off()
	plt.axis([np.power(10, 3), np.power(10, 7.5)] + [1E-7, 1E-1])
	plt.xticks([1E4, 1E5, 1E6, 1E7])
	plt.yticks([1E-6, 1E-4, 1E-2])
	plt.savefig('./FiguresSC/{}_SCP_AgeBins_03_Hist.png'.format(galname))
	plt.close()

	### --==--==--==-- ###
	print('>>>')
	print('>>> {}_SCP_AgeBins_04_EqualHist'.format(galname))
	print('>>>')
	fig, axes = plt.subplots(1, 3, sharex = 'all', sharey = 'all', figsize = (36, 12))
	fig.subplots_adjust(hspace = 0, wspace = 0, bottom = 0.15)
	plotaxes_equalhistogram(np_SCP_C_array, '{} (< 10 Myr)'.format(galname), SCP_C_complimits, axes[0], 1, 2)
	plotaxes_equalhistogram(np_SCP_C_array, '(10 - 100 Myr)', SCP_C_complimits, axes[1], 2, 2)
	plotaxes_equalhistogram(np_SCP_C_array, '(100 - 400 Myr)', SCP_C_complimits, axes[2], 3, 2)
	fig.text(0.5, 0.05, r'log (M/M$_\odot$)', ha = 'center', fontsize = 45)
	fig.text(0.05, 0.5, r'log (dN/dM)', va = 'center', rotation = 'vertical', fontsize = 45)
	plt.xscale('log', nonposx = 'clip')
	plt.yscale('log', nonposy = 'clip')
	axes[0].xaxis.set_major_formatter(log10_labels_format)
	axes[0].yaxis.set_major_formatter(log10_labels_format)
	axes[0].minorticks_off()
	plt.axis([np.power(10, 3), np.power(10, 7.5)] + [1E-7, 1E-1])
	plt.xticks([1E4, 1E5, 1E6, 1E7])
	plt.yticks([1E-6, 1E-4, 1E-2])
	plt.savefig('./FiguresSC/{}_SCP_AgeBins_04_EqualHist.png'.format(galname))
	plt.close()

	return 0

#------------------------------------------------------------------------------
###
# (13) Code Snippets (Analysis III)
###


def calccorrectionfactor(array1, complimits):

	'''
	Function: Calculated correction factor for age limits
	'''

	totalmass = np.sum(array1)
	print('The total mass in the synthetic cluster catalog is: {:.3e}'.format(totalmass))

	correctionfactorout = []
	for i in range(0, len(complimits)):

		restrictedmass = np.sum(array1[array1 > complimits[i]])
		correctionfactor = restrictedmass / totalmass
		correctionfactorout.append(correctionfactor)
		print('Output: {:.3e}, {:.3f}'.format(restrictedmass, correctionfactor))

	if len(correctionfactorout) == 1:
		correctionfactorout = correctionfactorout[0]

	return correctionfactorout

def calccorrectionfactor1(array1, lowlimit, output = False):

	'''
	Function: Calculated correction factor for age limits
	'''

	totalmass = np.sum(array1)
	restrictedmass = np.sum(array1[array1 > lowlimit])
	correctionfactor = restrictedmass / totalmass
	
	if output == True:
		print('The total mass in the synthetic cluster catalog is: {:.3e}'.format(totalmass))
		print('The min/max mass in the synthetic cluster catalog is: {:.3e}/{:.3e}'.format(np.nanmin(array1), np.nanmax(array1)))
		print('Output (1): Cut at {:.2f} - Correction Factor {:.3f}'.format(np.log10(lowlimit), correctionfactor))

	return correctionfactor
 
def calccorrectionfactor2(array1, lowlimit, highlimit, output = False):

	'''
	Function: Calculated correction factor for age limits
	'''

	array1_belowlimit = array1[array1 < highlimit]
	totalmass = np.sum(array1_belowlimit)
	restrictedmass = np.sum(array1[array1 > lowlimit])
	correctionfactor = restrictedmass / totalmass

	if output == True:
		print('The total mass in the synthetic cluster catalog is: {:.3e} below log M = {:.2f}'.format(totalmass, np.log10(highlimit)))
		print('The min mass in the synthetic cluster catalog is: {:.3e}'.format(np.nanmin(array1)))
		print('Output (2): Cut at {:.2f} - Correction Factor {:.3f}'.format(np.log10(lowlimit), correctionfactor))

	return correctionfactor

def calculategamma_oneage(gal_array, complimit, corrfactor, sfrc, galname):

	'''
	Function: Calculate Gamma for one age
	'''

	# Create age bins
	gal_array_age3_tmp = gal_array[gal_array[:,6] > 100.01*1E6]
	gal_array_age3 = gal_array_age3_tmp[gal_array_age3_tmp[:,6] <= 400.01*1E6]
	gal_array_age3_masslimit = gal_array_age3[gal_array_age3[:,4] > complimit]
	totalmass_age3 = np.sum(gal_array_age3_masslimit[:,4])
	totalmasscorr_age3 = totalmass_age3 / corrfactor
	cfr_age3 = totalmasscorr_age3 / ((400 * 1E6) - 1E8)
	gamma_age3 = cfr_age3 / sfrc
	print('{} - Gamma = {:.3f}'.format(galname, gamma_age3))

def calculategamma(gal_array, complimits, corrfactorarray, sfrc, galname):

	'''
	Function: Calculate Gamma with input corrfactor
	'''
	gal_array_age1 = gal_array[gal_array[:,6] <= 10.01*1E6]
	gal_array_age1_masslimit = gal_array_age1[gal_array_age1[:,4] > complimits[0]]
	
	gal_array_age2_tmp = gal_array[gal_array[:,6] > 10.01*1E6]
	gal_array_age2 = gal_array_age2_tmp[gal_array_age2_tmp[:,6] <= 100.01*1E6]
	gal_array_age2_masslimit = gal_array_age2[gal_array_age2[:,4] > complimits[1]]

	gal_array_age3_tmp = gal_array[gal_array[:,6] > 100.01*1E6]
	gal_array_age3 = gal_array_age3_tmp[gal_array_age3_tmp[:,6] <= 400.01*1E6]
	gal_array_age3_masslimit = gal_array_age3[gal_array_age3[:,4] > complimits[2]]
	
	gal_array_age5_tmp = gal_array[gal_array[:,6] > 0.99*1E6]
	gal_array_age5 = gal_array_age5_tmp[gal_array_age5_tmp[:,6] <= 200.01*1E6]
	gal_array_age5_masslimit = gal_array_age5[gal_array_age5[:,4] > complimits[2]]

	print('------ ' + galname.replace('_', '\\_') + ' [$<$ 10, 10 - 100, 100 - 400, 1 - 200]\\\\')
	print('Num: [{}, {}, {}, {}] \\\\'.format(len(gal_array_age1_masslimit), len(gal_array_age2_masslimit), len(gal_array_age3_masslimit), len(gal_array_age5_masslimit)))
	totalmass_age1 = np.sum(gal_array_age1_masslimit[:,4])
	totalmass_age2 = np.sum(gal_array_age2_masslimit[:,4])
	totalmass_age3 = np.sum(gal_array_age3_masslimit[:,4])
	totalmass_age5 = np.sum(gal_array_age5_masslimit[:,4])
	print('Total Masses: [{:.2e}, {:.2e}, {:.2e}, {:.2e}] \\\\'.format(totalmass_age1, totalmass_age2, totalmass_age3, totalmass_age5))
	totalmasscorr_age1 = totalmass_age1 / corrfactorarray[0]
	totalmasscorr_age2 = totalmass_age2 / corrfactorarray[1]
	totalmasscorr_age3 = totalmass_age3 / corrfactorarray[2]
	totalmasscorr_age5 = totalmass_age5 / corrfactorarray[2]
	print('Total Masses (corrected): [{:.2e}, {:.2e}, {:.2e}, {:.2e}] \\\\'.format(totalmasscorr_age1, totalmasscorr_age2, totalmasscorr_age3, totalmasscorr_age5))
	cfr_age1 = totalmasscorr_age1 / (9*1E6)
	cfr_age2 = totalmasscorr_age2 / (1E8 - 1E7)
	cfr_age3 = totalmasscorr_age3 / ((400 * 1E6) - 1E8)
	cfr_age5 = totalmasscorr_age5 / ((200 * 1E6) - 1E6)
	print('CFR: [{:.2e}, {:.2e}, {:.2e}, {:.2e}] \\\\'.format(cfr_age1, cfr_age2, cfr_age3, cfr_age5))
	print('SFR (input): [{:.3f}, {:.3f}, {:.3f}, {:.3f}] \\\\'.format(sfrc[0], sfrc[1], sfrc[2], sfrc[1]))
	gamma_age1 = cfr_age1 / sfrc[0]
	gamma_age2 = cfr_age2 / sfrc[1]
	gamma_age3 = cfr_age3 / sfrc[2]
	gamma_age5 = cfr_age5 / sfrc[1]
	print('Gamma: [{:.3f}, {:.3f}, {:.3f}, {:.3f}] \\\\'.format(gamma_age1, gamma_age2, gamma_age3, gamma_age5))

def read_powerlaw_contlog_nodest():

	'''
	Function: 
	'''
	
	# print('>>> Running - {}'.format(inspect.currentframe().f_code.co_name))

	powerlaw_contlog_nodest_file = open('./SimSCOutput/powerlaw_contlog_nodest.txt', 'r')
	powerlaw_contlog_nodest_array = []
	###
	i = 1
	while 1:
		powerlaw_contlog_nodest_line = powerlaw_contlog_nodest_file.readline()
		if not powerlaw_contlog_nodest_line:
			break
		else:
			powerlaw_contlog_nodest_cols = powerlaw_contlog_nodest_line.split(';')
			testvar = 2
			if len(powerlaw_contlog_nodest_cols) != testvar:
				print('Warning - Length of Data {} != {}'.format(len(powerlaw_contlog_nodest_cols), testvar))
			else:
				arrayentry = ['powerlaw_contlog_nodest', i, -1, -1, float(powerlaw_contlog_nodest_cols[0]), -1, float(powerlaw_contlog_nodest_cols[1]), -1]
				powerlaw_contlog_nodest_array.append(arrayentry)
			i = i + 1

	np_powerlaw_contlog_nodest_array = converttonumpy(powerlaw_contlog_nodest_array, 'powerlaw_contlog_nodest', output = False)

	return np_powerlaw_contlog_nodest_array

def calculategamma_v2(gal_array, complimits, sfrc, galname, alt_limit = False):

	'''
	Function: Calculate Gamma with calculated corrfactor
	'''

	# Read in array
	np_powerlaw_contlog_nodest_array = read_powerlaw_contlog_nodest()
	corrfactor1 = calccorrectionfactor1(np_powerlaw_contlog_nodest_array[:,4], complimits[0])
	corrfactor2 = calccorrectionfactor1(np_powerlaw_contlog_nodest_array[:,4], complimits[1])
	corrfactor3 = calccorrectionfactor1(np_powerlaw_contlog_nodest_array[:,4], complimits[2])

	if alt_limit == True:

		print ('-> Check: Alt Endpoints == True')
		gal_array_age1 = gal_array[gal_array[:,6] < 10.01*1E6]
		gal_array_age1_masslimit = gal_array_age1[gal_array_age1[:,4] > complimits[0]]

		gal_array_age2_tmp = gal_array[gal_array[:,6] >= 10.01*1E6]
		gal_array_age2 = gal_array_age2_tmp[gal_array_age2_tmp[:,6] < 100.01*1E6]
		gal_array_age2_masslimit = gal_array_age2[gal_array_age2[:,4] > complimits[1]]

		gal_array_age3_tmp = gal_array[gal_array[:,6] >= 100.01*1E6]
		gal_array_age3 = gal_array_age3_tmp[gal_array_age3_tmp[:,6] <= 400.01*1E6]
		gal_array_age3_masslimit = gal_array_age3[gal_array_age3[:,4] > complimits[2]]

	else:

		gal_array_age1 = gal_array[gal_array[:,6] <= 10.01*1E6]
		gal_array_age1_masslimit = gal_array_age1[gal_array_age1[:,4] > complimits[0]]

		gal_array_age2_tmp = gal_array[gal_array[:,6] > 10.01*1E6]
		gal_array_age2 = gal_array_age2_tmp[gal_array_age2_tmp[:,6] <= 100.01*1E6]
		gal_array_age2_masslimit = gal_array_age2[gal_array_age2[:,4] > complimits[1]]

		gal_array_age3_tmp = gal_array[gal_array[:,6] > 100.01*1E6]
		gal_array_age3 = gal_array_age3_tmp[gal_array_age3_tmp[:,6] <= 400.01*1E6]
		gal_array_age3_masslimit = gal_array_age3[gal_array_age3[:,4] > complimits[2]]


	print('------ ' + galname.replace('_', '\\_') + ' [< 10, 10 - 100, 100 - 400]')
	print('Limits: [{:.1f}, {:.1f}, {:.1f}]'.format(np.log10(complimits[0]), np.log10(complimits[1]), np.log10(complimits[2])))
	print('Corr Factor: [{:.3f}, {:.3f}, {:.3f}]'.format(corrfactor1, corrfactor2, corrfactor3))
	print('Num: [{}, {}, {}]'.format(len(gal_array_age1_masslimit), len(gal_array_age2_masslimit), len(gal_array_age3_masslimit)))
	totalmass_age1 = np.sum(gal_array_age1_masslimit[:,4])
	totalmass_age2 = np.sum(gal_array_age2_masslimit[:,4])
	totalmass_age3 = np.sum(gal_array_age3_masslimit[:,4])
	print('Total Masses: [{:.2e}, {:.2e}, {:.2e}]'.format(totalmass_age1, totalmass_age2, totalmass_age3))
	totalmasscorr_age1 = totalmass_age1 / corrfactor1
	totalmasscorr_age2 = totalmass_age2 / corrfactor2
	totalmasscorr_age3 = totalmass_age3 / corrfactor3
	print('Total Masses (corrected): [{:.2e}, {:.2e}, {:.2e}]'.format(totalmasscorr_age1, totalmasscorr_age2, totalmasscorr_age3))
	cfr_age1 = totalmasscorr_age1 / (9*1E6)
	cfr_age2 = totalmasscorr_age2 / (1E8 - 1E7)
	cfr_age3 = totalmasscorr_age3 / ((400 * 1E6) - 1E8)
	print('CFR: [{:.2e}, {:.2e}, {:.2e}]'.format(cfr_age1, cfr_age2, cfr_age3))
	print('SFR (input): [{:.3f}, {:.3f}, {:.3f}]'.format(sfrc[0], sfrc[1], sfrc[2]))
	gamma_age1 = cfr_age1 / sfrc[0]
	gamma_age2 = cfr_age2 / sfrc[1]
	gamma_age3 = cfr_age3 / sfrc[2]
	print('Gamma: [{:.3f}, {:.3f}, {:.3f}]'.format(gamma_age1, gamma_age2, gamma_age3))