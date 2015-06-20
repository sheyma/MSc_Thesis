#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
from math import factorial, sqrt, ceil
import matplotlib.pyplot as pl	
from matplotlib.pyplot import FormatStrFormatter
import random as rnd
import sys  
import glob
import os
import scipy.stats as sistat
import collections
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata	
from matplotlib import cm

# check the loaded matrix if it is symmetric
def load_matrix(file):
	A  = np.loadtxt(file, unpack=True)
	AT = np.transpose(A)
	# check the symmetry				
	if A.shape[0] != A.shape[1] or not (A == AT).all():
		print "error: loaded matrix is not symmetric"
		raise ValueError
	return AT


def corr_histo(corr_matrix):
		
	#corr_flat = []
	## removing the diagonal elements in both matrices to get
	## more reasonable Pearson correlation coef. between them
	#for i in range(0, np.shape(corr_matrix)[0]):
		#for j in range(0, np.shape(corr_matrix)[1]):
			#if (i == j) :

				#tmp_a  = np.append(corr_matrix[i, 0:j], corr_matrix[i, j+1:])
				#corr_flat  = np.append(corr_flat, tmp_a)

	# merge 2D corr_matrix into 1D numpy array by flatten
	corr_flat = np.ndarray.flatten(corr_matrix) 
	corr_max  = 1.0
	corr_min  = -1.0
	bin_nu    = 100
	# a normalized histogram is obtained
	hist, bin_edges = np.histogram(corr_flat, bins=bin_nu, range=[corr_min, corr_max], normed =True)#, histtype='bar')
	
	#hist = pl.hist(corr_flat, bins=bin_nu, range=[corr_min, corr_max], normed =True, histtype='bar')
	#pl.title(simfile)
	# type(hist) = <type 'tuple'> and len(hist) = 3
	# y_axis : hist[0] , normalized hist values
	# x_axis : hist[1] , start points of bins 
	return hist

def plot_histog(corr_matrix, STRING):
	corr_flat = np.ndarray.flatten(corr_matrix) 
	corr_max  = float(1.0)
	corr_min  = float(-1.0)
	bin_nu    = 100
	# a normalized histogram is obtained
	pl.hist(corr_flat, bins=bin_nu, range=[corr_min, corr_max], normed =True, histtype='bar', align='mid')
	pl.xlim(corr_min-0.01, corr_max+0.01)
	pl.ylim(0.0, 25)	
	pl.xticks(fontsize=20)
	pl.yticks(fontsize=20)
	
	pl.text(-0.5, 10, STRING,
				horizontalalignment='center',
				verticalalignment='center', fontsize=50)

	#pl.xlabel('$\\rho$', fontsize=25)

# intersection method to compare two histograms
# HA and HB must be normalized histograms
def intersec_hists(HA, HB):
	minsum  = 0
	for i in range(0, len(HA)) :
		minsum = minsum + min(HA[i], HB[i])
	# normalize minsum 
	minsum  = float(minsum) /  sum(HB) 
	return minsum
	
def correl_hists(HA, HB):
	N  = len(HA)
	HA_bar  = sum(HA) / float(N)
	HB_bar  = sum(HB) / float(N)
	tmp1    = 0 
	tmp2    = 0
	tmp3	= 0
	for i in range(0, N) : 
		tmp1 = tmp1 + (HA[i] - HA_bar) * (HB[i] - HB_bar)
		tmp2 = tmp2 + pow((HA[i] - HA_bar) , 2)
		tmp3 = tmp3 + pow((HB[i] - HB_bar) , 2)
	d = float(tmp1) / sqrt(tmp2 * tmp3)
	return d 
	
# Bhattacharyya method to compare two histograms
# HA and HB must be normalized histograms
def bhatta_hists(HA, HB):
	N  = len(HA)
	HA_bar  = sum(HA) / float(N)
	HB_bar  = sum(HB) / float(N)
	S1      = 1./ sqrt(HA_bar*HB_bar*N*N)
	S2      = 0
	for i in range(0, N) :
		S2 = S2 + sqrt(HA[i]*HB[i])
	
	S3  = sqrt(1 - S1*S2)	
	return S3
	

def chi2_hists(HA, HB):
	# compute the chi-squared distance
	squsum  = 0
	eps = 1e-10
	for i in range(0, len(HA)) :
		tmp1 = pow((HA[i] - HB[i]) , 2) 
		tmp2 = HA[i] + HB[i]
		if tmp2 == 0 :
			tmp2 = eps
		squsum = squsum + tmp1 / float(tmp2)
	return squsum	

# comparing histohrams with Bhatta tool in parameter space
def compare_hist(name_A, name_B, pathway, THR, SIG):
	R_thr = {}
	
	for THR in thr_array :
		R_temp = []

		#for VEL in vel_array :
			#input_empiri = name_A[0:18] + str(THR) + name_A[20:40] + str(VEL) + name_A[42:]		
			#input_simuli = name_B[0:18] + str(THR) + name_B[20:40] + str(VEL) + name_B[42:]

		for SIG in sig_array :
			input_A = name_A[0:18] + str(THR) + name_A[20:27] + str(SIG) + name_A[30:]
			input_B = name_B[0:18] + str(THR) + name_B[20:27] + str(SIG) + name_B[30:]			
			try:
				mtx_A = load_matrix(pathway + input_A)
				HistA = corr_histo(mtx_A)			
				have_A = True		
			except:
				have_A = False
				print "not found A:", input_A
			try:
				mtx_B = load_matrix(pathway + input_B)
				HistB = corr_histo(mtx_B)			
				have_B = True
			except:
				have_B = False
				print "not found B:", input_B
			
			if have_A and have_B:
				R_val = bhatta_hists(HistA, HistB)
			else:
				R_val = np.nan
			R_temp = np.append(R_temp, R_val)		
		
		R_thr[THR] 	   = np.array(R_temp)
	Ordered_R = collections.OrderedDict(sorted(R_thr.items()))	
	datam = np.array(Ordered_R.values())

	return datam

local_path   = '../data/jobs_corr/'	
name_E  = 'acp_w_0_ADJ_thr_0.54_sigma=0.3_D=0.05_v=30.0_tmax=45000_FHN_corr.dat'
name_Ra = 'acp_w_a_ADJ_thr_0.54_sigma=0.3_D=0.05_v=30.0_tmax=45000_FHN_corr.dat'

#local_path   = '../data/jobs_corr_bold/'	
#name_E  = 'acp_w_0_ADJ_thr_0.54_sigma=0.3_D=0.05_v=30.0_tmax=45000_NORM_BOLD_signal_corr.dat'
#name_Ra = 'acp_w_a_ADJ_thr_0.54_sigma=0.3_D=0.05_v=30.0_tmax=45000_NORM_BOLD_signal_corr.dat'

thr_array = np.arange(34, 86, 4)
print thr_array
sig_array = np.array([0.1,  0.050, 0.045, 0.040, 0.035, 0.030, 0.025, 0.020,  0.015, 0.010, 0.005 ])
#vel_array = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100, 110])

datam_a = compare_hist(name_E, name_Ra, local_path, thr_array, sig_array)

## Parameter Space Plot 
#fig , ax = pl.subplots(figsize=(15,12))
#pl.subplots_adjust(left=0.20, right=1.00, top=0.95, bottom=0.15)

#pl.subplot(1,1,1)
#pl.imshow(np.transpose(datam_a), interpolation='nearest', vmin= 0.0, vmax = 0.5, cmap='jet', aspect='auto')
#cbar = pl.colorbar()

#pl.ylabel('$c$', fontsize=65)
#pl.xlabel('$p$', fontsize = 65)

#a = np.array([0.38, 0.50, 0.62, 0.74])	
#b = sig_array

#separ_xthick = ceil(float(len(thr_array))/len(a)) -1
#print separ_xthick
#pl.xticks(np.arange(1,len(thr_array), separ_xthick),  a)
#pl.yticks(np.arange(0,len(b),1),  b)

#print thr_array

#for t in cbar.ax.get_yticklabels():
	#t.set_fontsize(50)
#pl.xticks(fontsize = 50)
#pl.yticks(fontsize = 50)
#ax.tick_params('both', length=12, width=5, which='major')
#pl.show()

# Single Histogram Plot
O = load_matrix(local_path + 'acp_w_0_ADJ_thr_0.54_sigma=0.005_D=0.05_v=30.0_tmax=45000_FHN_corr.dat')
A = load_matrix(local_path + 'acp_w_a_ADJ_thr_0.54_sigma=0.005_D=0.05_v=30.0_tmax=45000_FHN_corr.dat')

#O = load_matrix(local_path + 'acp_w_0_ADJ_thr_0.34_sigma=0.005_D=0.05_v=30.0_tmax=45000_NORM_BOLD_signal_corr.dat')
#A = load_matrix(local_path + 'acp_w_a_ADJ_thr_0.34_sigma=0.005_D=0.05_v=30.0_tmax=45000_NORM_BOLD_signal_corr.dat')


fig, ax = pl.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(25,12))
pl.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
fig.text(0.04, 0.5, 'number of pair of nodes', va='center', rotation='vertical',fontsize=45)
pl.subplot(1,2,1)
plot_histog(O, '$R_{BG}$')
pl.xticks(fontsize = 45)
pl.yticks(fontsize = 45)

pl.xlabel('$\\rho$', fontsize=55 )
pl.subplot(1,2,2)
plot_histog(A, '$R_{ER}$')
pl.xlabel('$\\rho$', fontsize=55 )
pl.xticks(fontsize = 45)
pl.yticks(fontsize = 45)

pl.show()

