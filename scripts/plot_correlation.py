#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

""" 
	input  : empirical matrix (e.g. fMRI-BOLD data), simulation outcome
	matrix (FHN or BOLD simulations)
	
	intermediate process : loading empirical and simulation matrices, 
	plotting color coded empirical mtx., calculating Pearson's correlat.
	coefficient between empirical and simulated matrices, parameter
	analysis plots for the Pearson coefficients over velocity-threshold 
	or sigma-threshold changing parameter regions 

	output : R_pearson, figures
"""
import networkx as nx
import numpy as np
from math import factorial 
import matplotlib.pyplot as pl	
from matplotlib import colors
from matplotlib.pyplot import FormatStrFormatter
import random as rnd
import sys  
import glob
import os
import scipy.stats as sistat
import collections
from matplotlib import rc
import copy

# check the loaded matrix if it is symmetric
def load_matrix(file):
	A  = np.loadtxt(file, unpack=True)
	AT = np.transpose(A)
	# check the symmetry				
	if A.shape[0] != A.shape[1] or not (A == AT).all():
		print "error: loaded matrix is not symmetric"
		raise ValueError
	return AT
	
# plots colorbar coded EMRPIRICAL correlation matrix 
def plot_corr(corr_matrix, simfile ):	
	N_col  = np.shape(corr_matrix)[1]
	extend = (0.5 , N_col+0.5 , N_col+0.5, 0.5 )
	fig , ax = pl.subplots(figsize=(15, 12))
	ax.tick_params('both', length=15, width=8, which='major')
	pl.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.12)	
	for i in range(0,N_col):
		for j in range(0, N_col):
			if i==j:
				corr_matrix[i,j] = 1
			
	#pl.imshow(corr_matrix, interpolation='nearest', extent=extend)
	pl.imshow(corr_matrix, interpolation='nearest', vmin=0, vmax=1.0, extent=extend)
	
	## vmin & vmax for EMPIRICAL data corr matrix :
	#pl.imshow(corr_matrix, interpolation='nearest', vmin=0.0, vmax=1.0, extent=extend)
	
	cbar = pl.colorbar()
	for t in cbar.ax.get_yticklabels():
		t.set_fontsize(50)
	pl.xticks(fontsize = 50)
	pl.yticks(fontsize = 50)
	pl.xlabel('Nodes', fontsize = 50)
	pl.ylabel('Nodes', fontsize = 50)
	return  	

# plots ADJACENCY matrix black - white 
def plot_adj(corr_matrix, simfile ):

	N_col  = np.shape(corr_matrix)[1]
	
	for i in range(0,N_col):
		for j in range(0, N_col):
			if i==j:
				corr_matrix[i,j] = 1
				print i,j

	extend = (0.5 , N_col+0.5 , N_col+0.5, 0.5 )
	fig , ax = pl.subplots(figsize=(15, 12))
	ax.tick_params('both', length=15, width=8, which='major')
	pl.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.12)				
	cmap = colors.ListedColormap(['white', 'black'])
	bounds=[0, 0.5, 1]
	norm = colors.BoundaryNorm(bounds, cmap.N)

	img = pl.imshow(corr_matrix, interpolation='nearest', vmin=0, vmax=1.0, extent=extend, cmap=cmap, norm=norm)
	cbar = pl.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[0, 1])
	
	for t in cbar.ax.get_yticklabels():
		t.set_fontsize(50)
	pl.xticks(fontsize = 50)
	pl.yticks(fontsize = 50)
	pl.xlabel('Nodes', fontsize = 50)
	pl.ylabel('Nodes', fontsize = 50)

	return  


# calculating Pearson's corr. coef. for two given matrices
# matrix_A : simulated neuronal activity correlations, 1's on diagonal
# matrix_B : empirical fMRI-BOLD correlations, 0's on diagonal	
def pearson_coef(matrix_A , matrix_B):
	vec_a = []
	vec_b = []
	
	for i in range(0, np.shape(matrix_A)[0]):
		
		## convert matrices into vectors without any manipulation
		#tmp_a  = (matrix_A[i, :])
		#vec_a  = np.append(vec_a, tmp_a)
		#print "dhpe : ", np.shape(vec_a)

		#tmp_b  = matrix_B[i,:]
		#vec_b  = np.append(vec_b, tmp_b)	
		
		# removing the diagonal elements in both matrices to get
		# more reasonable Pearson correlation coef. between them
		for j in range(0, np.shape(matrix_A)[1]):
			if i == j:
		 
				tmp_a  = np.append(matrix_A[i, 0:j], matrix_A[i, j+1:])
				vec_a  = np.append(vec_a, tmp_a)
				
				tmp_b  = np.append(matrix_B[i, 0:j], matrix_B[i, j+1:])
				vec_b  = np.append(vec_b, tmp_b)	
		
		## assigning 1's to the diagonal of the empirical matrix
		#for j in range(0, np.shape(matrix_A)[1]):
			#if i == j:
				
				#matrix_A[i,j] = 1
				#vec_a  = np.append(vec_a, matrix_A[i, :])
				
				#vec_b  = np.append(vec_b, matrix_B[i, :])
		
	[R_pearson , p_value] = sistat.pearsonr(vec_a , vec_b)
	return R_pearson	
	

if __name__ == '__main__':
	usage = 'Usage: %s method correlation_matrix [threshold]' % sys.argv[0]
	try:
		input_empiri = sys.argv[1]
		input_simuli = sys.argv[2]
	except:
		print usage
		sys.exit(1)
		
# fMRI-BOLD input matrix load , this is a correlation matrix already
mtx_empiri		=		load_matrix(input_empiri)

# loading correl. mtx. of fhn time series (output of correlation_fhn.py)
#name = 'A_aal_0_ADJ_thr_0.54_sigma=0.2_D=0.05_v=70.0_tmax=45000_FHN_corr.dat'
#name = 'acp_w_0_ADJ_thr_0.54_sigma=0.3_D=0.05_v=60.0_tmax=45000_FHN_corr.dat'

#name = 'A_aal_0_ADJ_thr_0.60_sigma=0.1_D=0.05_v=70.0_tmax=45000_NORM_BOLD_signal_corr.dat'
name = 'acp_w_0_ADJ_thr_0.60_sigma=0.1_D=0.05_v=30.0_tmax=45000_NORM_BOLD_signal_corr.dat'

#thr_array = np.array([54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66])	
#vel_array = np.array([150, 140, 130, 120, 110, 90, 80, 70, 60, 50, 40, 30])
#sig_array = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.018 ])

thr_array = np.arange(18, 86, 4)
vel_array = np.array([110, 90, 80, 70, 60, 50, 40, 30, 20])
#sig_array = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.018 ])
sig_array = np.array([0.1, 0.050, 0.045, 0.040, 0.030, 0.025, 0.020, 0.018, 0.015, 0.010, 0.005 ])

R_thr =  {}

for THR in thr_array :
	R_temp = []
	
	#local_path = '../data/jobs_corr/'
	local_path = '../data/jobs_corr_bold/'
	
	#for VEL in vel_array :
		
		#input_name = name[0:18] + str(THR) + name[20:40] + str(VEL) + name[42:]		
		
	for SIG in sig_array :
		
		input_name = name[0:18] + str(THR) + name[20:27] + str(SIG) + name[30:]		
		
		print input_name
	
		try:
			mtx_simuli = load_matrix(local_path + input_name)
		except :
			R_val      = np.nan
		else :
			R_val      = pearson_coef(mtx_empiri, mtx_simuli)
	
		
		R_temp     = np.append(R_temp, R_val)
	R_thr[THR] 	   = np.array(R_temp)


	
Ordered_R   = collections.OrderedDict(sorted(R_thr.items()))	
#print "Ordered dict"
#print Ordered_R

datam 		= np.array(Ordered_R.values())

#print "check its numpy array version"
#print datam

# PLOTTING BEGINS ! 

fig , ax = pl.subplots(figsize=(20, 15))
pl.subplots_adjust(left=0.15, right=1.0, top=0.98, bottom=0.12)
pl.imshow(np.transpose(datam), interpolation='nearest', cmap='jet', aspect='auto')
cbar = pl.colorbar()

#PLOT PA OVER SIGMA
b = sig_array
pl.ylabel('$c$', fontsize=65)

##PLOT PA OVER VELOCITY
#b = vel_array/10
#pl.ylabel('$v$ [m/s]', fontsize=65)


# acp_w_0 thr range, XTICKS: 
a = np.array([0.22, 0.34, 0.46, 0.58, 0.70, 0.82])
pl.setp(ax , xticks=np.arange(1,len(thr_array),3), xticklabels = a )
pl.xlabel('$p$', fontsize = 65)

## A_aal_0 thr range, XTICKS : 
#a = np.array([0.55,  0.58, 0.61, 0.64])	
#pl.setp(ax , xticks=np.arange(1,len(thr_array),3), xticklabels = a )
#pl.xlabel('$r$', fontsize = 65)

pl.setp(ax , yticks=np.arange(0,len(b),1), yticklabels = b)

for t in cbar.ax.get_yticklabels():
	t.set_fontsize(50)
pl.xticks(fontsize = 50)
pl.yticks(fontsize = 50)
ax.tick_params('both', length=12, width=5, which='major')
pl.show()		

#------------------------------------
#mtx_empiri			= 		load_matrix(input_empiri)
#pl.figure()		
#figure				=		plot_corr(mtx_empiri , input_empiri)
# plot_adj gets and "adjacency matrix"
#figure				=		plot_adj(mtx_empiri , input_empiri)
#mtx_simuli			= 		load_matrix(input_simuli)
#pl.figure()
#figure				=		plot_corr(mtx_simuli , input_simuli)

#R_pearson			= 	    pearson_coef(mtx_empiri , mtx_simuli)
#print "Pearson corr. coef. between empir.-simul. : " , R_pearson
#pl.show()
