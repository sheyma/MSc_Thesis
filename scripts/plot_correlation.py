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
from matplotlib.pyplot import FormatStrFormatter
import random as rnd
import sys  
import glob
import os
import scipy.stats as sistat
import collections

# check the loaded matrix if it is symmetric
def load_matrix(file):
	A  = np.loadtxt(file, unpack=True)
	AT = np.transpose(A)
	# check the symmetry				
	if A.shape[0] != A.shape[1] or not (A == AT).all():
		print "error: loaded matrix is not symmetric"
		raise ValueError
	return AT
	
# plots colorbar coded EMRPIRICAL correlation matrix (fMRI-BOLD data)
def plot_corr(corr_matrix, simfile ):	
	N_col  = np.shape(corr_matrix)[1]
	extend = (0.5 , N_col+0.5 , N_col+0.5, 0.5 )	
	pl.imshow(corr_matrix, interpolation='nearest', extent=extend)
	cbar = pl.colorbar()
	for t in cbar.ax.get_yticklabels():
		t.set_fontsize(50)
	pl.xticks(fontsize = 50)
	pl.yticks(fontsize = 50)
	pl.suptitle('FCM (BOLD-fMRI)', fontsize= 50)
	pl.xlabel('Nodes', fontsize = 50)
	pl.ylabel('Nodes', fontsize = 50)
	#image_name = simfile[0:-4] + '_CORR.eps'	
	#pl.savefig(simfile[0:-4]+'.eps', format="eps")
	#pl.show()
	return  	

def corr_histo(corr_matrix, simfile):
	# merge 2D corr_matrix into 1D numpy array by flatten
	corr_flat = np.ndarray.flatten(corr_matrix) 
	corr_max  = 1.0
	corr_min  = -1.0
	bin_nu    = 20
	# a normalized histogram is obtained
	hist = pl.hist(corr_flat, bins=bin_nu, range=[corr_min, corr_max], normed =True, histtype='bar')
	pl.title(simfile)
	# type(hist) = <type 'tuple'> and len(hist) = 3
	# y_axis : hist[0] , normalized hist values
	# x_axis : hist[1] , start points of bins 
	return hist

def intersec_hists(HA, HB):
	y_axis    = 0
	HA_norm = HA[y_axis]
	HB_norm = HB[y_axis]
	minsum  = 0
	for i in range(0, len(HA_norm)) :
		minsum = minsum + min(HA_norm[i], HB_norm[i])
	#minsum  = float(minsum) / (min( sum(HA_norm), sum(HB_norm) ))
	minsum  = float(minsum) /  sum(HA_norm) 
	print "minsum : ", minsum
	return minsum

def chi2_distance(histA, histB, eps = 1e-10):
	# compute the chi-squared distance
	d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
	for (a, b) in zip(histA, histB)])
 
	# return the chi-squared distance
	return d


def plot_hist(corr_matrix, simfile ):	
	N_col  = np.shape(corr_matrix)[1]
	extend = (0.5 , N_col+0.5 , N_col+0.5, 0.5 )	
	pl.imshow(corr_matrix, interpolation='nearest', extent=extend)
	cbar = pl.colorbar()
	for t in cbar.ax.get_yticklabels():
		t.set_fontsize(20)
	pl.xticks(fontsize = 20)
	pl.yticks(fontsize = 20)
	pl.suptitle('FCM (BOLD-fMRI)', fontsize= 20)
	pl.xlabel('Nodes', fontsize = 20)
	pl.ylabel('Nodes', fontsize = 20)
	#image_name = simfile[0:-4] + '_CORR.eps'	
	#pl.savefig(simfile[0:-4]+'.eps', format="eps")
	#pl.show()
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


## loading correl. mtx. of simulated bold activities (calcBOLD output)
#name = 'A_aal_0_ADJ_thr_0.54_sigma=0.2_D=0.05_v=90.0_tmax=45000_corrcoeff.dat'
#name_2 = 'A_aal_0_ADJ_thr_0.54_sigma=0.2_D=0.05_v=70.0_tmax=45000_corrcoeff.dat'

# loading correl. mtx. of fhn time series (output of correlation_fhn.py)
#name = 'A_aal_0_ADJ_thr_0.54_sigma=0.2_D=0.05_v=90.0_tmax=45000_FHN_corr.dat'
#name_2 = 'A_aal_0_ADJ_thr_0.54_sigma=0.2_D=0.05_v=70.0_tmax=45000_FHN_corr.dat'
name = 'acp_w_0_ADJ_thr_0.26_sigma=0.5_D=0.05_v=70.0_tmax=45000_FHN_corr.dat'

thr_array = np.array([16, 22, 26, 32, 36, 42, 46, 52, 54, 56, 58, 60, 
					  62, 64, 66, 72, 76, 82 ])
                      
vel_array = np.array([30, 50, 70, 90, 110, 130, 150])

sig_array = np.array([0.1, 0.3, 0.5, 0.9])

R_thr =  {}

for THR in thr_array :
	R_temp = []
	
	for VEL in vel_array :
		local_path = '../data/jobs_corr/'
 		input_name = name[0:18] + str(THR) + name[20:40] + str(VEL) + name[42:]		
		
		try:
			mtx_simuli = load_matrix(local_path + input_name)
		except :
			R_vel      = np.nan
		else :
			R_vel      = pearson_coef(mtx_empiri, mtx_simuli)
			
		R_temp     = np.append(R_temp, R_vel)
	R_thr[THR] 	   = np.array(R_temp)
		
	#for SIG in sig_array :
		#local_path = '../data/jobs_corr/'
		#input_name = name[0:18] + str(THR) + name[20:27] + str(SIG) + name[30:]		
	
		#try:
			#mtx_simuli = load_matrix(local_path + input_name)
		#except :
			#R_sig      = np.nan
		#else :
			#R_sig      = pearson_coef(mtx_empiri, mtx_simuli)
	
		#R_temp     = np.append(R_temp, R_sig)
	#R_thr[THR] 	   = np.array(R_temp)
	
	
Ordered_R   = collections.OrderedDict(sorted(R_thr.items()))	
#print "Ordered dict"
#print Ordered_R

datam 		= np.array(Ordered_R.values())

#print "check its numpy array version"
#print datam

## PLOTTING BEGINS ! 
#fig , ax = pl.subplots()
#pl.imshow(np.transpose(datam), interpolation='nearest', cmap='jet', aspect='auto')
#cbar = pl.colorbar()

## PLOT PA OVER SIGMA
#a = thr_array
#b = sig_array
## title for fhn....
#pl.title('A_aal_0...' + ' , FHN , ' + '  v = 7 [m/s] '+' T = 450 [s]',
		 #fontsize=20)
## title for bold...
##pl.title('A_aal_0...' + ' , BOLD , ' + '  v = 7 [m/s]', fontsize=20)
#pl.ylabel('$\sigma$ ', fontsize=20)

## PLOT PA OVER VELOCITY
#a = thr_array
#b = vel_array
## title for fhn....
#pl.title('acp_w_0_...' + ' , FHN , ' + '$\sigma$ = 0.5 '+' T = 450 [s]',
		 #fontsize=20)
## title for bold...
##pl.title('A_aal_0...' + ' , BOLD , ' + '$\sigma$ = 0.2', fontsize=20)
#pl.ylabel('v [m/s]', fontsize=20)

#pl.setp(ax , xticks=np.arange(0,len(a),1), xticklabels = a)
#pl.setp(ax , yticks=np.arange(0,len(b),1), yticklabels = b)
#pl.xlabel('thr', fontsize = 20)
#for t in cbar.ax.get_yticklabels():
	#t.set_fontsize(15)
#pl.xticks(fontsize = 15)
#pl.yticks(fontsize = 15)
#pl.show()		

#------------------------------------
mtx_empiri			= 		load_matrix(input_empiri)		
#figure				=		plot_corr(mtx_empiri , input_empiri)
mtx_simuli			= 		load_matrix(input_simuli)
pl.figure(1)
HistA = corr_histo(mtx_empiri, input_empiri)
pl.figure(2)
HistB = corr_histo(mtx_simuli, input_simuli)
intersec_hists(HistA, HistB)

#chi2_distance(HistA, HistB, eps = 1e-10)
#figure_hist			=		plot_hist(mtx_simuli, input_simuli)
#R_pearson			= 	    pearson_coef(mtx_empiri , mtx_simuli)
#print "Pearson corr. coef. between empir.-simul. : " , R_pearson
#figure_name 		= 		input_simuli[0:-3] + str('eps')	
#print figure_name
#pl.title('A_aal, 0-V , FHN \nthr=0.53 , $\sigma$ = 0.02 , tmax=450[s] v = 7 [m/s]', fontsize=20)
pl.show()
