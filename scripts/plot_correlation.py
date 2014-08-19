#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# plotting correlation matrixes from various input

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
	
# plots colorbar coded given correlation matrix
def plot_corr(corr_matrix, simfile ):	
	N_col  = np.shape(corr_matrix)[1]
	extend = (0.5 , N_col+0.5 , N_col+0.5, 0.5 )	
	pl.imshow(corr_matrix, interpolation='nearest', extent=extend)
	cbar = pl.colorbar()
	#cbar.set_ticks(np.arange(-0.8 , 1+0.2, 0.2))  #(np.arange(-0.8, 0.8+0.2, 0.2) )
	#cbar.set_ticklabels(np.arange(-0.8, 0.8+0.2, 0.2))
	for t in cbar.ax.get_yticklabels():
		t.set_fontsize(15)
	pl.xticks(fontsize = 20)
	pl.yticks(fontsize = 20)
	#pl.title('BOLD-fMRI', fontsize=20)
	#pl.xlabel('Nodes', fontsize = 20)
	#pl.ylabel('Nodes')
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
		for j in range(0, np.shape(matrix_A)[1]):
			
			# removing the diagonal elements in both matrices to get
			# more reasonable Pearson correlation coef. between them
			if i == j:
				
				tmp_a  = np.append(matrix_A[i, 0:j], matrix_A[i, j+1:])
				vec_a  = np.append(vec_a, tmp_a)
				
				tmp_b  = np.append(matrix_B[i, 0:j], matrix_B[i, j+1:])
				vec_b  = np.append(vec_b, tmp_b)	
				
	[R_pearson , p_value] = sistat.pearsonr(vec_a , vec_b)
	return R_pearson	
	

if __name__ == '__main__':
	usage = 'Usage: %s method correlation_matrix [threshold]' % sys.argv[0]
	try:
		input_empiri = sys.argv[1]
		#input_simuli  = sys.argv[2]
	except:
		print usage
		sys.exit(1)

# fMRI-BOLD input matrix load , this is a correlation matrix already
mtx_original		=		load_matrix(input_empiri)


## loading correl. mtx. of simulated bold activities (calcBOLD output)
#name = 'A_aal_0_ADJ_thr_0.54_sigma=0.2_D=0.05_v=90.0_tmax=45000_corrcoeff.dat'
#name_2 = 'A_aal_0_ADJ_thr_0.54_sigma=0.2_D=0.05_v=70.0_tmax=45000_corrcoeff.dat'


# loading correl. mtx. of fhn time series (output of correlation_fhn.py)
#name = 'A_aal_0_ADJ_thr_0.54_sigma=0.2_D=0.05_v=90.0_tmax=45000_FHN_corr.dat'
name_2 = 'A_aal_0_ADJ_thr_0.54_sigma=0.2_D=0.05_v=70.0_tmax=45000_FHN_corr.dat'


R_thr =  {}

for THR in np.array([54 , 56 , 58, 60 , 62, 63, 64, 65, 66]):
	R_temp = []

	#for VEL in (np.arange(150, 30-10, -10)):
		#input_name = name[0:18] + str(THR) + name[20:40] + str(VEL) + name[42:]		
		#mtx_simuli = load_matrix(input_name)
		#R_vel      = pearson_coef(mtx_original, mtx_simuli)
		#R_temp     = np.append(R_temp, R_vel)
	#R_thr[THR] 	   = np.array(R_temp)

	for SIG in np.arange(1.0, 0.1-0.1 , -0.1):
		input_name = name_2[0:18] + str(THR) + name_2[20:27] + str(SIG) + name_2[30:]		
		mtx_simuli = load_matrix(input_name)
		R_sig      = pearson_coef(mtx_original, mtx_simuli)
		R_temp     = np.append(R_temp, R_sig)
	R_thr[THR] 	   = np.array(R_temp)
	
	
Ordered_R   = collections.OrderedDict(sorted(R_thr.items()))	
#print "Ordered dict"
#print Ordered_R

datam 		= np.array(Ordered_R.values())

#print "check its numpy array version"
#print datam

# PLOTTING BEGINS ! 
fig , ax = pl.subplots()
pl.imshow(np.transpose(datam),  interpolation='nearest') 
cbar = pl.colorbar()

# PLOT PA OVER SIGMA
a = np.array([54 , 56 , 58, 60 , 62, 63, 64, 65, 66])
b = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2 , 0.1])
pl.title('A_aal_0...' + ' , FHN , ' + '  v = 7 [m/s]', fontsize=20)
pl.ylabel('$\sigma$ ', fontsize=20)

# PLOT PA OVER VELOCITY
#a = np.array([54 , 56 , 58, 60 , 62, 63, 64, 65, 66])
#b = np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3])
#pl.title('A_aal_0...' + ' , FHN , ' + '$\sigma$ = 0.2', fontsize=20)
#pl.ylabel('v [m/s]', fontsize=20)

pl.setp(ax , xticks=np.arange(0,len(a),1), xticklabels = a)
pl.setp(ax , yticks=np.arange(0,len(b),1), yticklabels = b)
pl.xlabel('thr', fontsize = 20)
for t in cbar.ax.get_yticklabels():
	t.set_fontsize(15)
pl.xticks(fontsize = 15)
pl.yticks(fontsize = 15)
pl.show()		



#------------------------------------
#mtx_empiri			= 		load_matrix(input_empiri)		
#figure				=		plot_corr(mtx_empiri , input_empiri)
#mtx_random			= 		load_matrix(input_simuli)
#mtx_random			= 		load_matrix(name_2)
#R_pearson			= 	    pearson_coef(mtx_random , mtx_empiri)
#figure				=		plot_corr(mtx_random , name_2)
#pl.title('A_aal, 0 , calcBOLD, thr=0.63 , $\sigma$ = 0.3 , v = 7 [m/s]', fontsize=20)
#pl.show()

#R					=       pearson_coef(mtx_original , mtx_random)
#print "pearson correlation coefficient is : " , R 
