#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# plotting correlation matrixes from various input

import networkx as nx
import numpy as np
from math import factorial 
import matplotlib.pyplot as pl	
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
	
def pearson_coef(matrix_A , matrix_B):
	vector_a = []
	vector_b = []
	for i in range(0, np.shape(matrix_A)[0]):
		vector_a = np.append(vector_a , matrix_A[i,:])
		vector_b = np.append(vector_b , matrix_B[i,:])
	[R_pearson , p_value] = sistat.pearsonr(vector_a , vector_b)
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

# loading correl. mtx. of simulated bold activities (calcBOLD output)
name = 'A_aal_0_ADJ_thr_0.54_sigma=0.2_D=0.05_v=90.0_tmax=45000_corrcoeff.dat'

R_thr =  {}

for THR in np.array([54 , 56 , 58, 60 , 62, 63, 64, 65, 66]):
	R_temp = []
	
	for VEL in (np.arange(30,150+10,10)):
		input_name = name[0:18] + str(THR) + name[20:40] + str(VEL) + name[42:]		
		mtx_simuli = load_matrix(input_name)
		R_vel      = pearson_coef(mtx_original, mtx_simuli)
		R_temp     = np.append(R_temp, R_vel)
	R_thr[THR] 	   = np.array(R_temp)

	
Ordered_R   = collections.OrderedDict(sorted(R_thr.items()))	
print "Ordered dict"
print Ordered_R

datam 		= np.array(Ordered_R.values())
#np.savetxt('delete.dat', datam, '%.10f', delimiter='\t')


print "check its numpy array version"
print datam

extend = [54 , 66, 3, 15]
fig  = pl.imshow(np.transpose(datam),  interpolation='nearest', extent=extend)
cbar = pl.colorbar()
pl.title('A_aal_0...' + '    $\sigma$ = 0.2', fontsize=20)
pl.xlabel('thr', fontsize = 20)
pl.ylabel('v [m/s]', fontsize=20)
a = fig.axes.get_xticklabels()
a = [54 , 56 , 58, 60 , 62, 63, 64, 65, 66]
fig.axes.set_xticklabels(a)
print "AAAAAAAAAAA" , a
for t in cbar.ax.get_yticklabels():
	t.set_fontsize(15)
pl.xticks(fontsize = 20)
pl.yticks(fontsize = 20)
pl.show()		


		
#mtx_random			= 		load_matrix(input_simuli)
#figure				=		plot_corr(mtx_random , input_random)
#pl.show()
#R					=       pearson_coef(mtx_original , mtx_random)
#print "pearson correlation coefficient is : " , R 
