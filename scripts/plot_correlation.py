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
	cbar.set_ticks(np.arange(-0.8 , 1+0.2, 0.2))  #(np.arange(-0.8, 0.8+0.2, 0.2) )
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
	
if __name__ == '__main__':
	usage = 'Usage: %s method correlation_matrix [threshold]' % sys.argv[0]
	try:
		input_original = sys.argv[1]
		input_random   = sys.argv[2]
	except:
		print usage
		sys.exit(1)


def pearson_coef(matrix_A , matrix_B):
	vector_a = []
	vector_b = []
	for i in range(0, np.shape(matrix_A)[0]):
		vector_a = np.append(vector_a , matrix_A[i,:])
		vector_b = np.append(vector_b , matrix_B[i,:])
	[R_pearson , p_value] = sistat.pearsonr(vector_a , vector_b)
	return R_pearson	
		
		
mtx_original		=		load_matrix(input_original)
mtx_random			= 		load_matrix(input_random)
figure				=		plot_corr(mtx_random , input_random)
#pl.show()
R					=       pearson_coef(mtx_original , mtx_random)
print "pearson correlation coefficient is : " , R 
