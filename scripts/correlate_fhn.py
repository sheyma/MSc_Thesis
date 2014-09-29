#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import subprocess as sp
import numpy as np
import sys 
import math
import scipy.stats as sistat
import matplotlib.pyplot as pl

""" 
	input  : job output from "fhn_time_delays.py" , m rows, n columns
	
	intermediate process : loading input, which has ".xz" or ".dat" 
	extentions, deleting the first column of input, which is the time 
	column, getting correlation coefficients between columns of input
	matrix 
	
	output : correlation matrix of the input [n-1 rows, n-1 columns],
	maximum and minumum correlation coefficients in this matrix and the 
	index of those values, plot correlation matrix colorcoded
"""

params = { # Fitzhugh-Nagumo simulation parameters...
        'dt': 0.001, 
			}

# load input data as numpy matrix
def load_matrix(file):
	print "reading data ..."
	A  = np.loadtxt(file, unpack=False)
	print "shape of input matrix : " , np.shape(A)
	return A

# obtain u_i time series from loaded matrix
def fhn_timeseries(simfile):
	print "subtracting u-time series as numpy matrix..."
	# extract first column of simout as time vector
	tvec = simfile[:,0]
	dt   = tvec[1] - tvec[0]
	# calculate total time of simulation 
	T    = int(math.ceil( (tvec[-1])  / dt * params['dt'] ))
	print "T = " , T , "[seconds]"
	# extract u-columns
	u_indices  = np.arange(1, simfile.shape[1] ,1)
	u_series   = simfile[:, u_indices]
	return u_series , T

# correlation coefficients among the columns of a given matrix
def correl_matrix(matrix , matrix_name):
	print "obtaining correlation coefficients among time series..."
	# numpy array must be transposed to get the right corrcoef
	tr_matrix = np.transpose(matrix)
	cr_matrix = np.corrcoef(tr_matrix)
	if matrix_name.endswith(".xz"):
		file_name       = str(matrix_name[:-7] + '_FHN_corr.dat') 	
	else :
		file_name       = str(matrix_name[:-4] + '_FHN_corr.dat')
	np.savetxt(file_name, cr_matrix, '%.6f',delimiter='\t')
	return cr_matrix

def node_index(matrix):
	# ignore diagonal elements by assigning it to zero
	for i in range(0,np.shape(matrix)[0]):
		for j in range(0,np.shape(matrix)[1]):
			if i == j :
				matrix[i,j] = 0
	print "max. corr. coef. in the correlation matrix:", matrix.max()
	print "min. corr. coef. in the correlation matrix:", matrix.min()

	# index of maximum value in matrix
	[nx , ny] = np.unravel_index(matrix.argmax() , matrix.shape)
	# index of maximum value in matrix
	[mx , my] = np.unravel_index(matrix.argmin() , matrix.shape)
	return nx, ny , mx, my

# plots the correlation matrix of SIMULATED signal
# input: any output of fhn_time_delays.py or output of calcBOLD.py 
# trick: remove 1's to 0's along the diagonals 
def plot_corr_diag(corr_matrix, matrix_name) :	
	N_col  = np.shape(corr_matrix)[1]
	extend = (0.5 , N_col+0.5 , N_col+0.5, 0.5 )	
	for i in range(0,N_col):
		for j in range(0,N_col):
			if i==j :
				corr_matrix[i,j] = 0
	pl.imshow(corr_matrix, interpolation='nearest', extent=extend)
	cbar = pl.colorbar()
	for t in cbar.ax.get_yticklabels():
		t.set_fontsize(15)
	pl.xticks(fontsize = 20)
	pl.yticks(fontsize = 20)
	#pl.title("FHN correlation matrix\t input_name", fontsize=20)
	pl.xlabel('Nodes', fontsize = 20)
	pl.ylabel('Nodes', fontsize = 20)
	if matrix_name.endswith(".xz"):
		image_name       = str(matrix_name[:-7] + '_FHN_CORR.eps') 	
	else :
		image_name       = str(matrix_name[:-4] + '_FHN_CORR.eps')
	pl.savefig(image_name, format="eps")
	#pl.show()
	return

# user defined input name
if __name__ == '__main__':
	try:
		input_name = sys.argv[1]
	except:
		sys.exit(1)

# handle xz files transparently

if input_name.endswith(".xz"):
	# non-portable but we don't want to depend on pyliblzma module
	xzpipe = sp.Popen(["xzcat", input_name], stdout=sp.PIPE)
	infile = xzpipe.stdout
else:
	# in non-xz case we just use the file name instead of a file object, 
	# numpy's loadtxt() can deal with this
	infile = input_name

data_matrix 	    =	load_matrix(infile)
[u_matrix , T]	    =	fhn_timeseries(data_matrix)
corr_matrix		    =	correl_matrix(u_matrix, input_name)
plot_corr_diag(corr_matrix, input_name )
[i, j, k , l ]		=   node_index(corr_matrix)

# nodes start from 1, not from 0, therefore add 1 to the index
print "nodes ", i+1," and ",j+1," best correlated  : ", corr_matrix[i,j] 
print "nodes ", k+1," and ",l+1," worst correlated : ", corr_matrix[k,l]

pl.show()
