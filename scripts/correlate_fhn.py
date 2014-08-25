#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import subprocess as sp
import numpy as np
import sys 
import math
import scipy.stats as sistat

""" 
	input  : job output from "fhn_time_delays.py" , m rows, n columns
	
	intermediate process : loading input, which has ".xz" or ".dat" 
	extentions, deleting the first column of input, which is the time 
	column, getting correlation coefficients 
	
	output : correlation matrix of the input, n-1 rows, n-1 rows
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
	# index of maximum value in matrix
	[nx , ny] = np.unravel_index(matrix.argmax() , matrix.shape)
	print nx, ny , matrix[nx,ny]
	# index of maximum value in matrix
	[mx , my] = np.unravel_index(matrix.argmin() , matrix.shape)
	print mx, my , matrix[mx,my]
	return nx, ny , mx, my

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
	# in non-xz case we just use the file name instead of a file object, numpy's
	# loadtxt() can deal with this
	infile = input_name



data_matrix 		    =	load_matrix(infile)
#[u_matrix , T ]	    =	fhn_timeseries(data_matrix)
#corr_matrix			=	correl_matrix(u_matrix, input_name)
#[i, j, k , l ]			=   node_index(corr_matrix)

[i, j, k , l ]		    = 	node_index(data_matrix)
[R_pearson_A , p_value] =   sistat.pearsonr(data_matrix[:,i] , data_matrix[:,j])
[R_pearson_B , p_value] = sistat.pearsonr(data_matrix[:,k] , data_matrix[:,l])

print "nodes i ", i , " and j ",j, " well correlated"
print "nodes k ", k,  " and l ",l, " worse correlated"
print "R_pearson i-j : " , R_pearson_A 
print "R_pearson k-l : " , R_pearson_B 
