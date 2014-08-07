#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# plotting correlation matrixes from various input

import subprocess as sp
import numpy as np
import matplotlib.pyplot as pl	
import sys 
import math

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
	#file_name       = str(matrix_name[:-4] + '_FHN_corrcoeff.dat') 	
	#np.savetxt(file_name, correl_matrix, '%.6f',delimiter='\t')
	return cr_matrix

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



data_matrix 		=		load_matrix(infile)
[u_matrix , T ]	    =		fhn_timeseries(data_matrix)
print u_matrix
corr_matrix			=		correl_matrix(u_matrix, input_name)
print corr_matrix
