#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# plotting correlation matrixes from various input

import subprocess as sp
import numpy as np
import matplotlib.pyplot as pl	
import sys 

# check the loaded matrix if it is symmetric
def load_matrix(file):
	A  = np.loadtxt(file, unpack=True)
	return A

def correl_matrix(matrix , matrix_name):
	# correlation coefficient among the columns of bold_input calculated
	# numpy array must be transposed to get the right corrcoef
	
	transpose_input = np.transpose(matrix)
	correl_matrix   = np.corrcoef(transpose_input)
	
	#file_name       = str(matrix_name[:-4] + '_FHN_corrcoeff.dat') 	
	#np.savetxt(file_name, correl_matrix, '%.6f',delimiter='\t')
	return correl_matrix

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
corr_matrix			=		correl_matrix(data_matrix, input_name)
print corr_matrix
