#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import subprocess as sp
import numpy as np
import sys 
import math

""" 
	input  : job output from "calcBOLD.py" , m rows, n columns
	
	intermediate process : loading input, getting correlation 
	coefficients' matrix, exporting it 
	
	output : correlation matrix of the input, n rows, n rows
"""

# load input data as numpy matrix
def load_matrix(file):
	print "reading data ..."
	A  = np.loadtxt(file, unpack=False)
	print "shape of input matrix : " , np.shape(A)
	return A

# correlation coefficients among the columns of a given matrix
def correl_matrix(matrix , matrix_name):
	# numpy array must be transposed to get the right corrcoef
	tr_matrix = np.transpose(matrix)
	cr_matrix = np.corrcoef(tr_matrix)
	file_name = str(matrix_name[:-4] + '_corr.dat')
	np.savetxt(file_name, cr_matrix, '%.6f',delimiter='\t')
	return cr_matrix

# user defined input name
if __name__ == '__main__':
	try:
		input_name = sys.argv[1]
	except:
		sys.exit(1)
		
data_matrix		=		load_matrix(input_name)		
corr_matrix		=		correl_matrix(data_matrix , input_name)
