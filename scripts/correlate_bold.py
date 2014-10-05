#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import subprocess as sp
import numpy as np
import sys 
import math
import matplotlib.pyplot as pl
import scipy.stats as sistat
from pylab import *

""" 
	input  : job output from "calcBOLD.py" , m rows, n columns
	
	intermediate process : loading input, getting correlation 
	coefficients' matrix, exporting it 
	
	output : correlation matrix of the input, n rows, n columns
"""

#class Params(object):
	#__slots__ = [ 'dt' ]

#params.dt = 0.001

# load input data as numpy matrix
def load_matrix(file):
	print "reading data ..."
	A  = np.loadtxt(file, unpack=False)
	print "shape of input matrix : " , np.shape(A)
	return A

# correlation coefficients among the columns of a given matrix
def correl_matrix(matrix , matrix_name):
	print "obtaining correlation coefficients among BOLD time series..."
	# numpy array must be transposed to get the right corrcoef
	tr_matrix = np.transpose(matrix)
	cr_matrix = np.corrcoef(tr_matrix)
	file_name = str(matrix_name[:-4] + '_corr.dat')
	np.savetxt(file_name, cr_matrix, '%.6f',delimiter='\t')
	return cr_matrix

# find the max and min values and their index in the correlation matrix
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
	
	# assign diagonal elements back to zero 
	for i in range(0,np.shape(matrix)[0]):
		for j in range(0,np.shape(matrix)[1]):
			if i == j :
				matrix[i,j] = 1
	#nodes start from 1, not from 0 on figure !
	print "nodes ",nx+1," and ",ny+1," best corr. : ", corr_matrix[i,j] 
	print "nodes ",mx+1," and ",my+1," worst corr.: ", corr_matrix[k,l]
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
				corr_matrix[i,j] = 1
	cmap   = pl.cm.jet
	pl.imshow(corr_matrix, interpolation='nearest', extent=extend, vmin=-0.5, vmax=0.5, cmap='jet', aspect='auto')
	cbar = pl.colorbar()
	for t in cbar.ax.get_yticklabels():
		t.set_fontsize(15)
	pl.xticks(fontsize = 20)
	pl.yticks(fontsize = 20)
	#pl.suptitle("BOLD-signal correlation matrix", fontsize=20)
	#pl.title('Method : 0 , ' + '$r$ = ' +'0.65'  +
				#r'  $  \sigma$ = '+'0.025'+ ' $   D$ = '+ 
				#'0.05' + '  $v$ = '+'7 [m/s]',	
				#fontsize=14, fontweight='bold')
	pl.xlabel('Nodes', fontsize = 20)
	pl.ylabel('Nodes', fontsize = 20)
	if matrix_name.endswith(".xz"):
		image_name       = str(matrix_name[:-7] + '_CORR.eps') 	
	else :
		image_name       = str(matrix_name[:-4] + '_CORR.eps')
	#pl.savefig(image_name, format="eps")
	#pl.show()
	return
	
def plot_bold_signal(timeseries, x, y):
	# plots timeseries of two given nodes in a specific time interval
	
	v1  = timeseries[:, x-1]
	v2  = timeseries[:, y-1]
	
	T   = len(v1)	
	time = np.linspace(0, T-1, T) / float(60000)
	
	[R_pearson , p_value] = sistat.pearsonr(v1 , v2)
	
	pl.plot(time, v1, 'r',label=('node '+str(x)))
	pl.plot(time, v2, 'b',label=('node '+str(y)))
	lg = legend()
	pl.xlabel('t [min]')
	pl.ylabel('$BOLD-signal_i(t)$')
	pl.title(('simulated BOLD activity, corr. coeff. of nodes : ' 
				+ str("%.2f" % R_pearson)), fontweight='bold')
	#pl.savefig(simfile[:-4]+"_timeseries.eps",format="eps")
	#pl.show()
	return	

# user defined input name
if __name__ == '__main__':
	try:
		input_name = sys.argv[1]
	except:
		sys.exit(1)
		
data_matrix		=		load_matrix(input_name)		
corr_matrix		=		correl_matrix(data_matrix , input_name)
# real node index : add 1!
[i, j, k , l ]  = 	    node_index(corr_matrix)
image			= 		plot_corr_diag(corr_matrix, input_name)

# BOLD activity of the nodes correlating the best
pl.figure(2)
plot_bold_signal(data_matrix, i+1,j+1)
# BOLD activity of the nodes correlating the worst
pl.figure(3)
plot_bold_signal(data_matrix, k+1,l+1)
pl.show()
