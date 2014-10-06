#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import subprocess as sp
import numpy as np
import sys 
import math
import scipy.stats as sistat
import matplotlib.pyplot as pl
from matplotlib import colors
from pylab import *


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
	print "T = " , T , "[seconds]", "dt = " , dt/100 ,"[seconds]"
	# extract u-columns
	u_indices  = np.arange(1, simfile.shape[1] ,1)
	u_series   = simfile[:, u_indices]
	return u_series , T, dt, tvec

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

# finding the index of max and min values in a given correlation matrix
def node_index(matrix):
	# ignore diagonal elements by assigning it to 0
	for i in range(0,np.shape(matrix)[0]):
		for j in range(0,np.shape(matrix)[1]):
			if i==j:
				matrix[i,j] = 0
	print "max. corr. coef. in the correlation matrix:", matrix.max()
	print "min. corr. coef. in the correlation matrix:", matrix.min()

	# index of maximum value in matrix
	[nx , ny] = np.unravel_index(matrix.argmax() , matrix.shape)
	# index of maximum value in matrix
	[mx , my] = np.unravel_index(matrix.argmin() , matrix.shape)
	
	# nodes start from 1, not from 0, therefore add 1 to the index
	print "nodes ",nx+1," and ",ny+1," best correlated  : ", matrix[nx,ny] 
	print "nodes ",mx+1," and ",my+1," worst correlated : ", matrix[mx,my]

	# assign diagonal elements back to 1 
	for i in range(0,np.shape(matrix)[0]):
		for j in range(0,np.shape(matrix)[1]):
			if i == j :
				matrix[i,j] = 1.0
	
	return nx, ny , mx, my

# plots the correlation matrix of SIMULATED signal
# input: any output of fhn_time_delays.py or output of calcBOLD.py 
def plot_corr_diag(corr_matrix, matrix_name) :	
	N_col  = np.shape(corr_matrix)[1]
	extend = (0.5 , N_col+0.5 , N_col+0.5, 0.5 )
		
	cmap   = pl.cm.jet
	pl.imshow(corr_matrix, interpolation='nearest', extent=extend, vmin=-0.5, vmax=0.5, cmap='jet', aspect='auto')
	cbar = pl.colorbar(cmap=cmap, norm=norm)
	for t in cbar.ax.get_yticklabels():
		t.set_fontsize(15)
	pl.xticks(fontsize = 20)
	pl.yticks(fontsize = 20)
	pl.suptitle("FHN correlation matrix", fontsize=20)
	#pl.title('Method : 0 , ' + '$r$ = ' +'0.64'  +
				#r'  $  \sigma$ = '+'0.025'+ ' $   D$ = '+ 
				#'0.05' + '  $v$ = '+'7 [m/s]',	
				#fontsize=14, fontweight='bold')
	pl.xlabel('Nodes', fontsize = 20)
	pl.ylabel('Nodes', fontsize = 20)
	if matrix_name.endswith(".xz"):
		image_name       = str(matrix_name[:-7] + '_FHN_CORR.eps') 	
	else :
		image_name       = str(matrix_name[:-4] + '_FHN_CORR.eps')
	#pl.savefig('FHN_corr_r_0_64_si_0_030.eps', format="eps")
	#pl.show()
	return

# plots timeseries of two given nodes in a specific time interval
def plot_timeseries(t_start , t_final, dt, timeseries, tvec, x, y):
	
	# corresponding index of t_start and t_final in tvec
	i_s =  (t_start /dt)
	i_f =  (t_final /dt)
	
	# extracting the timeseries of the given nodes as separate vectors
	v1   = timeseries[:, x-1]
	v2   = timeseries[:, y-1]
	
	# Pearson correlation value between two timeseries
	[R_pearson , p_value] = sistat.pearsonr(v1 , v2)
	
	# plot the timeseries of two nodes in specific interval
	# tvec multiplied by 0.01 to make dimensiion equal to [ms]
	pl.plot(0.01*tvec[i_s:i_f], v1[i_s : i_f],'r.-',label=('node '+str(x)))
	pl.plot(0.01*tvec[i_s:i_f], v2[i_s : i_f],'b.-',label=('node '+str(y)))
	lg = legend()
	pl.xlabel('t [s]')
	pl.ylabel('$u_i(t)$')
	pl.title(('FHN - timeseries, corr. coeff. of nodes : ' 
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

# handle xz files transparently

if input_name.endswith(".xz"):
	# non-portable but we don't want to depend on pyliblzma module
	xzpipe = sp.Popen(["xzcat", input_name], stdout=sp.PIPE)
	infile = xzpipe.stdout
else:
	# in non-xz case we just use the file name instead of a file object, 
	# numpy's loadtxt() can deal with this
	infile = input_name

data_matrix 	   			 =	load_matrix(infile)
[u_matrix , T, dt, tvec] 	 =	fhn_timeseries(data_matrix)
corr_matrix		   			 =	correl_matrix(u_matrix, input_name)

## if correlation matrix is given directly :
#corr_matrix			=   load_matrix(infile)

pl.figure(1)
plot_corr_diag(corr_matrix, input_name )
# node indexes, not forget to subtract 1 
[i, j, k , l ]		=   node_index(corr_matrix)

# user defined time range for timeseries plots
t_start = 1650
t_final = 1750
# plot the timeseries of best correlated nodes
pl.figure(2)
plot_timeseries(t_start, t_final, dt, u_matrix, tvec, i+1, j+1)
#plot the timeseries of worst correlated nodes
pl.figure(3)
plot_timeseries(t_start, t_final, dt, u_matrix, tvec, k+1, l+1)
pl.show()
