#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import sb_utils as sb
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

# correlation coefficients among the columns of a given matrix
def correl_matrix(matrix , out_basename):
	print "obtaining correlation coefficients among BOLD time series..."
	# numpy array must be transposed to get the right corrcoef
	tr_matrix = np.transpose(matrix)
	cr_matrix = np.corrcoef(tr_matrix)
	file_name = str(out_basename + '_corr.dat')
	#np.savetxt(file_name, cr_matrix, '%.6f',delimiter='\t')
	return cr_matrix

# find the max and min values and their index in the correlation matrix
def node_index(matrix):
	# ignore diagonal elements by assigning it to zero
	for i in range(0,np.shape(matrix)[0]):
		for j in range(0,np.shape(matrix)[1]):
			if matrix[i,j] >= 0.1 and matrix[i,j] < 0.18 :
				mx = i
				my = j

			if i == j :
				matrix[i,j] = 0
	
	print "some good correlation :", matrix.max()
	print "some bad correlation :", matrix[mx, my]

	# index of maximum value in matrix
	[nx , ny] = np.unravel_index(matrix.argmax() , matrix.shape)
		
	# assign diagonal elements back to one 
	for i in range(0,np.shape(matrix)[0]):
		for j in range(0,np.shape(matrix)[1]):
			if i == j :
				matrix[i,j] = 1

	# nodes start from 1, not from 0, therefore add 1 to the index
	print "nodes ",nx+1," and ",ny+1," good correlated  : ", matrix[nx,ny] 
	print "nodes ",mx+1," and ",my+1," bad correlated : ", matrix[mx,my]
	
	return nx, ny , mx, my

# plots the correlation matrix of SIMULATED signal
# input: any output of fhn_time_delays.py or output of calcBOLD.py 
# trick: remove 1's to 0's along the diagonals 
def plot_corr_diag(corr_matrix, out_basename):
	
	N_col  = np.shape(corr_matrix)[1]
	extend = (0.5 , N_col+0.5 , N_col+0.5, 0.5 )
	fig , ax = pl.subplots(figsize=(15, 12))
	ax.tick_params('both', length=15, width=8, which='major')
	pl.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.12)	
	
	for i in range(0,N_col):
		for j in range(0, N_col):
			if i==j:
				corr_matrix[i,j] = 1
			
	#pl.imshow(corr_matrix, interpolation='nearest', extent=extend)
	pl.imshow(corr_matrix, interpolation='nearest', vmin=-0.45, vmax=0.45, extent=extend)
	
	## vmin & vmax for EMPIRICAL data corr matrix :
	#pl.imshow(corr_matrix, interpolation='nearest', vmin=0.0, vmax=1.0, extent=extend)
	
	cbar = pl.colorbar()
	for t in cbar.ax.get_yticklabels():
		t.set_fontsize(50)
	pl.xticks(fontsize = 50)
	pl.yticks(fontsize = 50)
	pl.xlabel('Nodes', fontsize = 50)
	pl.ylabel('Nodes', fontsize = 50)	
	return
	
def plot_bold_signal(timeseries, x, y):
	# plots timeseries of two given nodes in a specific time interval
	
	v1  = timeseries[:, x]
	v2  = timeseries[:, y]
	
	T   = len(v1)	
	time = np.linspace(0, T-1, T) / float(60000)
	
	[R_pearson , p_value] = sistat.pearsonr(v1 , v2)
	
	## if the given signal downsampled :
	#time_bds = np.arange(0,  530,  float(530)/len(v1) )/float(60)
	#pl.plot(time_bds, v1, 'r',label=('node '+str(x)))
	#pl.plot(time_bds, v2, 'b',label=('node '+str(y)))
	
	# if no downsampling :

	fig , ax = pl.subplots(figsize=(25, 5.5))
	pl.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.20)

	pl.plot(time, v1, 'r')
	pl.plot(time, v2, 'b')
	pl.setp(pl.gca().get_xticklabels(), fontsize = 30)
	pl.setp(pl.gca().get_yticklabels(), fontsize = 30)
	
	ax.set_xlim(0, time.max()+0.05)
	ax.set_ylim(-0.32, 0.32)
	
	pl.legend(prop={'size':35})

	pl.xlabel('t [min]', fontsize=30)
	pl.ylabel('BOLD % change' ,fontsize=40)
	
	return	

def bold_fft(matrix, x, dtt) :
	
	# Sampling frequency (Hz)
	f_s = 1/float(dtt)
	# array of the signal given by the x'th column
	Y     = matrix[:,x]
	t_Y   = np.arange(0, len(Y), 1)
	m     = len(Y);
	m_pow = int(pow(2, math.ceil(math.log(m)/math.log(2))))
	# fast fourier transform of the x'th column
	Y_fft = np.fft.fft(Y , m_pow) /float(m)
	Y_fft = 2*abs(Y_fft[0:m_pow /2 +1])  
	# frequency domain [Hz]
	freq  = float(f_s)/2 * np.linspace(0,1, m_pow/2 + 1);
	
	return Y_fft, freq


# user defined input name
if __name__ == '__main__':
	try:
		input_name = sys.argv[1]
	except:
		sys.exit(1)

data_matrix = sb.load_matrix(input_name)
out_basename = sb.get_dat_basename(input_name)

corr_matrix = correl_matrix(data_matrix , out_basename)

## if data is already a correlation matrix :
#corr_matrix = data_matrix
image = plot_corr_diag(corr_matrix, out_basename)
# real node index : add 1!
[i, j, k , l ]  = 	    node_index(corr_matrix)
# BOLD activity of the nodes correlating the best
plot_bold_signal(data_matrix, i,j)
# BOLD activity of the nodes correlating the worst
plot_bold_signal(data_matrix, k,l)
pl.show()

#rnd_node_1 = 7
#rnd_node_2 = 24

## bold signal parameters, dtt [ms] is resolution of bold signal
#params = {'dtt' : 0.001}
#print params['dtt']

#[yfft_1, freq_1] = bold_fft(data_matrix, rnd_node_1-1, params['dtt'])
#[yfft_2, freq_2] = bold_fft(data_matrix, rnd_node_2-1, params['dtt'])
#pl.figure(4);
#pl.subplot(2,1,1)
#plot_bold_signal(data_matrix, rnd_node_1-1, rnd_node_2-1)
#pl.subplot(2,1,2)
#pl.plot(freq_1, yfft_1, 'r',label=('node '+str(rnd_node_1)))
#pl.plot(freq_2, yfft_2, 'b',label=('node '+str(rnd_node_2)))
#pl.setp(pl.gca().get_xticklabels(), fontsize = 15)
#pl.setp(pl.gca().get_yticklabels(), fontsize = 15)
#lg = legend()
#pl.title('Fourier Transformed Signal', fontsize=25)
#pl.xlabel('frequency [Hz]' , fontsize = 25 )
#pl.ylabel('bold signal (f)' , fontsize = 25 )
#pl.show()
