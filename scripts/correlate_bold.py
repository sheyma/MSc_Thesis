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
	print "nodes ",nx+1," and ",ny+1," best corr. : ", corr_matrix[nx,ny] 
	print "nodes ",mx+1," and ",my+1," worst corr.: ", corr_matrix[mx,ny]
	return nx, ny , mx, my

# plots the correlation matrix of SIMULATED signal
# input: any output of fhn_time_delays.py or output of calcBOLD.py 
# trick: remove 1's to 0's along the diagonals 
def plot_corr_diag(corr_matrix, matrix_name) :	
	N_col  = np.shape(corr_matrix)[1]
	extend = (0.5 , N_col+0.5 , N_col+0.5, 0.5 )	
	#for i in range(0,N_col):
		#for j in range(0,N_col):
			#if i==j :
				#corr_matrix[i,j] = 1
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
	pl.plot(time, v1, 'r',label=('node '+str(x+1)))
	pl.plot(time, v2, 'b',label=('node '+str(y+1)))
	
	pl.setp(pl.gca().get_xticklabels(), fontsize = 15)
	pl.setp(pl.gca().get_yticklabels(), fontsize = 15)
	lg = legend()
	pl.xlabel('t [min]', fontsize=25)
	pl.ylabel('BOLD change', fontsize=25)
	pl.title(('BOLD activity, corr. coeff. of nodes : ' 
			   +str("%.2f" % R_pearson)),fontsize=25)
	#pl.savefig(simfile[:-4]+"_timeseries.eps",format="eps")
	#pl.show()
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
corr_matrix		=		correl_matrix(data_matrix , input_name)
# real node index : add 1!
[i, j, k , l ]  = 	    node_index(corr_matrix)
#image			= 		plot_corr_diag(corr_matrix, input_name)

## BOLD activity of the nodes correlating the best
#pl.figure(2)
#plot_bold_signal(data_matrix, i+1,j+1)
## BOLD activity of the nodes correlating the worst
#pl.figure(3)
#plot_bold_signal(data_matrix, k+1,l+1)

rnd_node_1 = 7
rnd_node_2 = 24

# bold signal parameters, dtt [ms] is resolution of bold signal
params = {'dtt' : 0.001}
print params['dtt']

[yfft_1, freq_1] = bold_fft(data_matrix, rnd_node_1-1, params['dtt'])
[yfft_2, freq_2] = bold_fft(data_matrix, rnd_node_2-1, params['dtt'])
pl.figure(4);
pl.subplot(2,1,1)
plot_bold_signal(data_matrix, rnd_node_1-1, rnd_node_2-1)
pl.subplot(2,1,2)
pl.plot(freq_1, yfft_1, 'r',label=('node '+str(rnd_node_1)))
pl.plot(freq_2, yfft_2, 'b',label=('node '+str(rnd_node_2)))
pl.setp(pl.gca().get_xticklabels(), fontsize = 15)
pl.setp(pl.gca().get_yticklabels(), fontsize = 15)
lg = legend()
pl.title('Fourier Transformed Signal', fontsize=25)
pl.xlabel('frequency [Hz]' , fontsize = 25 )
pl.ylabel('bold signal (f)' , fontsize = 25 )
pl.show()
