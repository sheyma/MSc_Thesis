#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import sb_utils as sb
import numpy as np
import sys 
import math
import scipy.stats as sistat
import matplotlib.pyplot as pl
from matplotlib import colors
from pylab import *
from scipy.signal import  butter , filtfilt , correlate2d
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata	
from matplotlib import cm

""" 
	input  : job output from "fhn_time_delays.py" , m rows, n columns
	
	intermediate process : loading input, which has ".xz" or ".dat" 
	extentions, deleting the first (time) column of input, 
	calculationg Pearson's correlat. coeff.s between columns of input
	matrix, finding the best and worst correlated node pairs and 
	plotting their time-series, fast-fourier-transform analysis of 
	time series 
	
	output : correlation matrix of the input [n-1 rows, n-1 columns],
	maximum and minumum correlation coefficients in this matrix and the 
	index of those values, plot correlation matrix colorcoded
"""

params = { # Fitzhugh-Nagumo simulation parameters...
        'dt': 0.001, 
			}

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
def correl_matrix(matrix , out_basename):
	print "obtaining correlation coefficients among time series..."
	# numpy array must be transposed to get the right corrcoef
	tr_matrix = np.transpose(matrix)
	cr_matrix = np.corrcoef(tr_matrix)
	file_name = str(out_basename + '_FHN_corr.dat')
	np.savetxt(file_name, cr_matrix, '%.6f',delimiter='\t')
	return cr_matrix

# finding the index of max and min values in a given correlation matrix
def node_index(matrix):
	# ignore diagonal elements by assigning it to 0
	
	for i in range(0,np.shape(matrix)[0]):
		for j in range(0,np.shape(matrix)[1]):
			if matrix[i,j] >= 0.1 and matrix[i,j] < 0.18 :
				mx = i
				my = j
			if matrix[i,j] >= 0.85 and matrix[i,j] < 0.90 and i<45 and j<45:
				nx = i
				ny = j
			if i == j:
				matrix[i,j] = 0
			
		
	print "some good correlation :", matrix[nx,ny]
	print "some bad correlation :", matrix[mx, my]

	# nodes start from 1, not from 0, therefore add 1 to the index
	print "nodes ",nx+1," and ",ny," good correlated  : ", matrix[nx,ny] 
	print "nodes ",mx," and ",my," bad correlated : ", matrix[mx,my]
	
	# assign diagonal elements back to 1 
	for i in range(0,np.shape(matrix)[0]):
		for j in range(0,np.shape(matrix)[1]):
			if i == j :
				matrix[i,j] = 1.0
	
	return nx, ny , mx, my

# plots the correlation matrix of SIMULATED signal
# input: any output of fhn_time_delays.py or output of calcBOLD.py 
def plot_corr_diag(corr_matrix, out_basename):
	N_col  = np.shape(corr_matrix)[1]
	extend = (0.5 , N_col+0.5 , N_col+0.5, 0.5 )
		
	cmap   = pl.cm.jet
	#pl.imshow(corr_matrix, interpolation='nearest', extent=extend, cmap='jet', aspect='auto')
	pl.imshow(corr_matrix, interpolation='nearest', extent=extend, vmin=-0.5, vmax=0.5, cmap='jet', aspect='auto')
	cbar = pl.colorbar(cmap=cmap, norm=norm)
	for t in cbar.ax.get_yticklabels():
		t.set_fontsize(15)
	pl.xticks(fontsize = 25)
	pl.yticks(fontsize = 25)
	
	pl.suptitle("FHN correlation matrix, ACM", fontsize=20)
	pl.title('r=0.60, ' + ' c = 0.3, ' + ' v = 6 [m/s]')
	pl.xlabel('Nodes', fontsize = 25)
	pl.ylabel('Nodes', fontsize = 25)
	image_name = str(out_basename + '_FHN_CORR.eps')
	#pl.savefig('FHN_corr_r_0_64_si_0_030.eps', format="eps")
	#pl.show()
	return

# plots timeseries of two given nodes in a specific time interval
def plot_timeseries(t_start , t_final, dt, timeseries, tvec, x, y):
	
	# corresponding index of t_start and t_final in tvec
	i_s =  (t_start /dt)
	i_f =  (t_final /dt)
	
	# extracting the timeseries of the given nodes as separate vectors
	v1   = timeseries[:, x]
	v2   = timeseries[:, y]
	
	# Pearson correlation value between two timeseries
	[R_pearson , p_value] = sistat.pearsonr(v1 , v2)
	
	# plot the timeseries of two nodes in specific interval
	# tvec multiplied by 0.01 to make dimensiion equal to [ms]
	fig , ax = pl.subplots(figsize=(25, 5))
	pl.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.20)

	pl.plot(0.01*tvec[i_s:i_f], v1[i_s : i_f], linestyle='-',color='r',label=('$u_{' + str(x+1) + '}(t)$'))
	pl.plot(0.01*tvec[i_s:i_f], v2[i_s : i_f], linestyle='-',color='b',label=('$u_{' + str(y+1) + '}(t)$'))
	pl.setp(pl.gca().get_xticklabels(), fontsize = 30)
	pl.setp(pl.gca().get_yticklabels(), fontsize = 30)
	
	pl.legend(prop={'size':35})

	pl.xlabel('t [s]', fontsize=30)
	pl.ylabel('$u_{' + str(x+1) + '}(t)$ , ' + ' $u_{' + str(y+1) + '}(t)$' ,fontsize=40)
	return	

def fhn_fft(matrix, x, dtt) :
	
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

## if input is already a correlation matrix ::
#corr_matrix = sb.load_matrix(input_name)
#out_basename  = sb.get_dat_basename(input_name)
#pl.figure(1)
#plot_corr_diag(corr_matrix, out_basename)
#pl.show()
## node indexes, not forget to subtract 1 
#[i, j, k , l, x0, y0 ]		=   node_index(corr_matrix)

data_matrix   = sb.load_matrix(input_name)
out_basename  = sb.get_dat_basename(input_name)
[u_matrix , T, dt, tvec]    =	fhn_timeseries(data_matrix)
corr_matrix                 =   correl_matrix(u_matrix, out_basename)
#[i, j, k , l]      =   node_index(corr_matrix)
#print i,j,k,l
## user defined time range for timeseries plots
#t_start = 1650
#t_final = 1700
## plot the timeseries of some good correlated nodes
#pl.figure(2)
#plot_timeseries(t_start, t_final, dt, u_matrix, tvec, i, j)
##plot the timeseries of some bad correlated nodes
#pl.figure(3)
#plot_timeseries(t_start, t_final, dt, u_matrix, tvec, k, l)
#pl.show()

# plot FAST FOURIER TRANSFORM of each node at surface
N_nodes = np.shape(corr_matrix)[1]
step = 25000
YFFT = []
N = []
F = []
for i in range(0,N_nodes):
	[yfft, freq] = fhn_fft(u_matrix, i, params['dt'])
	tmp = yfft[0:step]
	f_tmp = freq[0:step]
	YFFT = np.append(YFFT, tmp)
	for j in range(0, len(tmp)):
		N = np.append(N, np.array([i]))
		F = np.append(F, f_tmp[j])
		
# converting YFFT into logarithmic scale
FFT = np.log(YFFT)	

xi = np.arange(N.min(),N.max(),0.1)
yi = np.arange(F.min(),F.max(),0.01)

X , Y = np.meshgrid(xi,yi)
print np.shape(X), np.shape(Y)

Z = griddata((N,F), FFT, (xi[None,:], yi[:,None]), method='linear')

fig = pl.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X,Y,Z, cmap=cm.jet, linewidth=0)
ax.set_zlim3d(np.min(Z), np.max(Z))
fig.colorbar(surf)
pl.show()

#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')
##fig.colorbar(surf, shrink=0.5, aspect=5)
##ax.set_zlim(data3.min(), data3.max())
#fig.colorbar(surf, ax=ax)
#pl.show()
		
	

##[yfft_1, freq_1] = fhn_fft(u_matrix, rnd_node_1-1, params['dt'])
##[yfft_2, freq_2] = fhn_fft(u_matrix, rnd_node_2-1, params['dt'])
##pl.figure(5);
##pl.subplot(2,1,1)
##plot_timeseries(t_start, t_final, dt, u_matrix, tvec, rnd_node_1-1, rnd_node_2-1)
##pl.subplot(2,1,2)
##pl.plot(freq_1, yfft_1, 'r',label=('node '+str(rnd_node_1)))
##pl.plot(freq_2, yfft_2, 'b',label=('node '+str(rnd_node_2)))
##pl.setp(pl.gca().get_xticklabels(), fontsize = 15)
##pl.setp(pl.gca().get_yticklabels(), fontsize = 15)
##lg = legend()
##pl.title('Fourier Transformed Signal', fontsize=25)
##pl.xlabel('frequency [Hz]' , fontsize = 25 )
##pl.ylabel('timeseries (f)' , fontsize = 25 )
###pl.axis([-1, 70, 0, 0.5])

