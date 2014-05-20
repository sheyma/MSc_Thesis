#!/usr/bin/python2.7 

# -*- coding: utf-8 -*-

import numpy as np
import sys
import math 
import pylab as pl
from scipy.signal import butter, filtfilt	  
	  
def BOLD(T,r):
	
	# T : total simulation time [s]
	# r : neural time series to be simulated
	
	params = {
			'taus'   : 0.65,    
			'tauf'   : 0.41,   
			'tauo'   :  0.98,    
			'alpha'  :  0.32,
			'dt' : 0.01,	 ## change it to 0.001
			'Eo' : 0.34
			}

	itaus = float(1/params['taus'])
	itauf = float(1/params['tauf'])
	itauo = float(1/params['tauo'])
	ialpha = float(1/params['alpha'])
	dt = float(params['dt'])    
	Eo = float(params['Eo'])
	
	vo     = float(0.02);
	k1     = float(7) * params['Eo'] 
	k2     = float(2); 
	k3     = 2 * params['Eo']-float(0.2)

	ch_int = 0

	t0 = np.array(np.arange(0,(T+params['dt']),params['dt']))
	
	n_t = len(t0)
	
	t_min = 1		#t_min = 20 #use this one!

	n_min = round(t_min / params['dt'])

	r_max = max(r)
	# initial conditions
	x0 = np.array([0 , 1, 1, 1])

	if ch_int==0:
		
		# Euler's Method
		
		t = t0		
		x = np.zeros((n_t,4))
		x[0,:] = x0
		for n in range(0,n_t-1):
			x[n+1 , 0] = x[n ,0] + dt * (r[n] - itaus * x[n,0] - itauf * (x[n,1] -float(1.0))) 
			x[n+1 , 1] = x[n, 1] + dt * x[n,0]
			x[n+1 , 2] = x[n, 2] + dt * itauo * (x[n, 1] - pow(x[n, 2] , ialpha))
			x[n+1 , 3] = x[n, 3] + dt * itauo * ( x[n, 1] * (1.-pow((1- Eo),(1./x[n,1])))/Eo - pow(x[n,2],ialpha) * x[n,3] / x[n,2])
	# discard first n_min points	
        t_new = t[n_min -1 :]
	s     = x[n_min -1 : , 0]
	fi    = x[n_min -1 : , 1]
	v     = x[n_min -1 : , 2]
	q     = x[n_min -1 : , 3]
	b = 100/Eo * vo * ( k1 * (1-q) + k2 * (1-q/v) + k3 * (1-v) )
	return b  


def calcBOLD(simfile):
	print "input huge time series u's and v's: ", simfile 
	# load simfile as numpy matrix
	simout = np.transpose(np.loadtxt(simfile, unpack=True))
	# extract first column of simout as time vector
	Tvec = simout[:,[0]]
	# length of time time vector
	n_Tvec = len(Tvec) 
	# dt of time vector
	dt_Tvec = Tvec[1] - Tvec[0] 
	# total number of excitators: u's 
	N = (np.shape(simout)[1] -1 ) /2
	
	# extract time series of u's from simout
	timeseries = np.zeros((n_Tvec, N))
	print "size of timeseries : ", np.shape(timeseries)
	for row in range(0,N):
		timeseries[:,[row]] = simout[:,[2*row +1]]
	# store timeseries and time as .dat
	timeseries_app = np.c_[Tvec , timeseries]
	np.savetxt(simfile[:-4] + '_timeseries.dat', timeseries_app)	
	# 1st column is time vector, others are u series
	
	# plotting time series in a specific time interval
	t_start = 600;
	t_range = 400;
	
	fig = pl.figure(1)
	pl.plot(timeseries[t_start : (t_start + t_range) , :])
	pl.xlabel('t [ms]')
	pl.ylabel('$u_i(t)$')
	pl.savefig(simfile[:-4]+"_timeseries.eps",format="eps")
	#pl.show()

	# define simulation time for BOLD
	T = 10.0
	# apply Balloon Windkessel model in fuction BOLD
	Bold_signal = {}
	for col in range(0, N):
		Bold_signal[col] = BOLD(T, timeseries[:,[col]])
		# count the number of NaN 's in simulated BOLD
		count_nan = 0
		for key,value in enumerate(Bold_signal[col]):
			if value == float('nan'):
				count_nan += 1
				#print "u_N , key, value : "
				#print col,key,Bold_signal[key][col] 
		if count_nan > 0:
			print "u_N, nu. of NaNs:", col, count_nan
			
	# filtering below 0.25 Hz = cut-off frequency
	f_c = 0.25  
	# resolution of BOLD signal : dtt [second]
	dtt = 0.001  
	# length of one u series after subjected to BOLD 
	n_T = len(np.array(Bold_signal[1]))
	# sampling frequency
	f_s = 1/dtt
	# Nyquist frequency
	f_n = f_s /2
	# Butterworth filter
	#b , a = butter(5, f_c/f_n , btype = 'low')########
	b , a = butter(5, 0.5 , btype = 'lowpass', analog=False)
	#print b,a
	
	# Low pass filtering the BOLD signal
	Bold_filt = np.zeros((n_T , N))
	for col in range(0, N):
		Bold_filt[:, col] = filtfilt( b  , a , Bold_signal[col])
	# Downsampling : select one point at each 'ds' [ms]
	ds = 0.1  # use 2.5!!
	index = np.arange(0, n_T, int(ds/dtt))
	down_Bold_filt = Bold_filt[index , :]
	
	# Cut first and last seconds (distorted from filtering)
	len_Bold = np.shape(down_Bold_filt)[0]
	nFramesToKeep = 4   #   use 260!
	limit_down = int( math.floor( len_Bold - nFramesToKeep )/2 )
	limit_up = int( math.floor( len_Bold + nFramesToKeep )/2 ) 
	indice = np.arange(limit_down-1, limit_up-1  , 1) 
	#print indice
	# cut rows from down sampled Bold
	cut_Bold_filt = down_Bold_filt[indice, :]
	print np.corrcoef(np.matrix('5 6 5 ; 6 5 7'))
		
input_name = sys.argv[1]	
calcBOLD(input_name)

























