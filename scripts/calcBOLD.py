#!/usr/bin/python2.7

# -*- coding: utf-8 -*-

import numpy as np
import sys
import math
import pylab as pl
from scipy.signal import butter, filtfilt
import scipy
import scipy.integrate as integ
from scipy.integrate import odeint


class Params(object):
	__slots__ = ['taus', 'tauf', 'tauo', 'alpha', 'dt', 'Eo', 'vo', 'k1', 'k2', 'k3']

def invert_params(params):
	params.taus = float(1/params.taus)
	params.tauf = float(1/params.tauf)
	params.tauo = float(1/params.tauo)
	params.alpha = float(1/params.alpha)
	return params

def bold_euler(T, r, iparams, x_init):
	# T : total simulation time [s]
	# r : neural time series to be simulated
	dt = iparams.dt
	
	# create a time array
	t = np.array(np.arange(0,(T+iparams.dt),iparams.dt))
	n_t = len(t)
	t_min = 1		#t_min = 20 #use this one!

	n_min = round(t_min / iparams.dt)
	
	r_max = np.amax(r)
	
	x = np.zeros((n_t,4))
	
	x[0,:] = x_init
	for n in range(0,n_t-1):
		print "n is ", n, r[n]
		x[n+1 , 0] = x[n ,0] + dt * (r[n] - iparams.taus * x[n,0] - iparams.tauf * (x[n,1] -float(1.0)))
		x[n+1 , 1] = x[n, 1] + dt * x[n,0]
		x[n+1 , 2] = x[n, 2] + dt * iparams.tauo * (x[n, 1] - pow(x[n, 2] , iparams.alpha))
		x[n+1 , 3] = x[n, 3] + dt * iparams.tauo * ( x[n, 1] * (1.-pow((1- iparams.Eo),(1./x[n,1])))/iparams.Eo - pow(x[n,2],iparams.alpha) * x[n,3] / x[n,2])
		
	# discard first n_min points	
	t_new = t[n_min -1 :]
	s     = x[n_min -1 : , 0]
	fi    = x[n_min -1 : , 1]
	v     = x[n_min -1 : , 2]
	q     = x[n_min -1 : , 3]
	b= 100/iparams.Eo * iparams.vo * ( iparams.k1 * (1-q) + iparams.k2 * (1-q/v) + iparams.k3 * (1-v) )
	
	# plot b over time
	pl.xlabel('t')
	pl.ylabel('BOLD signal')
	pl.plot(t_new,b[:],'g-')
	pl.show()
		
	return b

def bold_ode_eqns(X, t, T, r, iparams):
	
	x0, x1, x2, x3 = X
	tmp = r[r_t <= t]	
	r_index = len(tmp)-1
	if r_index < (len(r) - 1) and (r_t[r_index+1] - t) < (t - r_t[r_index]):
		r_index += 1

	print "t : " ,t,  "r_index :", r_index, "r[n] " , r[r_index]
	
	if (t % 1) < 0.0001:
		t_now = time.time()
		print 'seconds: %f => minutes %f to simulate %.1f time units of %f' % ((t_now-t_start),(t_now-t_start)/60., t, T)
	return [r[r_index] - iparams.taus * x0 - iparams.tauf * (x1 - float(1.0) ),
		x0,
		iparams.tauo * ( x1 - pow(x2 , iparams.alpha) ),
		iparams.tauo * ( x1 * (1.-pow((1.- iparams.Eo),(1./x1)))/iparams.Eo - pow(x2, iparams.alpha) * x3 / x2)]	


def bold_ode(T, r, iparams, x_init):

	N = T/iparams.dt
	t = np.linspace(0, T-iparams.dt, N)

	print "starting BOLD calculation..."

	sol = odeint(bold_ode_eqns, x_init, t, args=(T, r, iparams))

	b = 100/iparams.Eo * iparams.vo * ( iparams.k1 * (1-sol[:,3]) + iparams.k2 * (1-sol[:,3]/sol[:,2]) + iparams.k3 * (1-sol[:,2]) )
	
	pl.xlabel('t')
	pl.ylabel('BOLD signal')
	pl.plot(t,b[:],'g-')
	pl.show()

	return b
	
			
			
	
		
#def calcBOLD(simfile):
	#print "input huge time series u's and v's: ", simfile
	## load simfile as numpy matrix
	#simout = np.transpose(np.loadtxt(simfile, unpack=True))
	## extract first column of simout as time vector
	#Tvec = simout[:,[0]]
	## length of time time vector
	#n_Tvec = len(Tvec)
	## dt of time vector
	#dt_Tvec = Tvec[1] - Tvec[0]
	## total number of excitators: u's
	#N = (np.shape(simout)[1] -1 ) /2
	
	## extract time series of u's from simout
	#timeseries = np.zeros((n_Tvec, N))
	#print "size of timeseries : ", np.shape(timeseries)
	#for row in range(0,N):
		#timeseries[:,[row]] = simout[:,[2*row +1]]
	## store timeseries and time as .dat
	#timeseries_app = np.c_[Tvec , timeseries]
	#np.savetxt(simfile[:-4] + '_timeseries.dat', timeseries_app)	
	## 1st column is time vector, others are u series
	
	## plotting time series in a specific time interval
	#t_start = 600;
	#t_range = 400;
	
	#fig = pl.figure(1)
	#pl.plot(timeseries[t_start : (t_start + t_range) , :])
	#pl.xlabel('t [ms]')
	#pl.ylabel('$u_i(t)$')
	#pl.savefig(simfile[:-4]+"_timeseries.eps",format="eps")
	##pl.show()

	## define simulation time for BOLD
	#T = 10.0		# use this :  T = 700.0 [s]
	## apply Balloon Windkessel model in fuction BOLD
	#Bold_signal = {}
	#for col in range(0, N):
		#Bold_signal[col] = BOLD(T, timeseries[:,[col]])
		#print "timeseries vector used in bOLD function", timeseries[:,col]
		## count the number of NaN 's in simulated BOLD
		#count_nan = 0
		#for key,value in enumerate(Bold_signal[col]):
			#if value == float('nan'):
				#count_nan += 1
				##print "u_N , key, value : "
				##print col,key,Bold_signal[key][col]
		#if count_nan > 0:
			#print "u_N, nu. of NaNs:", col, count_nan
			
	## filtering below 0.25 Hz = cut-off frequency
	#f_c = 0.25
	## resolution of BOLD signal : dtt [second]
	#dtt = 0.001
	## length of one u series after subjected to BOLD
	#n_T = len(np.array(Bold_signal[1]))
	## sampling frequency
	#f_s = 1/dtt
	## Nyquist frequency
	#f_n = f_s /2
	## Butterworth filter
	##b , a = butter(5, f_c/f_n , btype = 'low')########
	#b , a = butter(5, 0.5 , btype = 'lowpass', analog=False)
	##print b,a
	
	## Low pass filtering the BOLD signal
	#Bold_filt = np.zeros((n_T , N))
	#for col in range(0, N):
		#Bold_filt[:, col] = filtfilt( b  , a , Bold_signal[col])
	## Downsampling : select one point at each 'ds' [ms]
	#ds = 0.1  # use 2.5!!
	#index = np.arange(0, n_T, int(ds/dtt))
	#down_Bold_filt = Bold_filt[index , :]
	
	## Cut first and last seconds (distorted from filtering)
	#len_Bold = np.shape(down_Bold_filt)[0]
	#nFramesToKeep = 4   #   use 260!
	#limit_down = int( math.floor( len_Bold - nFramesToKeep )/2 )
	#limit_up = int( math.floor( len_Bold + nFramesToKeep )/2 )
	#indice = np.arange(limit_down-1, limit_up-1  , 1)
	##print indice
	## cut rows from down sampled Bold
	#cut_Bold_filt = down_Bold_filt[indice, :]
	##print cut_Bold_filt
	## find correlation coefficient matrix
	#simcorr = scipy.corrcoef(np.transpose(cut_Bold_filt))
	##print simcorr
	#np.savetxt(simfile[:-4] + '_simcorr.dat', simcorr)
	
	#fig = pl.figure(2)
	#pl.imshow(simcorr, interpolation='nearest', extent=[0.5, 2.5, 0.5, 2.5])
	#pl.colorbar
	##pl.show()

# here we go

params = Params()
params.taus = 0.65
params.tauf = 0.41
params.tauo = 0.98
params.alpha  = 0.32
params.dt = 0.1  # check it!!!
params.Eo = 0.34
params.vo = 0.02;
params.k1 = 7.0 * params.Eo
params.k2 = 2.0
params.k3 = 2.0 * params.Eo - 0.2

iparams = invert_params(params)

# initial conditions	
x_init = np.array([0 , 1, 1, 1])	



	
input_name = sys.argv[1]	
print "reading data..."
R = np.loadtxt(input_name, unpack=True)

T =90
#bold_euler(T , R[1, :], iparams, x_init)

r_t = R[0,:]
bold_ode(T, R[1,:], iparams, x_init)







