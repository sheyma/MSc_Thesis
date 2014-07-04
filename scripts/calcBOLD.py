#!/usr/bin/python2.7

# -*- coding: utf-8 -*-

import numpy as np
import sys
import math
import pylab as pl
#import scipy.signal import butter, filtfilt, lfilter
from scipy.signal import  butter , filtfilt
import scipy.integrate as integ
from scipy.integrate import odeint
import time
#from sos import tf2sos, sosfilt
#import filter_design_coeff

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
	
	#dt = float(T) / len(r)
	
	
	# create a time array
	t = np.array(np.arange(0,(T+iparams.dt),iparams.dt))
	n_t = len(t)
	t_min = 1		#t_min = 20 #use this one!

	n_min = round(t_min / iparams.dt)
	
	r_max = np.amax(r)
	
	x = np.zeros((n_t,4))
	
	x[0,:] = x_init
	for n in range(0,n_t-1):
		
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
	pl.ylabel('BOLD signal euler')
	pl.plot(t_new,b[:],'g-')
	#pl.show()
	
	print "Euler's dt is : " , dt
		
	return b

def bold_ode_eqns(X, t, T, r, iparams):
	
	x0, x1, x2, x3 = X
	tmp = r[r_t <= t]	
	r_index = len(tmp)-1
	if r_index < (len(r) - 1) and (r_t[r_index+1] - t) < (t - r_t[r_index]):
		r_index += 1

	print "t : " ,t,  "r_index :", r_index, "r[n] " , r[r_index]
	
	#if (t % 1) < 0.0001:
		#t_now = time.time()
		#print 'seconds: %f => minutes %f to simulate %.1f time units of %f' % ((t_now-t_start),(t_now-t_start)/60., t, T)
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
	pl.ylabel('BOLD signal ode')
	pl.plot(t,b[:],'g-')
	pl.show()

	return b
	
		
			
	
		
def calcBOLD(simfile):
	print "input huge time series u's and v's: ", simfile
	print "reading data ..."
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
	print "size of extracted u-timeseries : ", np.shape(timeseries)
	for row in range(0,N):
		timeseries[:,[row]] = simout[:,[2*row +1]]
	
	np.savetxt(simfile[:-4] + '_timeseries.dat', timeseries)	
	
	# plotting time series in a specific time interval
	t_start = 325000;
	t_range = 500;
	
	fig = pl.figure(1)
	pl.plot(timeseries[t_start : (t_start + t_range) , :])
	pl.xlabel('t [ms]')
	pl.ylabel('$u_i(t)$')
	#pl.savefig(simfile[:-4]+"_timeseries.eps",format="eps")
	#pl.show()

	print "Bold-signalling of u-timeseries starts..."
	# !!!!!!!!!define simulation time for BOLD
	T = 700.0		
	# apply Balloon Windkessel model in fuction BOLD
	
	Bold_signal = {}
	for col in range(0, N):
		Bold_signal[col] = bold_euler(T, timeseries[:,[col]], iparams, x_init)
		#print "timeseries vector used in BOLD function", timeseries[:,col]
		#print "BOLD euler signal" , Bold_signal[col]
		
		# count the number of NaN 's in simulated BOLD
		count_nan = 0
		for key,value in enumerate(Bold_signal[col]):
			if value == float('nan'):
				count_nan += 1
				#print "u_N , key, value : "
				#print col,key,Bold_signal[key][col]
		if count_nan > 0:
			print "u_N, nu. of NaNs:", count_nan
		
		
			
	## filtering below 0.25 Hz = cut-off frequency
	f_c = 0.25
	# resolution of BOLD signal : dtt [second]
	dtt = 0.001
	# length of one u series after subjected to BOLD
	n_T = len(np.array(Bold_signal[1]))
	print "length of one column in Bold_signal : " , n_T
	# Sampling frequency [Hz]
	f_s = 1/dtt
	# Nyquist frequency [Hz]
	f_n = f_s /2
	print "Butterworth lowpass filter..."
	print "sampling freq : " , f_s, "Hz," "   nyquist frequency : ", f_n , "Hz"
	
	# Butterworth filter
	#b , a = filter_design_coeff.butter(5, f_c/f_n , btype = 'low')
	b , a = butter(5, float(f_c)/f_n , btype = 'low', analog=0, output='ba')
	
	print 'fc/fN' , float(f_c)/f_n 
	
	
	#f = open('Bs_python.dat','w')
	#for i in range(len(b)):
		#f.write("%.20f\t" % (b[i]))
	#f.close()	

	#f = open('As_python.dat','w')
	#for i in range(len(a)):
		#f.write("%.20f\t" % (a[i]))
	#f.close()	
	
	
	# Low pass filtering the BOLD signal

	Bold_filt = np.zeros((n_T , N))

	f = open('bold_filt_python.dat','w')
		
	for col in range(0,N):
		
		# for N= 1 : try :
		#f = open('bold_signal_python.dat','w')
		#for i in range(len(Bold_signal[col])):
			#f.write("%.6f\t" % (Bold_signal[col][i]))
		#f.close()		
				
		
		Bs = (np.loadtxt('Bs_matlab.dat'))
		As = (np.loadtxt('As_matlab.dat'))
		
		a
		Bold_filt[: , col] = filtfilt(Bs, As, Bold_signal[col])		
		
		
	print "size(Bold_filt) : " , np.shape(Bold_filt)
	
	f = open('bold_filt_python.dat','w')
	for row in range(0, np.shape(Bold_filt)[0]):
		for col in range(0 , np.shape(Bold_filt)[1]):
			f.write("%.6f\t" % (Bold_filt[row, col]))
		f.write("\n")
	f.close()
	
		
			
	# Downsampling : select one point at each 'ds' [ms]
	ds = 2.5  # use 2.5!!
	index = np.arange(0, n_T, int(ds/dtt))
	down_Bold = Bold_filt[index , :]
	print "Down_sampled Bold_filt size : " , np.shape(down_Bold)
	
	# Cut first and last seconds (distorted from filtering)
	len_down = np.shape(down_Bold)[0]
	nFramesToKeep = 260   #   use 260!
	limit_down = int( math.floor( len_down - nFramesToKeep )/2 )
	limit_up = int( math.floor( len_down + nFramesToKeep )/2 )
	print "limit_down" , limit_down
	print "limit_up" , limit_up
	#indice = np.arange(limit_down-1, limit_up-1  , 1)
	#print "indice" , indice
	# cut rows from down sampled Bold
	cut_Bold = down_Bold[limit_down : limit_up, :]
	print "shape of cut_Bold : " , np.shape(cut_Bold)
	
	
	f = open('cut_bold.dat','w')
	for row in range(0, np.shape(cut_Bold)[0]):
		for col in range(0 , np.shape(cut_Bold)[1]):
			f.write("%.6f\t" % (cut_Bold[row, col]))
		f.write("\n")
	f.close()
	
	
	# find correlation coefficient matrix
	#simcorr = scipy.corrcoef(np.transpose(cut_Bold))
	#print simcorr
	##np.savetxt(simfile[:-4] + '_simcorr.dat', simcorr)
	
	#fig = pl.figure(2)
	#pl.imshow(simcorr, interpolation='nearest', extent=[0.5, 2.5, 0.5, 2.5])
	#pl.colorbar
	#pl.show()

# here we go

params = Params()
params.taus = 0.65
params.tauf = 0.41
params.tauo = 0.98
params.alpha  = 0.32
params.dt = 0.001  # check it!!!
params.Eo = 0.34
params.vo = 0.02;
params.k1 = 7.0 * params.Eo
params.k2 = 2.0
params.k3 = 2.0 * params.Eo - 0.2

iparams = invert_params(params)

# initial conditions	
x_init = np.array([0 , 1, 1, 1])	



	
input_name = sys.argv[1]	
#print "reading data..."
#R = np.loadtxt(input_name, unpack=True)

#T = 700.0

#bold_euler(T , R[1, :], iparams, x_init)

#r_t = R[0,:]
#bold_ode(T, R[1,:], iparams, x_init)

calcBOLD(input_name)

#Bold_filt = scipy.signal.filtfilt( b  , a , np.array([1,2,3,4,5,6,7,8,9,10]), padlen=9  )
#print Bold_filt



#input_signal = np.linspace(1, 50 ,50)

#passband = [0.75*2/30, 5.0*2/30]
#b, a = sig.butter(5, 0.05, 'lowpass')



#y = sig.filtfilt(b, a, input_signal)
#y_ma = np.loadtxt('deleted.dat' , unpack=True)

#pl.plot(y)
#pl.plot(y_ma ,'r')
#pl.show()




#simout = np.transpose(np.loadtxt(input_name, unpack=True))


#from scipy import signal
#b, a = signal.butter(8, 0.125)
#y = signal.filtfilt(b, a, simout, padlen=None, padtype='odd')
#print "b : " , b
#print "a :  " , a
##print y






