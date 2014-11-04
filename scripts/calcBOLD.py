#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# calculating BOLD signal from netpy output

import sb_utils as sb
import numpy as np
import sys
import math
import pylab as pl
import scipy.stats as sta
from scipy.signal import  butter , filtfilt , correlate2d
import scipy.integrate as integ
from scipy.integrate import odeint
import time



class Params(object):
	__slots__ = ['taus', 'tauf', 'tauo', 'alpha', 'dt', 'Eo', 'vo', 'k1', 'k2', 'k3']

def invert_params(params):
	params.taus = float(1/params.taus)
	params.tauf = float(1/params.tauf)
	params.tauo = float(1/params.tauo)
	params.alpha = float(1/params.alpha)
	return params

def bold_euler(T, r, iparams, x_init):
	
	# Baloon-Windkessel model with Euler's method
	
	# T : total simulation time [s]
	# r : neural time series to be simulated
	
	dt  = iparams.dt
	#dt = float(T) / len(r) !!!!!!!!!!!
	
	t = np.array(np.arange(0,(T+iparams.dt),iparams.dt))
	n_t = len(t)
	# cut BOLD signal from beginning bcs of transient behavior
	t_min = 20		# [s] CHECK FOR THE SIGNAL 

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
	
	# Balloon-Windkessel model with integrate.odeint
	
	N = T/iparams.dt
	t = np.linspace(0, T-iparams.dt, N)
	
	sol = odeint(bold_ode_eqns, x_init, t, args=(T, r, iparams))

	b = 100/iparams.Eo * iparams.vo * ( iparams.k1 * (1-sol[:,3]) + iparams.k2 * (1-sol[:,3]/sol[:,2]) + iparams.k3 * (1-sol[:,2]) )
	
	return b
	
		
def fhn_timeseries(simfile):

	# load simfile as numpy matrix
	# extract first column of simout as time vector
	# read u_i time series from simout
	
	simout = sb.load_matrix(simfile)
	
	# extract time vector and dt
	tvec = simout[:,0]
	dt   = tvec[1] - tvec[0]
	T    = int(math.ceil( (tvec[-1])  / dt * params.dt ))

	# extract u-columns
	u_indices = np.arange(1, simout.shape[1] ,1)
	timeseries = simout[:, u_indices]
	
	print "extracted u-timeseries: shape =", timeseries.shape, ", dt = ", dt
	#np.savetxt('u_timeseries_python.dat',timeseries,fmt='%.6f',delimiter='\t')
	
	return timeseries, T

def plot_timeseries(t_start , t_range , timeseries):
	# plots the timeseries in a specific time interval
	# t_range corresponds to time interval
	
	pl.plot(timeseries[t_start : (t_start + t_range) , :])
	pl.xlabel('t [ms]')
	pl.ylabel('$u_i(t)$')
	#pl.savefig(simfile[:-4]+"_timeseries.eps",format="eps")
	#pl.show()
	return			

# standart normalization of timeseries (MATLAB's zscore)			
def normalize_timeseries(timeseries):
	N_timeseries = sta.mstats.zscore(timeseries)
	return N_timeseries
			
def calc_bold(timeseries , T, out_basename):
	
	# applies Balloon Windkessel model to the timeseries
	# calculates the simulated bold signal
	# counts the number of NaN 's in simulated bold (error-check)
		
	N = np.shape(timeseries)[1] 	# total number of u columns		
	
	print "Bold-signalling of u-timeseries starts..."
	# type(Bold_signal) = <type 'dict'>
	Bold_signal = {}				
	for col in range(0, N):
		Bold_signal[col] = bold_euler(T, timeseries[:,[col]], iparams, x_init)
		#Bold_signal[col] = bold_ode(T, timeseries[:,[col]], iparams, x_init)
		count_nan = 0				
		for key,value in enumerate(Bold_signal[col]):
			if value == float('nan'):
				count_nan += 1
		if count_nan > 0:
			print "u_N, nu. of NaNs:", Bold_signal[key][col], count_nan
	# exporting BOLD signal 
	file_name = str(out_basename + '_Norm_BOLD_signal.dat')
	print file_name
	f = open(file_name,'w')	
	for row in range( 0, len(Bold_signal[0]) ):
		for key in Bold_signal.iterkeys():
			f.write('%.6f\t' % ( Bold_signal[key][row] ))
		f.write('\n')
	f.close()
			
	return Bold_signal

def plot_bold_signal(T, bold_input):
	# plots the bold_signal obtained from Balloon-Windkessel model
		
	time = np.linspace(0, T, len(bold_input[0]) )	
	
	for key in bold_input.keys():
		print "KEY " , key
		print "bold_input[key]", bold_input[key]
		pl.plot( time , bold_input[key])
	pl.xlabel('t [s]')
	pl.ylabel('$bold signal,  u_i(t)$')
	#pl.savefig(simfile[:-4]+"_bold_signal.eps",format="eps")
	#pl.show()
	return	

		
def filter_bold(bold_input , out_basename):
	
	# Butterworth low pass filtering of the simulated bold signal		
	# type(bold_input) = <type 'dict'>
	# f_c : cut-off freq., f_s : sampling freq., f_n : Nyquist freq.
	# Or : order of filter, dtt : resolution of bold signal
	
	print "low pass filtering is applied..." 
	 
	n_T = len(np.array(bold_input[1]))
	N   = len(bold_input.keys())
	Or  = 5
	dtt = 0.001							# [second]	
	f_c = 0.25					 		# [Hz]	
	f_s = 1/dtt							# [Hz]
	f_n = f_s /2						# [Hz]
	
	# calculate butterworth coefficients with Python's butter function
	#b , a = butter(Or,float(f_c)/f_n, btype='low',analog=False, output='ba')
	
	# import butterworth coefficients from MATLAB results
	b   = np.loadtxt('Bs_matlab.dat')
	a   = np.loadtxt('As_matlab.dat')

	Bold_filt = np.zeros((n_T , N))
	for col in range(0,N):			
		Bold_filt[: , col] = filtfilt(b, a, bold_input[col])
			
	file_name = str(out_basename + '_BOLD_filtered.dat')
	print "file_name : " , file_name
	np.savetxt(file_name, Bold_filt,'%.6f',delimiter='\t')
	return Bold_filt


def plot_bold_filt(bold_input):
	# plots the low-pass filtered bold_signal 
	fig = pl.figure(3)
	pl.plot( bold_input[0:-1 , :])
	pl.xlabel('t [s]')
	pl.ylabel('$filtered bold signal,  u_i(t)$')
	#pl.savefig(simfile[:-4]+"_bold_filt.eps",format="eps")
	#pl.show()
	return	
	
	
def down_sample(bold_input, ds, dtt, out_basename):
	
	# downsampling of the filtered bold signal
	# select one point every 'ds' [ms] to match fmri resolution

	print "downsampling..."
	n_T = np.shape(bold_input)[0] 
	index = np.arange(0 , n_T , int(ds/dtt))
	down_bold = bold_input[index, :]
	
	#file_name = str(out_basename + '_BOLD_ds.dat')
	#print "file_name : " , file_name
	#np.savetxt(file_name, down_bold,'%.6f',delimiter='\t')
	
	return down_bold
						
						
def keep_frames(bold_input, cut_percent, out_basename):
	
	# cut array from beginning and end (distorted from filtering)
	
	print "cut the distorted parts of signal..."
	length  = np.shape(bold_input)[0]
	limit_down = int(math.ceil(length * cut_percent) -1) 
	limit_up   = int(length - limit_down -1)	
	index      = np.arange(limit_down, limit_up , 1)
	cut_bold   = bold_input[index, :]
	# exporting downsampled + begin./end cut signal !
	file_name = str(out_basename + '_BOLD_bds.dat')
	np.savetxt(file_name, cut_bold,'%.6f',delimiter='\t')	
	return cut_bold

# here we go

params = Params()
params.taus = 0.65
params.tauf = 0.41
params.tauo = 0.98
params.alpha  = 0.32
params.dt = 0.001
params.Eo = 0.34
params.vo = 0.02;
params.k1 = 7.0 * params.Eo
params.k2 = 2.0
params.k3 = 2.0 * params.Eo - 0.2

t_start = 0;
t_range = 10000;
ds = 2.3  
dtt = 0.001
cut_percent = float(2) / 100

iparams = invert_params(params)
# initial conditions for the bold differential equations
x_init = np.array([0 , 1, 1, 1])		

input_name      = sys.argv[1]

out_basename    = sb.get_dat_basename(input_name)

[timeseries, T] = fhn_timeseries(input_name)

print "T : " , T, " [seconds]"

Norm_timeseries   = normalize_timeseries(timeseries)

#pl.figure(1)
#plot_timeseries(t_start , t_range , timeseries)
#pl.figure(2)
#pl.plot(N_timeseries)
#pl.figure(3)
#plot_timeseries(t_start , t_range , N_timeseries)

#bold_signal     =   calc_bold(timeseries, T, out_basename)

Norm_bold_signal   =   calc_bold(Norm_timeseries, T, out_basename)

#pl.figure(5)
#pl.plot(bold_signal[1])
#plot_bold_signal(T , bold_signal)
#pl.figure(6)
#pl.plot(bold_signal_N[1])
#plot_bold_signal(T , Norm_bold_signal)


pl.show()
### THIS NEEDS TO BE CHANGED !!! ################## 
#timeseries      =   np.loadtxt(input_name)
#T = 550.0

#bold_signal     =   calc_bold(timeseries, T, out_basename)

#signal_image    =   plot_bold_signal(T , bold_signal)

#bold_filt		=   filter_bold(bold_signal, out_basename)
#bold_filt       =   np.loadtxt('bold_filt_matlab.dat')
#filt_image		=   plot_bold_filt(bold_filt)

#bold_down =  down_sample(bold_filt , ds, dtt, out_basename)

#bold_cut = keep_frames(bold_down ,cut_percent, out_basename)

#pl.show()

#######################################

#bold_euler(T , R[1, :], iparams, x_init)

#r_t = R[0,:]

#bold_ode(T, R[1,:], iparams, x_init)
