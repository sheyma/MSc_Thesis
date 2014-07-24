#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# calculating BOLD signal from netpy output

import numpy as np
import subprocess as sp
import sys
import math
import pylab as pl
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
	t_min = 20		# [s]

	n_min = round(t_min / iparams.dt)   # 2000 points to cut 
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

	print "reading data ..."
	simout = np.loadtxt(simfile)

	# extract time vector and dt
	tvec = simout[:,0]
	dt   = tvec[1] - tvec[0]
	T    = int(math.ceil( (tvec[-1])  / dt * params.dt ))

	# extract u-columns
	u_indices = np.arange(1, simout.shape[1] ,1)
	timeseries = simout[:, u_indices]
	
	print "extracted u-timeseries: shape =", timeseries.shape, ", dt = ", dt
	#np.savetxt('bold_timeseries_python.dat',timeseries,fmt='%.2f',delimiter='\t')
	
	return timeseries, T

def plot_timeseries(t_start , t_range , timeseries):
	# plots the timeseries in a specific time interval
	# t_range corresponds to time interval
	
	fig = pl.figure(1)
	pl.plot(timeseries[t_start : (t_start + t_range) , :])
	pl.xlabel('t [ms]')
	pl.ylabel('$u_i(t)$')
	#pl.savefig(simfile[:-4]+"_timeseries.eps",format="eps")
	#pl.show()
	return			
			
def calc_bold(bold_input , T):
	
	# applies Balloon Windkessel model to the timeseries
	# calculates the simulated bold signal
	# counts the number of NaN 's in simulated bold (error-check)
		
	N = np.shape(timeseries)[1] 	# total number of u columns		
	
	print "Bold-signalling of u-timeseries starts..."
	
	Bold_signal = {}				# type(Bold_signal) = <type 'dict'>
	for col in range(0, N):
		Bold_signal[col] = bold_euler(T, timeseries[:,[col]], iparams, x_init)
		#Bold_signal[col] = bold_ode(T, timeseries[:,[col]], iparams, x_init)
		count_nan = 0				
		for key,value in enumerate(Bold_signal[col]):
			if value == float('nan'):
				count_nan += 1
		if count_nan > 0:
			print "u_N, nu. of NaNs:", Bold_signal[key][col], count_nan

	#f = open('bold_signal_python.dat','w')	
	#for row in range( 0, len(Bold_signal[0]) ):
		#for key in Bold_signal.iterkeys():
			#f.write('%.6f\t' % ( Bold_signal[key][row] ))
		#f.write('\n')
	#f.close()
			
	return Bold_signal
		
def filter_bold(bold_input):
	
	# Butterworth low pass filtering of the simulated bold signal		
	# type(bold_input) = <type 'dict'>
	# f_c : cut-off freq., f_s : sampling freq., f_n : Nyquist freq.
	# Or : order of filter, dtt : resolution of bold signal
	 
	n_T = len(np.array(bold_input[1]))
	N   = len(bold_input.keys())
	Or  = 5
	dtt = 0.001							# [second]	
	f_c = 0.25					 		# [Hz]	
	f_s = 1/dtt							# [Hz]
	f_n = f_s /2						# [Hz]
	
	b , a = butter(Or,float(f_c)/f_n, btype='low',analog=False, output='ba')
		
	#f = open('Bs_python.dat','w')
	#for i in range(len(b)):
		#f.write("%.20f\t" % (b[i]))
	#f.close()	
	
	#f = open('As_python.dat','w')
	#for i in range(len(a)):
		#f.write("%.20f\t" % (a[i]))
	#f.close()	
	
	#b = (np.loadtxt('Bs_matlab.dat'))
	#a = (np.loadtxt('As_matlab.dat'))

	Bold_filt = np.zeros((n_T , N))
	for col in range(0,N):			
		Bold_filt[: , col] = filtfilt(b, a, bold_input[col])	
	
	#f = open('bold_filt_matlab_py.dat','w')
	
	#f = open('bold_filt_python_py.dat','w')
	#for row in range(0, np.shape(Bold_filt)[0]):
		#for col in range(0 , np.shape(Bold_filt)[1]):
			#f.write("%.6f\t" % (Bold_filt[row, col]))
		#f.write("\n")
	#f.close()
		
	return Bold_filt

	
def down_sample(bold_input, ds, dtt):
	
	# downsampling of the filtered bold signal
	# select one point every 'ds' [ms] to match fmri resolution

	n_T = np.shape(bold_input)[0] 
	index = np.arange(0 , n_T , int(ds/dtt))
	down_bold = bold_input[index, :]
	
	#np.savetxt('bold_down_python.dat', down_bold,'%.6f',delimiter='\t')
	
	return down_bold
						
						
def keep_frames(bold_input, cut_percent):
	
	# cut array from beginning and end (distorted from filtering)
	
	length  = np.shape(bold_input)[0]
	limit_down = int(math.ceil(length * cut_percent) -1) 
	limit_up   = int(length - limit_down -1)	
	index      = np.arange(limit_down, limit_up-2 , 1)
	print "index : ", index
	cut_bold   = bold_input[index, :]
	
	#np.savetxt('bold_cut_python.dat', cut_bold,'%.6f',delimiter='\t')
	
	return cut_bold

def correl(bold_input):
	
	#find correlation coefficient matrix
	
	col = np.shape(bold_input)[1]
	correl_matrix = np.zeros((col , col))
		
	for i in range(0,col) :
		correl_row = np.array([])			
		
		for j in range(0,col):
			A = np.corrcoef(bold_input[:,i] , bold_input[:,j])	
			correl_row = np.append(correl_row, A[0,1])
		
		correl_matrix[i,:] = correl_row	
	#correl_matrix = np.corrcoef(bold_input)
	#np.savetxt('bold_corr_python.dat', correl_matrix, '%.10f',delimiter='\t')

	return correl_matrix

def image(bold_input, simfile):
	
	# plots simulated functional connectivity
	
	N_col = np.shape(bold_input)[1]
	extend = (0.5 , N_col+0.5 , N_col+0.5, 0.5 )	
	pl.imshow(bold_input, interpolation='nearest', extent=extend)
	pl.colorbar()
	
	image_name = simfile[0:-4] + '_CORR.eps'	
	#pl.savefig(image_name, format="eps")
	pl.show()
	return  
	

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

t_start = 325000;
t_range = 500;
ds = 2.3  #### NEEDS TO BE CHECKED!
dtt = 0.001
cut_percent = float(2) / 100

iparams = invert_params(params)

x_init = np.array([0 , 1, 1, 1])	# initial conditions	

input_name = sys.argv[1]

# handle xz files transparently
if input_name.endswith(".xz"):
	# non-portable but we don't want to depend on pyliblzma module
	xzpipe = sp.Popen(["xzcat", input_name], stdout=sp.PIPE)
	infile = xzpipe.stdout
else:
	# in non-xz case we just use the file name instead of a file object, numpy's
	# loadtxt() can deal with this
	infile = input_name

# "infile" can only used one time because it might be a pipe"!

[timeseries, T] = fhn_timeseries(infile)

print "T : " , T, " [seconds]"

fhn_image       =   plot_timeseries(t_start , t_range , timeseries)

bold_signal 	=   calc_bold(timeseries, T)


bold_filt		=   filter_bold(bold_signal)

#bold_filt       =   np.loadtxt('bold_filt_matlab.dat')


bold_down  		=   down_sample(bold_filt , ds, dtt)

#bold_down  = np.loadtxt('bold_down_matlab.dat')

bold_cut 		= 	keep_frames(bold_down ,cut_percent)

##bold_cut = np.loadtxt('bold_cut_matlab.dat')
correl_matrix 	= 	correl(bold_cut)
corr_image		= 	image(correl_matrix , input_name)

pl.show()


#######################################



#bold_euler(T , R[1, :], iparams, x_init)

#r_t = R[0,:]

#bold_ode(T, R[1,:], iparams, x_init)
