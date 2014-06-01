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
import time  

class Params(object):
	__slots__ = ['taus', 'tauf', 'tauo', 'alpha', 'dt', 'Eo']

def invert_params(params):
	iparams = Params()
	iparams.taus = float(1/params.taus)   
	iparams.tauf = float(1/params.tauf)  
	iparams.tauo = float(1/params.tauo)    
	iparams.alpha = float(1/params.alpha)
	iparams.dt = float(params.dt)
	iparams.Eo = float(params.Eo)
	return iparams

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


def bold_ode(T, r, iparams):
    
	N = T/iparams.dt
	t = np.linspace(0, T-iparams.dt, N)

	print "starting BOLD calculation..."

	sol = odeint(bold_ode_eqns, init_con[:], t, args=(T, r, iparams))

	b = 100/iparams.Eo * vo * ( k1 * (1-sol[:,3]) + k2 * (1-sol[:,3]/sol[:,2]) + k3 * (1-sol[:,2]) )
	
	pl.xlabel('t')
	pl.ylabel('BOLD signal')
	pl.plot(t,b[:],'g-') 
	pl.show()

	return b

t_start = time.time()

params = Params()
params.taus = 0.65
params.tauf = 0.41
params.tauo = 0.98
params.alpha  = 0.32
params.dt = 0.001
params.Eo = 0.34

iparams = invert_params(params)

vo     = float(0.02);
k1     = float(7) * params.Eo 
k2     = float(2); 
k3     = 2 * params.Eo-float(0.2)

init_con = [0., 1.0, 1.0, 1.0]

T = 90
print "reading data..."


input_name = sys.argv[1]
R = np.loadtxt(input_name, unpack=True)
# extract time array from R 
r_t = R[0,:]
r = R[1,:]

# find maximum of time in R
r_t_max = np.amax(r_t)   # 99.9
print "max time in r : ",  r_t_max 

if r_t_max < T:
	print "simulation time T is greater than r-dimension !!!"
	sys.exit(1)

t_now = time.time()
print 'seconds: %f => minutes %f to read the data' % ((t_now-t_start),(t_now-t_start)/60.)

bold_ode( T, r, iparams)

#print b
t_now = time.time()
print 'seconds: %f => minutes %f to simulate' % ((t_now-t_start),(t_now-t_start)/60.)

