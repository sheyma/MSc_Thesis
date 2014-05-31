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


def Bold_eqns(X, t, itaus, itauf, itauo, ialpha, Eo, r, T, dt):
	x0, x1, x2, x3 = X
	print "t : " ,t, "dt :" , dt , "r_index :" ,[min(round(t*1000),(T-dt)*1000)]
	
	file_dbg_Bold_eqns.write("%f\t%f\t%f\n" % (
		t , min(round(t*1000),(T-dt)*1000),
		r[min(round(t*1000),(T-dt)*1000)])    )

	
	if (t % 1) < 0.0001:
	  t_now = time.time()
	  print 'seconds: %f => minutes %f to simulate %.1f time units of %f' % ((t_now-t_start),(t_now-t_start)/60., t, T)
	return [r[min(round(t*1000),(T-dt)*1000)] - itaus*x0 - itauf * (x1 - float(1.0) ),
		x0,
		itauo * ( x1 - pow(x2 , ialpha) ),
		itauo * ( x1 * (1.-pow((1.- Eo),(1./x1)))/Eo - pow(x2,ialpha) * x3 / x2)]	


t_start = time.time()

params = {
	'taus'   : 0.65,    
	'tauf'   : 0.41,   
	'tauo'   :  0.98,    
	'alpha'  :  0.32,
	'dt' : 0.01, #dt = 0.001
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

init_con = [0., 1.0, 1.0, 1.0]

print "reading data..."
input_name = sys.argv[1]
R = np.loadtxt(input_name, unpack=True)
r = R[1,:]
#R = np.transpose(np.loadtxt(input_name, unpack=True))
#r = R[:,1]

t_now = time.time()
print 'seconds: %f => minutes %f to read the data' % ((t_now-t_start),(t_now-t_start)/60.)

# T = 300.0
T = 10.0
N = T/params['dt']
t = np.linspace(0, T-params['dt'], N)

print "starting BOLD calculation..."
file_dbg_Bold_eqns = open('r_index_ode.dat','w')

sol = odeint(Bold_eqns, init_con[:], t, args=(itaus, itauf, itauo, ialpha, Eo, r, T, params['dt']))

file_dbg_Bold_eqns.close()

b = 100/Eo * vo * ( k1 * (1-sol[:,3]) + k2 * (1-sol[:,3]/sol[:,2]) + k3 * (1-sol[:,2]) )
#print b
t_now = time.time()
print 'seconds: %f => minutes %f to simulate' % ((t_now-t_start),(t_now-t_start)/60.)

pl.xlabel('t')
pl.ylabel('BOLD signal')
pl.plot(t,b[:],'g-') 

#pl.show()
