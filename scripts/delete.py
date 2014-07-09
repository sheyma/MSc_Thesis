#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np
import sys
import pylab as pl
from scipy.signal import  butter , filtfilt

Bold_signal = np.loadtxt('bold_signal_python.dat')

n_T = np.shape(Bold_signal)[0]
N   = np.shape(Bold_signal)[1]
print "shape of Bold_signal : " , np.shape(Bold_signal)

Bs = (np.loadtxt('Bs_python.dat'))
As = (np.loadtxt('As_python.dat'))

Bold_filt_1 = np.zeros((n_T , N))
for col in range(0,N):			
	Bold_filt_1[: , col] = filtfilt(Bs, As, Bold_signal[:, col])	

np.savetxt('bold_filt_python_py.dat',Bold_filt_1,fmt='%.6f',delimiter='\t')	


b = np.loadtxt('Bs_matlab.dat');
a = np.loadtxt('As_matlab.dat');

Bold_filt_2 = np.zeros((n_T , N))
for col in range(0,N):			
	Bold_filt_2[: , col] = filtfilt(b, a, Bold_signal[:, col])	

np.savetxt('bold_filt_matlab_py.dat',Bold_filt_2,fmt='%.6f',delimiter='\t')	

pl.plot(Bold_filt_1[:,1], 'b')
pl.plot(Bold_filt_2[:,2], 'r')
pl.show()
