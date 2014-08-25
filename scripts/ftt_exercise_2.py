#!/usr/bin/python2.7
# -*- coding: utf-8 -*-



import numpy as np
import subprocess as sp
import sys
import math
import pylab as pl
from scipy.signal import  butter , filtfilt , correlate2d
import scipy.integrate as integ
from scipy.integrate import odeint
import time

# generating sine wave 
v1 = 0.05   # [Hz]
v2 = 0.10   # [Hz]

Fs = 10     # [Hz], Sampling frequency
T  = 1.0/Fs	# [Hz], Sample time
tmax = 100  # [s], max time of signal run

t = np.arange(0,  tmax , T )  # time array 

x = np.sin(2*math.pi*v1*t) + np.sin(2*math.pi*v2*t) 

#%noise = randn(size(t));
noise = 0
X     = x+noise

#np.savetxt('sinwave.dat',X,fmt='%.6f',delimiter='\t')
#np.savetxt('sinwave_time.dat',t,fmt='%.6f',delimiter='\t')

N     =  len(x)
N_pow =  int(pow(2, math.ceil(math.log(N)/math.log(2))))

Xfft  = np.fft.fft(X , n=N_pow) /float(N);
Xfft  = 2*abs(Xfft[0:N_pow /2 +1]) 

fre   = float(Fs)/2 * np.linspace(0,1, N_pow/2 + 1);

#np.savetxt('sinwave_fft.dat',Xfft, fmt='%.6f',delimiter='\t')
#np.savetxt('sinwave_fre.dat',fre, fmt='%.6f',delimiter='\t')


Wn    = 0.01   #Hz , cut-off freq 
[Bs,As] = butter(5, Wn, btype='low',analog=False, output='ba');

X_filt     = filtfilt(Bs,As,X)

#np.savetxt('sinwave_filt.dat',X_filt, fmt='%.6f',delimiter='\t')


X_filt_fft = np.fft.fft(X_filt, n=N_pow) /float(N)
X_filt_fft = 2*abs(X_filt_fft[0:N_pow /2 +1])  

#np.savetxt('sinwave_filt_fre.dat',X_filt_fft, fmt='%.6f',delimiter='\t')


f_c       = 0.25		
dtt       = 0.001	# (here 1 millisecond)
f_s       = 1/dtt	# Sampling frequency (Hz)
f_N       = f_s/2	# Nyquist frequency (Hz)

y     = np.loadtxt('bold_signal_matlab.dat')
y     = y[:,0]

t_y   = np.arange(0, len(y), 1)

#np.savetxt('bold_y.dat',y, fmt='%.6f',delimiter='\t')
#np.savetxt('bold_yt.dat',t_y, fmt='%.6f',delimiter='\t')

m     = len(y);
m_pow = int(pow(2, math.ceil(math.log(m)/math.log(2))))

yfft  = np.fft.fft(y , m_pow) /float(m)
yfft  = 2*abs(yfft[0:m_pow /2 +1])  
freq  = float(f_s)/2 * np.linspace(0,1, m_pow/2 + 1);

#np.savetxt('bold_y_fft.dat',yfft, fmt='%.6f',delimiter='\t')
#np.savetxt('bold_yfre.dat',freq, fmt='%.6f',delimiter='\t')



[Bs,As] = butter(5, f_c/f_N, btype='low',analog=False, output='ba')
Bs = np.loadtxt('Bs_matlab.dat')
As = np.loadtxt('As_matlab.dat')

y_filt  = filtfilt(Bs,As,y)
y_filt_fft  = np.fft.fft(y_filt , m_pow) /float(m)
y_filt_fft  = 2*abs(y_filt_fft[0:m_pow /2 +1])  


#np.savetxt('bold_y_filt.dat',y_filt, fmt='%.6f',delimiter='\t')
#np.savetxt('bold_y_filt_fft.dat',y_filt_fft, fmt='%.6f',delimiter='\t')


t_ma   = np.loadtxt('t_matlab.dat', unpack=True)
X_ma   = np.loadtxt('sine_matlab.dat', unpack=True)	
fre_ma = np.loadtxt('freq_matlab.dat', unpack=True)
Xfft_ma= np.loadtxt('sine_fft_matlab.dat', unpack=True)

X_filt_ma     = np.loadtxt('X_filt_matlab.dat', unpack=True)
X_filt_fft_ma = np.loadtxt('X_filt_fft_matlab.dat', unpack=True)

t_y_ma		  = np.loadtxt('t_y_matlab.dat', unpack=True)
y_ma		  = np.loadtxt('y_matlab.dat', unpack=True)
freq_ma		  = np.loadtxt('freq_matlab.dat', unpack=True)
yfft_ma		  = np.loadtxt('yfft_matlab.dat', unpack=True)

y_filt_ma	  = np.loadtxt('y_filt_matlab.dat', unpack=True)
y_filt_fft_ma = np.loadtxt('y_filt_fft_matlab.dat', unpack=True)

pl.figure(4)
pl.subplot(1,2,1)
pl.plot(t_y/1000, y_filt)
pl.plot(t_y_ma/float(1000), y_filt_ma, 'r')
pl.xlabel('time [s]')
pl.ylabel('filtered bold signal (t)')
#%axis([0, t(end), 4.9 , 5.4])
pl.subplot(1,2,2)
pl.plot(freq[0:50] , y_filt_fft[0:50])
pl.plot(freq_ma[0:50] , y_filt_fft_ma[0:50], 'r')
pl.ylabel('|filtered bold signal (f)|')
pl.xlabel('f [Hz]')
pl.show()


pl.figure(1);
pl.subplot(121)
pl.plot(t,X)
pl.plot(t_ma, X_ma, '.r')
lg = pl.legend(['python', 'matlab']) 
lg.draw_frame(False)
pl.title('sin('+str(v1)+'t)'+'+sin('+str(v2)+'t)')
pl.subplot
pl.ylabel('signal')
pl.xlabel('time [s]')
pl.subplot(1,2,2)
pl.plot(fre[0:100] ,Xfft[0:100] )
pl.plot(fre_ma[0:100] , Xfft_ma[0:100], '.r')
lg = pl.legend(['python', 'matlab']) 
lg.draw_frame(False)
pl.title('fourier transform')
pl.ylabel('|signal(f)|')
pl.xlabel('f [Hz]')
#pl.show()

pl.figure(2)
pl.subplot(1,2,1)
pl.plot(t, X_filt)
pl.plot(t_ma, X_filt_ma, '.r')
lg = pl.legend(['python', 'matlab'])
lg.draw_frame(False)
pl.title('Low-Pass filter, Wn=' +str(Wn)+ 'Hz\nsin('+str(v1)+'t)'+'+sin('+str(v2)+'t)')
pl.ylabel('signal filt')
pl.xlabel('time [s]')
pl.axis([0 , 100, -2 , 2])
pl.subplot(1,2,2)
pl.plot(fre[0:100], X_filt_fft[0:100])
pl.plot(fre_ma[0:100], X_filt_fft_ma[0:100], '.r')
lg = pl.legend(['python', 'matlab'])
lg.draw_frame(False)
pl.title('Low-Pass filtered \n fourier transform')
pl.ylabel('|signal filt (f)|')
pl.xlabel('f [Hz]')
#pl.show()

pl.figure(3);
pl.subplot(1,2,1)
pl.plot((t_y) /float(1000)  , y)
pl.plot((t_y_ma) /float(1000)  , y_ma, 'r')
lg = pl.legend(['python', 'matlab']) 
lg.draw_frame(False)
pl.title('imported bold signal')
pl.xlabel('time [s]')
pl.ylabel('bold signal')
#pl.axis([0, t[-1], 4 , 5.5])
pl.subplot(1,2,2)
pl.plot(freq[0:50], yfft[0:50])
pl.plot(freq_ma[0:50], yfft_ma[0:50], '.r')
lg = pl.legend(['python', 'matlab']) 
lg.draw_frame(False)
pl.title('fourier transform')
pl.ylabel('|bold signal (f)|')
pl.xlabel('f [Hz]')
#pl.show()

