#!/usr/bin/python2.7 

# -*- coding: utf-8 -*-

import numpy as np

taus   =  0.65;    
tauf   = 0.41;    
tauo   =  0.98;    
alpha  =  0.32;


params = {
	  'itaus' : float(1/taus),
	  'itauf' : float(1/tauf),
	  'itauo' : float(1/tauo),
	  'ialpha' : float(1/alpha),
	  'Eo' : 0.34,
	  'dt' : 0.001
	  }

	  
def BOLD(T,r):
  ch_int = 0
  
  t0 = np.transpose(np.arange(0,(T+params['dt']),params['dt']))

  n_t = len(t0)
  
  t_min = 20
  n_min = round(t_min / params['dt'])
  
  r_max = max(r)

  vo     = 0.02;
  k1     = 7 * params['Eo'] 
  k2     = 2; 
  k3     = 2 * params['Eo']-0.2
  
  x0 = np.array([0 , 1, 1, 1])
  
  if ch_int==0:
    
    t = t0
    x = np.zeros((n_t,4))
    
    x[0,:] = x0
    print x
  
  
  
BOLD(1,np.array([1, 2, 3, 5]))