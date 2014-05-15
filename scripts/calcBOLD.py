#!/usr/bin/python2.7 

# -*- coding: utf-8 -*-

import numpy as np
import sys

	  
#def BOLD(T,r):
	
	#taus   =  0.65;    
	#tauf   = 0.41;    
	#tauo   =  0.98;    
	#alpha  =  0.32;


	#params = {
	#'itaus' : float(1/taus),
	#'itauf' : float(1/tauf),
	#'itauo' : float(1/tauo),
	#'ialpha' : float(1/alpha),
	#'Eo' : 0.34,
	#'dt' : 0.001
	#}

	#vo     = 0.02;
	#k1     = 7 * params['Eo'] 
	#k2     = 2; 
	#k3     = 2 * params['Eo']-0.2

	#ch_int = 0

	#t0 = np.transpose(np.arange(0,(T+params['dt']),params['dt']))

	#n_t = len(t0)

	#t_min = 20
	#n_min = round(t_min / params['dt'])

	#r_max = max(r)

	#x0 = np.array([0 , 1, 1, 1])

	#if ch_int==0:

		#t = t0
		#x = np.zeros((n_t,4))

		#x[0,:] = x0
		
		#for n in range(0,n_t-1):
			#x[n+1,0] = x[n,0]+ params['dt']* ( r[0,n] - params['itaus'] * x[n,0] - params['itauf'] * (x[n,1])-1)
			#x[n+1,1] = x[n,1]+ params['dt']*x[n,1]
			#x[n+1,2] = x[n,2]+ params['dt']*params['itauo']*(x[n,1] - pow(x[n,2],params['ialpha']))
			#x[n+1,3] = x[n,3] + params['dt']*params['itauo']*(x[n,2]*(1-pow((1-params['Eo']),(1/x[n,1])))/float(params['Eo']) - (pow(x[n,2],params['ialpha']))*x[n,3]/ float(x[n,2]))
 
		#print x
		
	

#r = np.zeros((1,1001))  
#r[0,5] = 2.0

#BOLD(1,r)


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
	print (np.shape(simout)[1] -1 ) /2
	#print dt_Tvec
	
		
input_name = sys.argv[1]	
calcBOLD(input_name)
