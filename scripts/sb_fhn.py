#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl
import sys
from pydelay import dde23
from netpy import simnet
from scipy.optimize import fsolve

matrix_01 = sys.argv[1]		# test matrix
matrix_02 = sys.argv[2]		# distances	

G = np.loadtxt(matrix_01)
d = np.loadtxt(matrix_02)

#FitzHugh-Naugmo 
eqns = { r'x{i}': '(y{i} + gamma * x{i} - pow(x{i},3.0)/3.0) * TAU  ',   
         r'y{i}': '- (x{i} - alpha + b * y{i}) / TAU'
			}

params = { 
		 'gamma': 0.9, 
		 'alpha': 1.3,  
		 'b': -0.01,
		 'TAU' : 10,	  
		 'D' : 0,
		 'v' : 70.0,
		 'sigma' : 0.5,    # C, coupling strength
		 'K_constant' : 0.3}  # K, self coupling strength		  
		 
noise = {'x': 'D * gwn()', 'y': 'D * gwn()'}

C = params['sigma']
K = params['K_constant']

H = [ [C,0] , [0,0]]

T = d / params['v']



coupling = '+{G:.12f}*{H}({self}(t-{tau})-{self})'

neuronet = simnet(eqns, G, H, T, params, coupling, noise)

#x_limit = 2.5
#x_range = np.linspace(-2.5,2.5,500)

## nullclines
#def nullcl_01(x):
	#return -(pow(x,3)/3 - params['gamma']*x)

#def nullcl_02(x):
	#return (params['alpha'] - x + params['I']) / params['b']

## calculate the intersection of nullclines
#def intersection(x):
  #return nullcl_01(x) - nullcl_02(x)

#x0 = -1.5
#x_int = fsolve(intersection,x0)
#y_int = nullcl_01(x_int)

#dde = dde23(eqns=eqns, params=params, noise=noise) # initialize the solver

#tfinal = 100

#dde.set_sim_params(tfinal)

#dde.hist_from_funcs({'x': lambda t: x_int,
		             #'y': lambda t: y_int})	  # initial conditions
#dde.run()						  # run the simulation

#sol_samp1 = dde.sample(0, tfinal, 0.1)  		# sampled numerical solution	
	
#sol_calc = dde.sol				   		# actual numerical solution

#t = sol_samp1['t']

#dde.hist

print "FitzHugh-Nagumo"
#print "variables : " , dde.vars
#print len(t)
#print "equations : ", dde.eqns
#print "parameters : ", dde.params
#print "noise : ", dde.noise
#print "delays : ", dde.delays
#print "fixed point [x,y] : ", x_int, y_int

#f = open('deneme.dat','w')	

#for i, t0 in enumerate(t):	
	#f.write("%s\t" % (t0))
	#f.write("%.5f\t" % (sol_samp1['x'][i]) )
	#f.write("%.5f\t" % (sol_samp1['y'][i]))
	#f.write("\n")
#f.close()

#print len(sol_samp1['t'][0:])
#pl.subplot(121)
#pl.plot(sol_samp1['t'],sol_samp1['x'],'r')
#pl.plot(sol_samp1['t'],sol_samp1['y'], 'b')
#pl.plot(sol_calc['t'],sol_calc['x'],'or')
#pl.plot(sol_calc['t'],sol_calc['y'],'ob')
#pl.xlabel('$t$')
#pl.ylabel('$x_1,y_1$')

#pl.subplot(122)
#pl.plot(sol_samp1['x'],sol_samp1['y'], 'r')
#pl.plot(sol_samp1['x'][0],sol_samp1['y'][0], 'ok')
#pl.plot(sol_calc['x'],sol_calc['y'],'or')
#pl.plot(x_range, nullcl_01(x_range), 'b')
#pl.plot(x_range, nullcl_02(x_range), 'k')
#pl.plot(x_int,y_int,'ok')



#pl.axis([-2.3, 2.3, -1, 1])
#pl.xlabel('$x$')
#pl.ylabel('$y$')


#pl.show()




