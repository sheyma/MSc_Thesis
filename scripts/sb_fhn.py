#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl
import sys
from pydelay import dde23
from scipy.optimize import fsolve

eqns = { 
		 'x': '(y + gamma * x - pow(x,3.0)/3.0) * TAU',
         'y': '- (x - alpha + b * y - I) / TAU'
			}

params = { 
		 'gamma': 0.9, 
		 'alpha': 1.3,  
		 'b': -0.1,
		 'TAU' : 10,	  
		 'I' : -2.2,
		 'D' : 0} 								  # change noise strength

noise = {'x': 'D * gwn()', 'y': 'D * gwn()'}

# defining nullclines
x_limit = 2.5
x_range = np.linspace(-2.5,2.5,500)

nullcl_01 = (pow(x_range,3)/3 - params['gamma']*x_range)
nullcl_02 = (params['alpha'] - x_range + params['I']) / params['b']

# calculate the intersection of nullclines
def intersection(x):
  return ((pow(x,3)/3 - params['gamma']*x) - (params['alpha'] - x + params['I']) / params['b'] )

x0 = -1.5
x_int = fsolve(intersection,x0)
y_int = ( (pow(x_int ,3)/3) - params['gamma']*x_int)


dde = dde23(eqns=eqns, params=params, noise=noise) # initialize the solver

tfinal = 50

dde.set_sim_params(tfinal)

dde.hist_from_funcs({'x': lambda t: -0.99,
		     'y': lambda t: -0.72})	  # initial conditions
dde.run()								  # run the simulation

sol_samp1 = dde.sample(0,50, 0.1)  		# sampled numerical solution	
	
sol_calc = dde.sol				   		# actual numerical solution

t = sol_samp1['t']

dde.hist






print "FitzHugh-Nagumo"
print "variables : " , dde.vars
print len(t)
print "equations : ", dde.eqns
print "parameters : ", dde.params
print "noise : ", dde.noise
print "delays : ", dde.delays

f = open('deneme.dat','w')	

for i, t0 in enumerate(t):	
	f.write("%s\t" % (t0))
	f.write("%.5f\t" % (sol_samp1['x'][i]) )
	f.write("%.5f\t" % (sol_samp1['y'][i]))
	f.write("\n")
f.close()




  
pl.subplot(121)
pl.plot(sol_samp1['t'],sol_samp1['x'],'r')
pl.plot(sol_samp1['t'],sol_samp1['y'], 'b')
pl.plot(sol_calc['t'],sol_calc['x'],'or')
pl.plot(sol_calc['t'],sol_calc['y'],'ob')
pl.xlabel('$t$')
pl.ylabel('$x_1,y_1$')

pl.subplot(122)
pl.plot(sol_samp1['x'],sol_samp1['y'], 'r')
pl.plot(sol_samp1['x'][0],sol_samp1['y'][0], 'ok')
pl.plot(x_range, nullcl_01, 'b')
pl.plot(x_range, nullcl_02, 'k')
pl.plot(x_int,y_int,'or')
pl.axis([-x_limit, x_limit, -1, 1])
pl.xlabel('$x$')
pl.ylabel('$y$')

pl.show()
