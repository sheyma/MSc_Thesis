#!/usr/bin/python2.7 

# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import sys
import pylab as pl
from pydelay import dde23
from scipy.optimize import fsolve

eqns = {r'x1' : '(y1 + gamma*x1 - pow(x1,3)/3.0) * TAU ',
	r'y1' : '-(x1 - alpha + b*y1 ) / TAU',
	r'x2' : '(y2 + gamma*x2 - pow(x2,3)/3.0) * TAU ',
	r'y2' : '-(x2 - alpha + b*y2 ) / TAU'}
	
params = {'gamma' : 0.85,
	  'alpha' : 0.5,
	  'TAU' : 10,
	  'b' : -0.01,
	  }

dde = dde23(eqns=eqns,params=params)

tfinal = 200

dde.set_sim_params(tfinal)

dde.hist_from_funcs( {'x1': lambda t : -1.2 , 'y1': lambda t: -0.5,
		       'x2': lambda t : -1.8 , 'y2': lambda t: -0.1})

dde.run()

dt = float(sys.argv[1])
sol_sample = dde.sample(0,tfinal,dt)

t = sol_sample['t']
x1 = sol_sample['x1']
x2 = sol_sample['x2']
y1 = sol_sample['y1']
y2 = sol_sample['y2']

print "FitzHugh Nagumo Local Dynamics"
print "equations: ", dde.eqns
print "parameters: ", dde.params
print "simulation time : ", tfinal
print "Sampling time interval : " , dt

x_limit = 2.5
x_range = np.linspace(-2.5,2.5,500)

# nullclines
def nullcl_01(x):
  return -(pow(x,3)/3 - params['gamma']*x)

def nullcl_02(x):
  return (params['alpha'] - x ) / params['b']

# calculate the intersection of nullclines
def intersection(x):
  return nullcl_01(x) - nullcl_02(x)

x0 = -2
x_int = fsolve(intersection,x0)
y_int = nullcl_01(x_int)

print "intersection of nullclines (x_0, y_0): ", x_int, y_int




pl.subplot(221)
pl.plot(t, x1, 'r')
pl.plot(t, y1, 'g')
pl.xlabel('$t$')
pl.ylabel('$x_1, y_1$')

pl.subplot(222)
pl.plot(x1, y1, 'r')
pl.plot(x_range,nullcl_01(x_range),'b')
pl.plot(x_range,nullcl_02(x_range),'k')
pl.axis([-2.3, 2.3, -1, 1])
pl.xlabel('$x_1$')
pl.ylabel('$y_1$')

pl.subplot(223)
pl.plot(t, x2, 'r')
pl.plot(t, y2, 'g')
pl.xlabel('$t$')
pl.ylabel('$x_2, y_2$')

pl.subplot(224)
pl.plot(x2, y2, 'g')
pl.xlabel('$x_2$')
pl.ylabel('$y_2$')

pl.show()