#!/usr/bin/python2.7 

# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import sys
import pylab as pl
from pydelay import dde23
from scipy.optimize import fsolve
from pylab import *

eqns = {r'x' : '(y + gamma*x - pow(x,3)/3.0) * TAU ',
	r'y' : '-(x - alpha + b*y +I ) / TAU' }
	
params = {'gamma' : 0.9,
	  'alpha' : 1.6,
	  'TAU' : 5,
	  'b' : -0.01,
	  'I' : 2.5
	  }

x_limit = 2.5
x_range = np.linspace(-2.5,2.5,500)

# nullclines
def nullcl_01(X):
  return -(pow(X,3)/3 - params['gamma']*X)
  # make x minus
  
def nullcl_02(X):
  return (params['alpha'] - X -params['I']) / params['b']

# calculate the intersection of nullclines
def intersection(X):
  return nullcl_01(X) - nullcl_02(X)

X0 = -2
X_int = fsolve(intersection,X0)
Y_int = nullcl_01(X_int)

dde = dde23(eqns=eqns,params=params)

tfinal = 50

dde.set_sim_params(tfinal)

dde.hist_from_funcs( {'x': lambda t : X_int , 'y': lambda t: Y_int })

dde.run()

dt = float(sys.argv[1])
sol_sample = dde.sample(0,tfinal,dt)

t = sol_sample['t']
x = sol_sample['x']
y = sol_sample['y']


print "FitzHugh Nagumo Local Dynamics"
print "equations: ", dde.eqns
print "parameters: ", dde.params
print "simulation time : ", tfinal
print "Sampling time interval : " , dt


print "intersection of nullclines (x_0, y_0): ", X_int, Y_int


pl.subplot(121, xlabel='t', ylabel='x , y')
pl.plot(t, x, 'r', label='$x(t)$')
pl.plot(t, y, 'b', label='$y(t)$')
#pl.xlabel('$t$')
#pl.ylabel('$x, y$')
lg = legend()
lg.draw_frame(False)

pl.subplot(122, xlabel='x', ylabel='y')
pl.plot(x, y, 'r', label='$x,y$')
pl.plot(x_range,nullcl_01(x_range),'b', label='$x_{nullcline}$')
pl.plot(x_range,nullcl_02(x_range),'k',label='$y_{nullcline}$')
pl.plot(X_int,Y_int, 'or', linewidth=3)
pl.axis([-2.3, 2.3, -1, 1])
lg = legend()
lg.draw_frame(False)

pl.show()
