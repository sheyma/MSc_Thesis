#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# solving delayed differential equations of Mackey_Glass
# author = V.Flunkert and E.Schöll

# import pydelay and numpy and pylab
import numpy as np
import pylab as pl
from pydelay import dde23

# define equations in a dictionary
eqns = {
	'x' : '0.25* x(t-tau) / (1.0 + pow(x(t-tau),p)) -0.1*x'
	}

# define parameters in a dictionary
params = {
	'tau' : 15,
	'p' : 10,
	}

# initialize the solver
dde = dde23(eqns=eqns, params=params)

# set the simulation parameters
# (solve from t=0 to t=1000 and limit the maximum step size to 1.0)
dde.set_sim_params(tfinal=1000,dtmax=1.0)

# set the history of the constant function 0.5 (using a python lambda function)

histfun = {
	'x' : lambda t : 0.5
	}
dde.hist_from_funcs(histfun, 51)

# run the simulator
dde.run()

# make a plot of x(t) vs x(t-tau):
# sample the solution twice with a step size of dt=0.1:

# once in interval [515, 1000]
sol1 = dde.sample(515, 1000, 0.1)
x1 = sol1['x']

# and once between [500, 1000-15]
sol2 = dde.sample(515-15, 1000-15, 0.1)
x2 = sol2['x']

print dde.hist() 

pl.plot(x1,x2)
pl.xlabel('$x(t)$')
pl.ylabel('$x(t-15)$')
pl.show()

