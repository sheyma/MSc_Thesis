#!/usr/bin/python2.7 
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import sys
from netpy import simnet
import random
import numpy as np
import math
import os
import pylab as pl

# adjacency matrix
gfilename = sys.argv[1]
# distance matrix
dfilename = sys.argv[2]

# FitzHugh-Nagumo Local Dynamics
"""Attention: alphabethical order of equation advised!!! """
eqns = {r'x{i}': '(y{i} + gamma * x{i} - pow(x{i},3.0)/3.0) * TAU',
        r'y{i}': '- (x{i} - alpha + b * y{i}) / TAU'}

params = { # Fitzhugh-Nagumo parameters...
        'gamma': 1.0, 
        'alpha': 0.85,  
        'b': 0.2,
        'TAU': 1.25, 
		# global coupling constant
		'sigma': float(sys.argv[3]),  
		# noise strength
		'D' : float(0.05),  
		# velocity [0.1 m/s]
	    'v' : float(sys.argv[4]), 
}

# Gaussian White Noise
noise = {'x': 'D * gwn()', 'y': 'D * gwn()'}  

""" Topology """
G = np.loadtxt(gfilename) 

C = params['sigma'] 

H = [ [C, 0],

      [0, 0] ]

print 'H', H

"""Delay-Matrix, T = distance / velocity"""
# check if dfilename exists
try:
	D_matrix = np.loadtxt(dfilename)
except:
	print 'File not found:', dfilename

T  = D_matrix/params['v']

print 'maximum delay: ', T.max()
print 'ceiling maximum delay: ', math.ceil(T.max())
max_tau = math.ceil(T.max())

""" coupling term """
#diffusive coupling: \dot var_i = ... (var_j - var_i)
#coupling = '+{G:.12f}*{H}*({var}-{selfpy})'
#direct coupling: \dot var_i = ... var_j
coupling = '-{G:.1f}*{H}*{var}(t-{tau})'

"""Let's go """
neuronetz = simnet(eqns, G, H, T, params, coupling, noise)

random.seed()

# generate history function (initial conditions) 
thist = np.linspace(0, max_tau, 10000)
xhist = np.zeros(len(thist))
yhist = np.zeros(len(thist)) + 0.5

# generate a dictionary and fill it with initial conditions
dic = {'t' : thist}

for i in range(len(G)):
  # all elements identical
  dic['x'+str(i)] = xhist  
  dic['y'+str(i)] = yhist	
  # constant shift added
  #dic['x'+str(i)] = xhist + i / len(G)
  # random values added
  #dic['x'+str(i)] = xhist + 1.*random.random()

neuronetz.ddeN.hist_from_arrays(dic)

""" Start simulation with t = [0,tmax] """
# total simulation time [10 ms] 
tmax = 45000  
neuronetz.run(tmax)

# alternative way to generate history function/initial conditions
#initial_conditions = {'x0': -0.95, 'y0': -0.95+ pow(0.95,3.0)/3.0,
                      #'x1': -0.95, 'y1': -0.95+ pow(0.95,3.0)/3.0}
#neuronetz.run(initial_conditions, tmax=600)

series = neuronetz.sol

#sample solution for output starting at t=-max_tau
solution = neuronetz.ddeN.sample(0,  dt=0.1)

print "starting print-out of data..."
# time array
t = solution['t'][0:]
x = {}
y = {}
for i in range(0,len(G[0])):
	x[i] =  solution['x'+str(i)][0:]	
	# skip y for now to minimize output
	# y[i] =  solution['y'+str(i)][0:]

f = open(gfilename[:-4]+'_sigma='+str(params['sigma'])+
		 '_D='+str(params['D'])+'_v='+str(params['v'])+
		 '_tmax='+str(tmax)+'.dat', 'w')

for i, t0 in enumerate(t):
	f.write('%s' % (t0))
	for j in range(0, len(x)):
		f.write('\t%.4f' % (float(x[j][i])))
	f.write('\n')
f.close()

print "done!"

#pl.subplot(221)
#pl.plot(solution['t'],solution['x0'],'r')
#pl.plot(solution['t'],solution['y0'], 'b')
##pl.plot(sol_calc['t'],sol_calc['x'],'or')
##pl.plot(sol_calc['t'],sol_calc['y'],'ob')
#pl.xlabel('$t$')
#pl.ylabel('$x_1,y_1$')

#pl.subplot(222)
#pl.plot(solution['x0'],solution['y0'], 'r')
#pl.plot(solution['x0'][0],solution['y0'][0], 'or')
##pl.plot(sol_calc['x'],sol_calc['y'],'or')
##pl.plot(x_range, nullcl_01(x_range), 'b')
##pl.plot(x_range, nullcl_02(x_range), 'k')
##pl.plot(x_int,y_int,'ok')

##pl.axis([-2.3, 2.3, -1, 1])
#pl.xlabel('$x_1$')
#pl.ylabel('$y_1$')

#pl.subplot(223)
#pl.plot(solution['t'],solution['x1'],'r')
#pl.plot(solution['t'],solution['y1'], 'b')
##pl.plot(sol_calc['t'],sol_calc['x'],'or')
##pl.plot(sol_calc['t'],sol_calc['y'],'ob')
#pl.xlabel('$t$')
#pl.ylabel('$x_2,y_2$')

#pl.subplot(224)
#pl.plot(solution['x1'],solution['y1'], 'r')
#pl.plot(solution['x1'][0],solution['y1'][0], 'or')
#pl.xlabel('$x_2$')
#pl.ylabel('$y_2$')

#pl.show()

