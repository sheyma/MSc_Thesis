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
from scipy.optimize import fsolve
from pylab import *

# FitzHugh Nagumo Model Equations
eqns = {r'x{i}': '(y{i} + gamma * x{i} - pow(x{i},3.0)/3.0) * TAU',
        r'y{i}': '- (x{i} - alpha + b * y{i}) / TAU'}

# Fitzhugh-Nagumo parameters
params = {  'gamma': 1.0, 
			'alpha': 0.85,  
			'b'    : 0.2,
            'TAU'  : 1.25, 
	        'sigma': 0.5,  
			'D'    : 0.05,  			# noise strength
			'v'    : 10.0, } 			# velocity in 0.1 m/s 

# gaussian white noise distribution
noise = {'x': 'D * gwn()', 'y': 'D * gwn()'} 

# coupling term 
C = params['sigma'] 

H = [ [C, 0],

      [0, 0] ]
      
gfilename = sys.argv[1]
dfilename = sys.argv[2]

try:
	G        = np.loadtxt(gfilename)
	D_matrix = np.loadtxt(dfilename)

except:

	print 'File not found:', dfilename

# time delays among nodes 
T  = D_matrix/params['v']

print "max delay in time delay matrix", T.max()
max_tau = math.ceil(T.max())


coupling = '-{G:.6f}*{H}*{var}(t-{tau})'
#coupling = '-{G:.1f}*{H}*{var}(t-{tau})'
neuronetz = simnet(eqns, G, H, T, params, coupling, noise)
 
thist = np.linspace(0, max_tau, 10000)
xhist = np.zeros(len(thist)) 
yhist = np.zeros(len(thist)) 

dic = {'t' : thist}

for i in range(0,len(G)):

    dic['x'+str(i)] = xhist+0  
    dic['y'+str(i)] = yhist+0

neuronetz.ddeN.hist_from_arrays(dic)

""" Start simulation with t = [0,tmax] """

tmax = 100
neuronetz.run(tmax)

print "FitzHugh-Nagumo"
print "variables : " , neuronetz.ddeN.vars
print "equations : ", neuronetz.ddeN.eqns
print "parameters : ", neuronetz.params
print "noise : ", neuronetz.noise
print "delays : ", neuronetz.ddeN.delays

sol_samp1 = neuronetz.ddeN.sample(0,tmax,dt=0.1)

sol_adap  = neuronetz.sol
#print (sol_adap['t'][0:10000])

t = sol_samp1['t'][0:]
x = {}
y = {}

for i in range(0,len(G[0])):
  x[i] = sol_samp1['x'+str(i)][0:]
  y[i] = sol_samp1['y'+str(i)][0:]
  
x_limit = -5
x_range = np.linspace(-x_limit, x_limit, 500)

#nullclines
def nullcl_01(X):
  return -(pow(X,3)/3 - params['gamma']*X)
  #make x minus
  
def nullcl_02(X):
  return (params['alpha'] - X ) / params['b']

#calculate the intersection of nullclines
def intersection(X):
 return nullcl_01(X) - nullcl_02(X)

# Warning : Estimate where to search for the root as X0 parameter !!
X0 = 0
X_int = fsolve(intersection,X0)
Y_int = nullcl_01(X_int)  
  
print "intersection of nullclines", X_int, Y_int
  
#f = open('deneme.dat','w')	
#for i, t0 in enumerate(t):	
	#f.write("%s\t" % (t0))
	#f.write("%.5f\t" % (sol_samp1['x'][i]) )
	#f.write("%.5f\t" % (sol_samp1['y'][i]))
	#f.write("\n")
#f.close()


fig = pl.figure(num=None, figsize=(15, 8), dpi=100, facecolor='w', edgecolor='k')
#fig.suptitle('[PAN12] Global Dynamics :  a = '+str(params['a'])+
	      #r'  $\varepsilon$'+' = '+str(params['eps']) +
	      #'  C = '+ str(params['C']) +'  '  + r'$\tau^C$= '+
	      #str(params['tau'])+ '  K = '+ str(params['K']) +'  '  + 
	      #r'$\tau^K$= '+ str(params['tau']),fontsize=14, fontweight='bold')
#fig.suptitle('FHN - time series :  '+r'$\alpha$ = ' 
		#+ str(params['alpha']) +r'  $\gamma$ = '+str(params['gamma']) 
		#+ ' $ b$ = '+  str(params['b'])+' ' + 
		#r'$\tau$ = '+str(params['TAU'])+'  C = ' + str(params['sigma'])
		#+ '  D = 0.05 '  +  '  v = 1 [m/s] '
		 #+'  '    + r'$\Delta\tau_{ij}$=$d_{ij}/v$'
	      ##'  K = '+ str(params['K']) +'  '  + r'$\tau^K$= '+
	      ##str(params['tau'])#
	      #, fontsize=25)

pl.figure(1)
#pl.subplot(2,1,1)
pl.plot(sol_samp1['t'],sol_samp1['x2'], 'r',label='$x_1(t)$')
pl.plot(sol_samp1['t'],sol_samp1['y2'], 'b',label='$y_1(t)$')
#pl.plot(sol_adap['t'],sol_adap['x0'],'.r',linewidth=0.05)
#pl.plot(sol_adap['t'][10573:10758],sol_adap['y0'][10573:10758],'.b',linewidth=0.05)
pl.xlabel('t', fontsize = 25)
pl.ylabel('$x_1(t) , y_1(t)$', fontsize = 30)
pl.tick_params(labelsize=20)
lg = legend(loc=2, prop={'size':25})
lg.draw_frame(False)

#pl.subplot(212)
#pl.plot(sol_samp1['x1'],sol_samp1['y1'], 'r')
#pl.plot(sol_samp1['x1'][0],sol_samp1['y1'][0], 'or')
##pl.plot(sol_adap['x'],sol_calc['y'],'or')
#pl.plot(x_range,nullcl_01(x_range),'b', label='$x_{nullcline}$')
#pl.plot(x_range,nullcl_02(x_range),'k',label='$y_{nullcline}$')
#pl.plot(X_int[0],Y_int[0], 'ok', linewidth=5)
##pl.axis([-2.3, 2.3, -1.5, 1.5])
#lg = legend(loc=2)
#lg.draw_frame(False)
#pl.axis([-5, 5, -5 ,5])
##pl.axis([-2.3, 2.3, -1, 1])
#pl.xlabel('$x_1$')
#pl.ylabel('$y_1$')

#pl.savefig("FHN_time_series_C_small.eps",format="eps")
pl.show()


