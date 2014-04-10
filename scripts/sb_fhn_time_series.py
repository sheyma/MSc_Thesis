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

gfilename = sys.argv[1]
dfilename = sys.argv[2]

eqns = {r'x{i}': '(y{i} + gamma * x{i} - pow(x{i},3.0)/3.0) * TAU',

        r'y{i}': '- (x{i} - alpha + b * y{i}) / TAU'}

params = { # Fitzhugh-Nagumo parameters...

        'gamma': 0.9, 
        'alpha': 0.89,  
        'b': 0.1,
        'TAU': 4, 
	'sigma': 4,  
	'D' : 0,  
	'v' : 70.0, } # velocity in 0.1 m/s 

noise = {'x': 'D * gwn()', 'y': 'D * gwn()'} 


G = np.loadtxt(gfilename) # weight matrix
 
C = params['sigma'] 

H = [ [C, 0],

      [0, 0] ]
try:

	D_matrix = np.loadtxt(dfilename)

except:

	print 'File not found:', dfilename

T  = D_matrix/params['v']

print "max delay in time delay matrix", T.max()
max_tau = math.ceil(T.max())


coupling = '-{G:.6f}*{H}*({var}(t-{tau})-{self})'

neuronetz = simnet(eqns, G, H, T, params, coupling, noise)

#random.seed()

 
thist = np.linspace(0, max_tau, 10000)

xhist = np.zeros(len(thist)) 

yhist = np.zeros(len(thist)) 

dic = {'t' : thist}

for i in range(0,len(G)):

    dic['x'+str(i)] = xhist+0  # 1.67
    dic['y'+str(i)] = yhist+0


neuronetz.ddeN.hist_from_arrays(dic)


""" Start simulation with t = [0,tmax] """

tmax = 70000
neuronetz.run(tmax)


print "FitzHugh-Nagumo"
print "variables : " , neuronetz.ddeN.vars
print "equations : ", neuronetz.ddeN.eqns
print "parameters : ", neuronetz.params
print "noise : ", neuronetz.noise
print "delays : ", neuronetz.ddeN.delays

sol_samp1 = neuronetz.ddeN.sample(480,500,dt=0.1)
sol_samp2 = neuronetz.ddeN.sample(480,500,dt=1)


sol_adap = neuronetz.adaptive_sol
print (sol_adap['t'][0:10000])


t = sol_samp1['t'][0:]
x = {}
y = {}

for i in range(0,len(G[0])):
  x[i] = sol_samp1['x'+str(i)][0:]
  y[i] = sol_samp1['y'+str(i)][0:]
  

x_limit = 2.5
x_range = np.linspace(-2.5,2.5,500)

#nullclines
def nullcl_01(X):
  return -(pow(X,3)/3 - params['gamma']*X)
  #make x minus
  
def nullcl_02(X):
  return (params['alpha'] - X ) / params['b']

#calculate the intersection of nullclines
def intersection(X):
 return nullcl_01(X) - nullcl_02(X)

# please estimate where to search for the root as X0 parameter
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


fig = pl.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
#fig.suptitle('[PAN12] Global Dynamics :  a = '+str(params['a'])+
	      #r'  $\varepsilon$'+' = '+str(params['eps']) +
	      #'  C = '+ str(params['C']) +'  '  + r'$\tau^C$= '+
	      #str(params['tau'])+ '  K = '+ str(params['K']) +'  '  + 
	      #r'$\tau^K$= '+ str(params['tau']),fontsize=14, fontweight='bold')
fig.suptitle('FHN time series :  '+r'$\alpha$ = ' +str(params['alpha'])+
	      r'  $\gamma$ = '+str(params['gamma']) + ' $ b$ = '+ 
	      str(params['b']) + r'  $\tau$ = '+str(params['TAU']) +
	      '  C = ' + str(params['sigma']) +'  '  + r'$\Delta\tau_{ij}$=$d_{ij}/v$'
	      #'  K = '+ str(params['K']) +'  '  + r'$\tau^K$= '+
	      #str(params['tau'])#
	      , fontsize=14, fontweight='bold')

pl.subplot(2,1,1)
pl.plot(sol_samp1['t'],sol_samp1['x0'],'r',label='x(t)')
pl.plot(sol_samp1['t'],sol_samp1['y0'], 'b',label='y(t)')
pl.plot(sol_adap['t'][10573:10758],sol_adap['x0'][10573:10758],'.r',linewidth=0.05)
pl.plot(sol_adap['t'][10573:10758],sol_adap['y0'][10573:10758],'.b',linewidth=0.05)
pl.xlabel('$t$')
pl.ylabel('$x_1,y_1$')
lg = legend(loc=2)
lg.draw_frame(False)

pl.subplot(2,1,2)
pl.plot(sol_samp2['t'],sol_samp2['x0'],'r')
pl.plot(sol_samp2['t'],sol_samp2['y0'], 'b')
pl.plot(sol_adap['t'][10573:10758],sol_adap['x0'][10573:10758],'.r',linewidth=0.05)
pl.plot(sol_adap['t'][10573:10758],sol_adap['y0'][10573:10758],'.b',linewidth=0.05)
pl.xlabel('$t$')
pl.ylabel('$x_1,y_1$')
	      
pl.show()
	      
#pl.subplot(221)
#pl.plot(sol_samp1['t'],sol_samp1['x0'],'r')
#pl.plot(sol_samp1['t'],sol_samp1['y0'], 'b')
##pl.plot(sol_adap['t'][10573:10758],sol_adap['x0'][10573:10758],'.r',linewidth=0.05)
##pl.plot(sol_adap['t'][10573:10758],sol_adap['y0'][10573:10758],'.b',linewidth=0.05)
#pl.xlabel('$t$')
#pl.ylabel('$x_1,y_1$')

#pl.subplot(222)
#pl.plot(sol_samp1['x0'],sol_samp1['y0'], 'r')
#pl.plot(sol_samp1['x0'][0],sol_samp1['y0'][0], 'or')
##pl.plot(sol_adap['x'],sol_calc['y'],'or')
#pl.plot(x_range,nullcl_01(x_range),'b', label='$x_{nullcline}$')
#pl.plot(x_range,nullcl_02(x_range),'k',label='$y_{nullcline}$')
#pl.plot(X_int[0],Y_int[0], 'ok', linewidth=5)
##pl.axis([-2.3, 2.3, -1.5, 1.5])
#lg = legend(loc=2)
#lg.draw_frame(False)
##pl.plot(x_int,y_int,'ok')

#pl.axis([-2.3, 2.3, -1, 1])
#pl.xlabel('$x_1$')
#pl.ylabel('$y_1$')

#pl.subplot(223)
#pl.plot(sol_samp1['t'],sol_samp1['x1'],'r')
#pl.plot(sol_samp1['t'],sol_samp1['y1'], 'b')

##pl.plot(sol_calc['t'],sol_calc['x'],'or')
##pl.plot(sol_calc['t'],sol_calc['y'],'ob')
#pl.xlabel('$t$')
#pl.ylabel('$x_2,y_2$')

#pl.subplot(224)
#pl.plot(sol_samp1['x1'],sol_samp1['y1'], 'r')
#pl.plot(sol_samp1['x1'][0],sol_samp1['y1'][0], 'or')
#pl.plot(x_range,nullcl_01(x_range),'b', label='$x_{nullcline}$')
#pl.plot(x_range,nullcl_02(x_range),'k',label='$y_{nullcline}$')
#pl.plot(X_int[0],Y_int[0], 'ok', linewidth=5)
#pl.axis([-2.3, 2.3, -1.5, 1.5])
#pl.xlabel('$x_2$')
#pl.ylabel('$y_2$')
#lg = legend(loc=2)
#lg.draw_frame(False)

##pl.savefig("FHN_time_series_C_big.eps",format="eps")
##pl.savefig("FHN_time_series_C_small.eps",format="eps")

pl.show()

