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


gfilename = sys.argv[1]
dfilename = sys.argv[2]

eqns = {r'x{i}': '(y{i} + gamma * x{i} - pow(x{i},3.0)/3.0) * TAU',

        r'y{i}': '- (x{i} - alpha + b * y{i}) / TAU'}

params = { # Fitzhugh-Nagumo parameters...

        'gamma': 0.9, 
        'alpha': 0.8,  
        'b': -0.1,
        'TAU': 3, 
	'sigma': 0.5,  
	'D' : 0,  
	'v' : 70.0, } # velocity in 0.1 m/s 

noise = {'x': 'D * gwn()', 'y': 'D * gwn()'} 


G = np.loadtxt(gfilename) # weight matrix
D_matrix = np.loadtxt(dfilename) 
C = params['sigma'] 

H = [ [C, 0],

      [0, 0] ]


T  = D_matrix/params['v']
print "time delay matrix", T
max_tau = math.ceil(T.max())


coupling = '-{G:.6f}*{H}*({self}(t-{tau})-{var})'

neuronetz = simnet(eqns, G, H, T, params, coupling, noise)

random.seed()

 
thist = np.linspace(0, max_tau, 10000)

xhist = np.zeros(len(thist)) 

yhist = np.zeros(len(thist)) 

dic = {'t' : thist}

for i in range(len(G)):

  if i==1:
    dic['x'+str(i)] = xhist-1.5
    dic['y'+str(i)] = yhist-0.5
  else:
    dic['x'+str(i)] = xhist+2
    dic['y'+str(i)] = yhist-0.5

neuronetz.ddeN.hist_from_arrays(dic)


""" Start simulation with t = [0,tmax] """

tmax = 50
neuronetz.run(tmax)


print "FitzHugh-Nagumo"
print "variables : " , neuronetz.ddeN.vars
print "equations : ", neuronetz.ddeN.eqns
print "parameters : ", neuronetz.params
print "noise : ", neuronetz.noise
print "delays : ", neuronetz.ddeN.delays

sol_samp1 = neuronetz.ddeN.sample(0, dt=0.1)

t = sol_samp1['t'][0:]
x = {}
y = {}

for i in range(0,len(G[0])):
  x[i] = sol_samp1['x'+str(i)][0:]
  y[i] = sol_samp1['y'+str(i)][0:]
  
#f = open('deneme.dat','w')	

#for i, t0 in enumerate(t):	
	#f.write("%s\t" % (t0))
	#f.write("%.5f\t" % (sol_samp1['x'][i]) )
	#f.write("%.5f\t" % (sol_samp1['y'][i]))
	#f.write("\n")
#f.close()

print len(sol_samp1['t'][0:])
pl.subplot(221)
pl.plot(sol_samp1['t'],sol_samp1['x1'],'r')
pl.plot(sol_samp1['t'],sol_samp1['y1'], 'b')
#pl.plot(sol_calc['t'],sol_calc['x'],'or')
#pl.plot(sol_calc['t'],sol_calc['y'],'ob')
pl.xlabel('$t$')
pl.ylabel('$x_1,y_1$')

pl.subplot(222)
pl.plot(sol_samp1['x1'],sol_samp1['y1'], 'r')
pl.plot(sol_samp1['x1'][0],sol_samp1['y1'][0], 'or')
#pl.plot(sol_calc['x'],sol_calc['y'],'or')
#pl.plot(x_range, nullcl_01(x_range), 'b')
#pl.plot(x_range, nullcl_02(x_range), 'k')
#pl.plot(x_int,y_int,'ok')

#pl.axis([-2.3, 2.3, -1, 1])
pl.xlabel('$x_1$')
pl.ylabel('$y_1$')

pl.subplot(223)
pl.plot(sol_samp1['t'],sol_samp1['x2'],'r')
pl.plot(sol_samp1['t'],sol_samp1['y2'], 'b')
#pl.plot(sol_calc['t'],sol_calc['x'],'or')
#pl.plot(sol_calc['t'],sol_calc['y'],'ob')
pl.xlabel('$t$')
pl.ylabel('$x_2,y_2$')

pl.subplot(224)
pl.plot(sol_samp1['x2'],sol_samp1['y2'], 'r')
pl.plot(sol_samp1['x2'][0],sol_samp1['y2'][0], 'or')
pl.xlabel('$x_2$')
pl.ylabel('$y_2$')

pl.show()

# plot for smaller data matrices over different dt's at once
#pl.subplot(2,3,1)
#for i in range(1):
  #pl.plot(sampleSol['t'],sampleSol['x%s' % i], 'r', label='x, $\Delta$ t = 1 s')
  #pl.plot(sampleSol['t'],sampleSol['y%s' % i],'b',label='y, $\Delta$ t = 1 s')
  #pl.plot(adaptiSol['t'],adaptiSol['x%s' % i], '.r' )
#pl.xlabel('$t$')
#pl.ylabel('$x_1,y_1$')
#pl.legend()  

#pl.subplot(2,3,2)
#for i in range(1):
  #pl.plot(sampleSol_2['t'],sampleSol_2['x%s' % i], 'r', label='x, $\Delta$ t = 2.5 s')
  #pl.plot(sampleSol_2['t'],sampleSol_2['y%s' % i], 'b', label='y, $\Delta$ t = 2.5 s')
#pl.xlabel('$t$')
#pl.ylabel('$x_1,y_1$')
#pl.legend()  

#pl.subplot(2,3,3)  
#for i in range(1):
  #pl.plot(sampleSol_3['t'],sampleSol_3['x%s' % i], 'r', label='x, $\Delta$ t = 5 s')
  #pl.plot(sampleSol_3['t'],sampleSol_3['y%s' % i], 'b',label='y, $\Delta$ t = 5 s')  
#pl.xlabel('$t$')
#pl.ylabel('$x_1,y_1$')
#pl.legend()  
  

#pl.subplot(2,3,4)
#pl.plot(sampleSol['x1'],sampleSol['y1'],'k', label='$\Delta$ t = 1 s')
#pl.plot(adaptiSol['x1'],adaptiSol['y1'],'ok',label='adaptive')
#pl.legend()
#pl.ylabel('$y_1$')
#pl.xlabel('$x_1$')

#pl.subplot(2,3,5)
#pl.plot(sampleSol_2['x1'],sampleSol_2['y1'],'k',label='$\Delta$ t = 2.5 s')
#pl.plot(adaptiSol['x1'],adaptiSol['y1'],'ok',label='adaptive')
#pl.legend()
#pl.ylabel('$y_1$')
#pl.xlabel('$x_1$')

#pl.subplot(2,3,6)
#pl.plot(adaptiSol['x1'],adaptiSol['y1'],'ok',label='adaptive')
#pl.plot(sampleSol_3['x1'],sampleSol_3['y1'],'k',label='$\Delta$ t = 5 s')
#pl.legend()
#pl.ylabel('$y_1$')
#pl.xlabel('$x_1$')

#pl.show()