#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl
import sys
import math
import random
from pydelay import dde23
from netpy import simnet


test_mtx = sys.argv[1]
distance = sys.argv[2]

G = np.loadtxt(test_mtx)
d = np.loadtxt(distance)

eqns = { r'x{i}': '(y{i} + gamma * x{i} - pow(x{i},3.0)/3.0) * TAU',
         r'y{i}': '- (x{i} - alpha + b * y{i}) / TAU' }
		
params = { 
		 'gamma': 1.0, 
		 'alpha': 0.9,  
                 'b': 0.2,
		 'TAU' : 1,	  
		 'D' : 0,
		 'v' : 70.0,
		 'sigma' : 0.005 } 	# change noise strength
		 
noise = {'x': 'D * gwn()', 'y': 'D * gwn()'}

C = params['sigma']

H = [ [C , 0] , [0 , 0] ]

# time delay matrix, distance between nodes over velocity
T_delay = (d)/(params['v'])								

max_tau = math.ceil(T_delay.max())

coupling = '-{G:.1f}*{H}*{var}(t-{tau})' #??

# initializing the solver
neuronetz = simnet(eqns, G, H, T_delay, params, coupling, noise)

random.seed()

# initial conditions for the variables / history 
thist = np.linspace(0, max_tau, 10)
xhist = np.zeros(len(thist))
yhist = np.zeros(len(thist)) + 0.5

# importing initial conditions in dictionary
dic = {'t' : thist}
for i in range(len(G)):
	dic['x'+str(i)] = xhist
	dic['y'+str(i)] = yhist

# history function from dictionary of arrays
neuronetz.ddeN.hist_from_arrays(dic)

# starting simulation with t=[0,tmax]
tmax = 100
neuronetz.run(tmax)

dt = float(sys.argv[3])
sampleSol = neuronetz.ddeN.sample(0,tmax,dt = dt)
#sampleSol = neuronetz.ddeN.sample(0,tmax,dt=0.1) #dt= 1 s
#sampleSol_2 = neuronetz.ddeN.sample(0,tmax,dt=0.25) #dt= 2.5 s 	
#sampleSol_3 = neuronetz.ddeN.sample(0,tmax, dt=0.5) # dt = 5 s
adaptiSol = neuronetz.ddeN.sol

t_sample = sampleSol['t']
#t_sample_2 = sampleSol_2['t']
#t_sample_3 = sampleSol_3['t']

x = {}
y = {}

print "number of nodes : ", len(G[0])  	# column numbers of matrix G

for i in range(0,len(G[0])):
	x[i] = sampleSol['x'+str(i)][0:]
	y[i] = sampleSol['y'+str(i)][0:]

f = open(test_mtx[:-4]+'_sigma='+str(params['sigma']),'w')

for i, t0 in enumerate(t_sample):
  f.write('%s\t' % (t0))
  for j in range(0,len(x)):
     f.write('%.2f\t%.2f\t' % (float(x[j][i]), float(y[j][i]) ))
  f.write('\n')
f.close()

print str(dt)
pl.plot(sampleSol['t'],sampleSol['x1'], 'r', label=('x, $\Delta$ t =' % (str(dt)))  )
pl.legend()
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