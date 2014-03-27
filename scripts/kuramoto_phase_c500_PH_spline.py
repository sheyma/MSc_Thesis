#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import sys
from netpy import simnet
import random
#import pylab as pl
import math
import os

"""Attention: alphabethical order of equation advised!!! """
G = np.loadtxt(sys.argv[1])
N = len(G)

"""Attention: alphabethical order of equation advised!!! """
OMEGA   =  376.9911* np.ones(N)   # 376.9911
#A   = 1.3 * np.ones(N)
#A   = 1.3 * np.ones(N)  + 0.1 * np.random.rand(N)
    
print 'generating distribution...'
param_mu = 376.9911    #62.832
param_sigma = 1.
#g = open('data_values_gaussian_mu%f_sigma%f_N%d.dat' % (param_mu, param_sigma, N),'w')
numbers = []
for i in range(N):
  number = random.gauss(param_mu, param_sigma)      
  numbers.append(number)
  OMEGA[i] = number
#g.write('%f\n' % (number))
#print 'done!'

#print 'eqns', eqns
params = {
         #'omega': 376.9911,
        #'sigma': 50.0,
        'sigma': 500.0,
        'phi': float(sys.argv[2]),
        #'alpha': float(sys.argv[2]),
        'tau': 0., 
        'D': 0.
	#'v' : 70.0,
	#'tau' : 1.0, 
	}

print 'params' , params
#noise = {'x': 'D * gwn()'}  # this is for the noise for the first system only if noise = {'x': 'D * gwn()', 'y': 'D * gwn()'} it is for all nodes
noise = { 'x': '0' }

""" Topology """
#G = np.loadtxt(sys.argv[1])
C = params['sigma']
H = [ [C, 0],
      [0, 0] ]
#H = [[0,0],[0,0]]
print 'H', H

"""Delay-Matrix. """
#simple version: all delay identical => specily 'tau' in params-array
T = params['tau'] * np.ones_like(G)
max_tau = math.ceil(T.max())

""" coupling term """
#diffusive coupling: \dot var_i = ... (var_j - var_i)
#coupling = '+{G:.12f}*{H}*({var}-{self})'
#direct coupling: \dot var_i = ... var_j
#coupling = '-{G:.1f}*{H}*{var}(t-{tau})'
#Kuramoto phase oscillators
#coupling = '+{G:.12f}*{H}*{sigma}*sin({var}(t-{tau})-1.*{self}-alpha)'
#coupling = '-{G:.12f}*{sigma}*sin({self}-{var}+alpha)'
#coupling = '-{G:.12f}*{sigma}*sin({self}-{var}-alpha)'
#coupling = '+{G:.12f}*{sigma}*sin({self}-{var}-alpha)'
#coupling = '+{G:.12f}*{sigma}*sin({var}-{self}-alpha)'
coupling = '+{G:.12f}*{sigma}*sin({var}-{self}-phi)'

for i in range(N):
  params['omega_%s' %i] = OMEGA[i]

eqns = {r'x{i}' : 'omega_{i}'}

"""Let's go """
neuronetz = simnet(eqns, G, H, T, params, coupling, noise)
#print neuronetz.eqnsN
random.seed()
# generate history function/initial conditions: 
#100 specifies the number of points in the history array
#thist = np.linspace(0, 100, 10000) # thist = np.linspace(0, 100, 10000) does'n have any effect!!! this is history for td
#xhist = np.zeros(len(thist))
##yhist = np.zeros(len(thist)) + 0.5
#dic = {'t' : thist}
#for i in range(len(G)):
  ## all elements identical
  #dic['x'+str(i)] = xhist
  ## constant shift added
  #dic['x'+str(i)] = xhist + i / len(G)
  # random values added
  #dic['x'+str(i)] = xhist + 1.*random.random()
  #dic['y'+str(i)] = yhist

#neuronetz.ddeN.hist_from_arrays(dic)

""" Start simulation with t = [0,tmax] """
neuronetz.run(tmax=50) # original 50 
#neuronetz.run(tmax=550) # original 550 

# alternative way to generate history function/initial conditions
#initial_conditions = {'x0': -0.95, 'y0': -0.95+ pow(0.95,3.0)/3.0,
                      #'x1': -0.95, 'y1': -0.95+ pow(0.95,3.0)/3.0}
#neuronetz.run(initial_conditions, tmax=600)
#series = neuronetz.sol #sampled at Delta t = 0.0001 s = 0.1 ms

calculated_series = neuronetz.adaptive_sol #varying step size 
#sample solution for output starting at t=-max_tau
solution = neuronetz.ddeN.sample(0, dt=0.0001) #sampled at Delta t = 0.001 s = 1 ms
sp_sol = neuronetz.spline_sol #sampled at Delta t = 0.001 s = 1 ms
#solution = neuronetz.ddeN.sample(0, dt=0.001) #sampled at Delta t = 0.001 s = 1 ms

t = solution['t']
#print only last 10% of time series
#tpre = solution['t']
#t = tpre[int(0.9*len(tpre)):]

print "starting print-out of data..."

t=solution['t'][0:]
ts=calculated_series['t'][0:]
tsp=sp_sol['t'][0:]
#coupling = '+{G:.12f}*{sigma}*sin({var}-{self}-alpha)'
coupling = '+{G:.12f}*{sigma}*sin({var}-{self}-phi)'

x = {}
#y = {}
calc_x = {}
spl_x = {}
for i in range(0,len(G[0])):
  x[i] =  solution['x'+str(i)][0:]
  calc_x[i] = calculated_series['x'+str(i)][0:]
  spl_x[i] = sp_sol['x'+str(i)][0:]
  #x[i] =  np.sin(np.unwrap(solution['x'+str(i)]))[0:]
  #y[i] =  solution['y'+str(i)][0:]

#filepath='/home/vuksanovic/Python/AAL_data'
#filename = "phase_d%s_df%s_c%.2f_a%.2f.dat" % (params['D'], param_mu, params['sigma'],params['alpha'])
filename = "phase_d%s_df%s_c%.2f_a%.2f.dat" % (params['D'], param_mu, params['sigma'],params['phi'])

#f = open(os.path.join(filepath,filename),'w')  
f = open(os.path.join(filename),'w')
g = open(os.path.join(filename[:-4]+'_adaptive.dat'),'w')
h = open(os.path.join(filename[:-4]+'_spline.dat'),'w')

for i, t0 in enumerate(t):
  if t0 >=45:
  #if t0 >=0:
    try:
      f.write("%s\t" % (t0))
    except SyntaxError:
      print t0
      sys.exit()
    for j in range(0, len(x)):
      f.write("%.2f\t%.4f\t" % (float(x[j][i]), np.sin(float(x[j][i]))) )
    #f.write("%.2f\t" % (float(x[j][i])))
    f.write("\n")

for i, t0 in enumerate(ts):
  if t0 >=45:
  #if t0 >=0:
    try:
      g.write("%s\t" % (t0))
    except SyntaxError:
      print t0
      sys.exit()
    for j in range(0, len(calc_x)):
      g.write("%.2f\t%.4f\t" % (float(calc_x[j][i]), np.sin(float(calc_x[j][i]))) )
    #f.write("%.2f\t" % (float(x[j][i])))
    g.write("\n")

for i, t0 in enumerate(tsp):
  if t0 >=45:
  #if t0 >=0:
    try:
      h.write("%s\t" % (t0))
    except SyntaxError:
      print t0
      sys.exit()
    for j in range(0, len(calc_x)):
      h.write("%.2f\t%.4f\t" % (float(spl_x[j][i]), np.sin(float(spl_x[j][i]))) )
    #f.write("%.2f\t" % (float(x[j][i])))
    h.write("\n")

f.close()
g.close()
h.close()
print "done!"

#for i in range(1):
     ##pl.plot(series['t'], series['x%s' % i], '+', label='Delta t = 0.01ms')
     #pl.plot(solution['t'], solution['x%s' % i], '.', label='Delta t = 1 ms')
     ##pl.plot(calculated_series['t'], calculated_series['x%s' % i], 'o', label='calculated points')
     ##l.plot(t[:-1], t[1:] - t[:-1], '.r', label='step size')
#pl.show()

#for i in range(1):
     #pl.plot(series['t'], np.sin(series['x%s' % i]), '+-', label='Delta t = 0.01ms')
     #pl.plot(solution['t'], np.sin(solution['xs%s' % i]), '.', label='Delta t = 1 ms')
     #pl.plot(calculated_series['ts'], np.sin(calculated_series['xs%s' % i]), 'o-', label='calculated points')
   
#pl.ylim((0,3))
#pl.legend()
#pl.show()

##for i in range(N):
    ##pl.plot(neuronetz.sol['t'], neuronetz.sol['x_%s' % i])