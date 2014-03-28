import numpy as np
import pylab as pl
import sys
from pydelay import dde23

eqns = { 
		 'x': '(y + gamma * x - pow(x,3.0)/3.0) * TAU',
         'y': '- (x - alpha + b * y) / TAU'
			}

params = { 
		 'gamma': 1.0, 
		 'alpha': 0.9,  
         'b': 0.2,
		 'TAU' : 1,	  
		 'D' : 0} 								  # change noise strength

noise = {'x': 'D * gwn()', 'y': 'D * gwn()'}

dde = dde23(eqns=eqns, params=params, noise=noise) # initialize the solver

tfinal = 50

dde.set_sim_params(tfinal)

dde.hist_from_funcs({'x': lambda t: 1,
					 'y': lambda t: 0})	  # initial conditions
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
pl.xlabel('$x$')
pl.ylabel('$y$')

#pl.show()
