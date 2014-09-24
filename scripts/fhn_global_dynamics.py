#!/usr/bin/python2.7 
# -*- coding: utf-8 -*-

""" 
	visualizing the gloabal dynamics of FitzHugh Nagumo model with 
	different parameters chosen from literature. 
	
	Literature ; 
	[PAN12] = D.Rosin, P.Hövel, E.Schöll, "Synchronization of 
	coupled neuronal oscillators with heterogenous delays", 2012
	
	[GHOS08a] = A.Ghosh, Y.Rho, A.R.McIntosh, R.Kötter, V.K.Jirsa,
	"Cortical Network Dynamics with time delays reveals functional
	connectivity in the resting brain", 2008
	
	[VUK13] = V.Vuksanovic, P.Hövel, "Large-scale neuronal network
	model for functional networks of the human cortex", 2013
	
	output : plotting activator(x) and inhibitor(y) over time(t)
	and in state space
"""

import numpy as np
import pylab as pl
from pydelay import dde23
from scipy.optimize import fsolve
from pylab import *
import math

## [PAN12] global dynamics ############################################
eqns = { 
   'x1': '(x1-pow(x1,3)/3.0-y1+C*(x2(t-tau)-x1)+K*(x1(t-tau)-x1))/eps',
   'y1': 'x1 + a',
   'x2': '(x2 -pow(x2,3)/3.0-y2+C*(x1(t-tau)-x2)+K*(x2(t-tau)-x2))/eps',
   'y2': 'x2 + a'
        }
 
params = params = { 
        'a'     : 1.3,
        'eps'   : 0.01,
        'C'     : 0.5,
        'K'     : 0.1,
        'tau'   : 3.0,
        'tau_2' : 3.0 #dde23 does not get different time delays
        }
        
# nullclines 
def nullcl_01(X):
  return -(pow(X,3)/3 - X)
  #make x minus
  
def nullcl_02(X):
  return (-params['a']  ) 

# intersection of nullclines [PAN12] : 
X_int = -params['a']
Y_int = nullcl_01(-params['a'])

x_limit = 2.5
y_limit = 2
x_range = np.linspace(-x_limit,x_limit,500)

xx_1 = -params['a'] * np.ones(len(x_range))
yy_1 = nullcl_01(x_range)

print "intersection of nullclines x_0 , y_0 : ", X_int, Y_int

# initalise the solver
dde = dde23(eqns=eqns, params=params)

tfinal = 500
dde.set_sim_params(tfinal)

dde.hist_from_funcs({'x1': lambda t : -0.05 , 'y1': lambda t: -0.75 })

dde.run()

# sampling the numerical solution with sample size dt
sol = dde.sample(0,tfinal,dt=0.01)

# save the solution
x1 = sol['x1']
y1 = sol['y1']
x2 = sol['x2']
y2 = sol['y2']
t  = sol['t']

# plot
fig = pl.figure(num=None, figsize=(14, 6), dpi=100, facecolor='w', edgecolor='k')
fig.suptitle('[PAN12] Global Dynamics :  a = '+str(params['a'])+
	      r'  $\varepsilon$'+' = '+str(params['eps']) +
	      '  C = '+ str(params['C']) +'  '  + r'$\tau^C$= '+
	      str(params['tau'])+ '  K = '+ str(params['K']) +'  '  + 
	      r'$\tau^K$= '+ str(params['tau']),fontsize=14, 
	      fontweight='bold')

index = np.arange(int(len(t)*0.96),len(t),1)

pl.subplot(221)
pl.plot(t[index], x1[index], 'r',label='$x_1(t)$')
pl.plot(t[index], y1[index], 'b',label='$y_1(t)$')
pl.xlabel('$t$')
pl.ylabel('$x_1(t) , y_1(t)$')
lg = legend()
lg.draw_frame(False)

pl.subplot(222)
pl.plot(x1[index], y1[index], 'r')
pl.plot(xx_1, x_range,'k', label='$y_{nullcline}$')
pl.plot(x_range,yy_1 ,'b', label='$x_{nullcline}$')
pl.plot(X_int,Y_int, 'ok', linewidth=3)
pl.xlabel('$x_1$')
pl.ylabel('$y_1$')
lg = legend()
lg.draw_frame(False)
pl.axis([-2.3, 2.3, -1.5, 1.5])

pl.subplot(223)
pl.plot(t[index], x2[index], 'r',label='$x_2(t)$')
pl.plot(t[index], y2[index], 'b',label='$y_2(t)$')
pl.xlabel('$t$')
pl.ylabel('$x_2(t) , y_2(t)$')
lg = legend()
lg.draw_frame(False)

pl.subplot(224)
pl.plot(x2[index], y2[index], 'r')
pl.xlabel('$x_2$')
pl.ylabel('$y_2$')
pl.plot(x_range,nullcl_01(x_range),'b', label='$x_{nullcline}$')
pl.plot(xx_1, x_range,'k', label='$y_{nullcline}$')
pl.plot(X_int,Y_int, 'ok', linewidth=3)
pl.axis([-2.3, 2.3, -1.5, 1.5])
lg = legend()
lg.draw_frame(False)
#pl.savefig("PAN12_global_dynamics_C.eps",format="eps")
#pl.show()

########################################################################

## [GHO08a] global dynamics ###########################################
eqns = { 
        'x1': '(y1 + gamma*x1 - pow(x1,3)/3.0)* TAU + C*(x2(t-tau)-x1)', 
        #+ K*(x1(t-tau)-x1) ',
        'y1': '-(x1 - alpha + b*y1)/ TAU',
        'x2': '(y2 + gamma*x2 - pow(x2,3)/3.0 )*TAU + C*(x1(t-tau)-x2)', 
        #+ K*(x2(t-tau)-x2)',
        'y2': '-(x2 - alpha + b*y2)/ TAU'
        }

#set the parameters and the delay
params = { 
		'alpha': 1.05,  #[GHO08a]
        #'alpha': 0.85,  #[VUK13] 
		'gamma': 1.0,
        'b': 0.2,
        'C': 3,
        'K' : 0,
        'TAU': 1.25,
        'tau' : 5.00}
       
x_limit = 2.5
x_range = np.linspace(-2.5,2.5,500)

# nullclines
def nullcl_01(X):
  return -(pow(X,3)/3 - params['gamma']*X)
  # make x minus
  
def nullcl_02(X):
  return (params['alpha'] - X ) / params['b']

# calculate the intersection of nullclines
def intersection(X):
  return nullcl_01(X) - nullcl_02(X)

X0    = 0
X_int = fsolve(intersection,X0)
Y_int = nullcl_01(X_int)

print "intersection of nullclines x_0 , y_0 : ", X_int, Y_int

# initalise the solver
dde = dde23(eqns=eqns, params=params)

tfinal = 500
dde.set_sim_params(tfinal)

dde.hist_from_funcs({'x1': lambda t : -0.05 , 'y1': lambda t: -0.75 })

dde.run()

# sampling the numerical solution with sample size dt
sol = dde.sample(0,tfinal,dt=0.01)

# save the solution
x1 = sol['x1']
y1 = sol['y1']
x2 = sol['x2']
y2 = sol['y2']
t  = sol['t']

fig = pl.figure(num=None, figsize=(14, 6), dpi=100, facecolor='w', edgecolor='k')
### [GHO08a] and [VUK13]############################################
fig.suptitle('[GHO08a]- Global Dynamics :  '+r'$\alpha$ = ' +str(params['alpha'])+
#fig.suptitle('[VUK13]- Global Dynamics :  '+r'$\alpha$ = ' +str(params['alpha'])+
	      r'  $\gamma$ = '+str(params['gamma']) + ' $ b$ = '+ 
	      str(params['b']) + r'  $\tau$ = '+str(params['TAU']) + 
	      '  C = '+ str(params['C']) +'  '  + r'$\tau^C$= ' +str(params['tau']) #+
	      #'  K = '+ str(params['K']) +'  '  + r'$\tau^K$= '+
	      #str(params['tau'])#
	      , fontsize=14, fontweight='bold')

index = np.arange(0,len(t),1)

pl.subplot(221)
pl.plot(t[index], x1[index], 'r',label='$x_1(t)$')
pl.plot(t[index], y1[index], 'b',label='$y_1(t)$')
pl.xlabel('$t$')
pl.ylabel('$x_1(t) , y_1(t)$')
lg = legend()
lg.draw_frame(False)

pl.subplot(222)
pl.plot(x1[index], y1[index], 'r')
pl.plot(x_range, nullcl_01(x_range),'b', label='$y_{nullcline}$')
pl.plot(x_range, nullcl_02(x_range),'k', label='$x_{nullcline}$')
pl.plot(X_int,Y_int, 'ok', linewidth=3)
pl.xlabel('$x_1$')
pl.ylabel('$y_1$')
lg = legend()
lg.draw_frame(False)
pl.axis([-2.3, 2.3, -1.5, 1.5])

pl.subplot(223)
pl.plot(t[index], x2[index], 'r',label='$x_2(t)$')
pl.plot(t[index], y2[index], 'b',label='$y_2(t)$')
pl.xlabel('$t$')
pl.ylabel('$x_2(t) , y_2(t)$')
lg = legend()
lg.draw_frame(False)

pl.subplot(224)
pl.plot(x2[index], y2[index], 'r')
pl.xlabel('$x_2$')
pl.ylabel('$y_2$')
pl.plot(x_range, nullcl_01(x_range),'b', label='$y_{nullcline}$')
pl.plot(x_range, nullcl_02(x_range),'k', label='$x_{nullcline}$')
pl.plot(X_int,Y_int, 'ok', linewidth=3)
pl.axis([-2.3, 2.3, -1.5, 1.5])
lg = legend()
lg.draw_frame(False)
pl.show()
