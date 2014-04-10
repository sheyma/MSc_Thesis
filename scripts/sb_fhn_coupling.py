#!/usr/bin/python2.7 

# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl
from pydelay import dde23
from scipy.optimize import fsolve
from pylab import *

# PAN12 global dynamics

#eqns = { 
        #'x1': '(x1 - pow(x1,3)/3.0 - y1 + C*(x2(t-tau) - x1) + K*(x1(t-tau)-x1) )/eps',
        #'y1': 'x1 + a',
        #'x2': '(x2 - pow(x2,3)/3.0 - y2 + C*(x1(t-tau) - x2) + K*(x2(t-tau)-x2) )/eps',
        #'y2': 'x2 + a'
        #}
 
#params = params = { 
        #'a': 1.3,
        #'eps': 0.01,
        #'C': 0.5,
        #'K' : 0,
        #'tau': 3.0,
        #'tau_2' : 3.0 #dde23 does not get different time delays
        #}

#x_limit = 2.5
#y_limit = 2
#x_ = np.linspace(-x_limit,x_limit,10000)
#y_ = x_ - pow(x_ , 3)/3
#yy = np.linspace(-y_limit, y_limit, 10000)
#xx = -params['a'] * np.ones(len(yy))

eqns = { 
        'x1': '(y1 + gamma*x1 - pow(x1,3)/3.0 ) * TAU + C*(x2(t-tau) - x1)', #+ K*(x1(t-tau)-x1) ',
        'y1': '-(x1 - alpha + b*y1)/ TAU',
        'x2': '(y2 + gamma*x2 - pow(x2,3)/3.0 )* TAU + C*(x1(t-tau) - x2)', #+ K*(x2(t-tau)-x2)',
        'y2': '-(x2 - alpha + b*y2)/ TAU'
        }

#set the parameters and the delay
params = { 
        'alpha': 0.89,
	'gamma': 0.9,
        'b': 0.1,
        'C': 5,
        'K' : 4,
        'TAU': 4,
        'tau' : 3 }


        
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

X0 = -2
X_int = fsolve(intersection,X0)
Y_int = nullcl_01(X_int)

print "intersection of nullclines x_0 , y_0 : ", X_int, Y_int
# initalise the solver
dde = dde23(eqns=eqns, params=params)

# set the simulation parameters
tfinal = 500
dde.set_sim_params(tfinal=500)

# When nothing is specified, the history for all variables 
# is initialized to 0.
#
dde.hist_from_funcs({'x1': lambda t : -0.05 , 'y1': lambda t: -0.75 })


# run the simulation
dde.run()

# sample the solution with sample size dt=0.01 between 170 and 200
sol = dde.sample(480,tfinal,dt=0.1)

# plot the solution
x1 = sol['x1']
y1 = sol['y1']
x2 = sol['x2']
y2 = sol['y2']
t = sol['t']

fig = pl.figure(num=None, figsize=(14, 6), dpi=100, facecolor='w', edgecolor='k')
#fig.suptitle('[PAN12] Global Dynamics :  a = '+str(params['a'])+
	      #r'  $\varepsilon$'+' = '+str(params['eps']) +
	      #'  C = '+ str(params['C']) +'  '  + r'$\tau^C$= '+
	      #str(params['tau'])+ '  K = '+ str(params['K']) +'  '  + 
	      #r'$\tau^K$= '+ str(params['tau']),fontsize=14, fontweight='bold')
fig.suptitle('[GHO08] Global Dynamics :  '+r'$\alpha$ = ' +str(params['alpha'])+
	      r'  $\gamma$ = '+str(params['gamma']) + ' $ b$ = '+ 
	      str(params['b']) + r'  $\tau$ = '+str(params['TAU']) + 
	      '  C = '+ str(params['C']) +'  '  + r'$\tau^C$= ' #+
	      #'  K = '+ str(params['K']) +'  '  + r'$\tau^K$= '+
	      #str(params['tau'])#
	      , fontsize=14, fontweight='bold')
	      
pl.subplot(221)
pl.plot(t, x1, 'r',label='$x_1(t)$')
pl.plot(t, y1, 'b',label='$y_1(t)$')
pl.xlabel('$t$')
pl.ylabel('$x_1(t) , y_1(t)$')
lg = legend()
lg.draw_frame(False)

pl.subplot(222)
pl.plot(x1, y1, 'r')
#pl.plot(x_, y_ , 'b',label='$x_{nullcline}$')
#pl.plot(xx,yy,'k',label='$y_{nullcline}$')
#pl.plot(-params['a'],(-params['a']+pow(params['a'],3)/3),'ok')
pl.xlabel('$x_1$')
pl.ylabel('$y_1$')
pl.plot(x_range,nullcl_01(x_range),'b', label='$x_{nullcline}$')
pl.plot(x_range,nullcl_02(x_range),'k',label='$y_{nullcline}$')
pl.plot(X_int,Y_int, 'ok', linewidth=3)
lg = legend()
lg.draw_frame(False)
pl.axis([-2.3, 2.3, -1.5, 1.5])
#lg = legend()
#lg.draw_frame(False)

pl.subplot(223)
pl.plot(t, x2, 'r',label='$x_2(t)$')
pl.plot(t, y2, 'b',label='$y_2(t)$')
pl.xlabel('$t$')
pl.ylabel('$x_2(t) , y_2(t)$')
lg = legend()
lg.draw_frame(False)

pl.subplot(224)
pl.plot(x2, y2, 'r')
pl.xlabel('$x_2$')
pl.ylabel('$y_2$')
#pl.plot(x_, y_ , 'b',label='$x_{nullcline}$')
#pl.plot(xx,yy,'k',label='$y_{nullcline}$')
#pl.plot(-params['a'],(-params['a']+pow(params['a'],3)/3),'ok')
pl.plot(x_range,nullcl_01(x_range),'b', label='$x_{nullcline}$')
pl.plot(x_range,nullcl_02(x_range),'k',label='$y_{nullcline}$')
pl.plot(X_int,Y_int, 'ok', linewidth=3)
pl.axis([-2.3, 2.3, -1.5, 1.5])
lg = legend()
lg.draw_frame(False)

#pl.savefig("GHO08_global_dynamics_C_K.eps",format="eps")
#pl.savefig("GHO08_global_dynamics_C.eps",format="eps")
#pl.savefig("PAN12_global_dynamics_C_K.eps",format="eps")
#pl.savefig("PAN12_global_dynamics_C.eps",format="eps")
pl.show()
