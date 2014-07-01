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

params = {'gamma' : 1.0, #0.9
	  'alpha' : 0.85, #1.9
	  'TAU' : 1.25,
	  'b' : 0.2,
	  'sigma' : 0.9
	  }



def load_matrix(file):
	A = np.loadtxt(file, unpack=True)
	AT = np.transpose(A)
	return AT

gfilename = sys.argv[1]

AT = load_matrix(gfilename)

fig = pl.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')

fig.suptitle('FHN - time series :  '+r'$\alpha$ = ' +str(params['alpha'])+
	      r'  $\gamma$ = '+str(params['gamma']) + ' $ b$ = '+ 
	      str(params['b']) + r'  $\tau$ = '+str(params['TAU']) + 
	      '  C = ' + str(params['sigma']) +'  '  + r'$\Delta\tau_{ij}$=$d_{ij}/v$'
	      , fontsize=14, fontweight='bold')


pl.subplot(2,1,1)
pl.plot(AT[0:1000,0],AT[0:1000,1],'r', label='x(t)')
pl.plot(AT[0:1000,0],AT[0:1000,2],'b', label='y(t)')
pl.xlabel('$t$')
pl.ylabel('$x_1,y_1$')
lg = legend(loc=2)
lg.draw_frame(False)

pl.subplot(2,1,2)
pl.plot(AT[0:1000,0],AT[0:1000,3],'r', label='x(t)')
pl.plot(AT[0:1000,0],AT[0:1000,4],'b', label='y(t)')
pl.xlabel('$t$')
pl.ylabel('$x_2,y_2$')
lg = legend(loc=2)
lg.draw_frame(False)

pl.savefig("FHN_time_series_real.eps",format="eps")


pl.show()
