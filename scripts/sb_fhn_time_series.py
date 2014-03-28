import numpy as np
import pylab as pl
import sys
import math
import random
from pydelay import dde23
from netpy import simnet


#test_mtx = sys.argv[1]
#distance = sys.argv[2]

test_mtx = np.ones((20,20))
distance = np.ones((20,20))

for i in range(0,len(test_mtx)):
	for j in range(0,len(distance)):
		test_mtx[i][j]=0


G = test_mtx
d = distance

#G = np.loadtxt(test_mtx)
#d = np.loadtxt(distance)

eqns = { 
		 r'x{i}': '(y{i} + gamma * x{i} - pow(x{i},3.0)/3.0) * TAU',
         r'y{i}': '- (x{i} - alpha + b * y{i}) / TAU'
			}

params = { 
		 'gamma': 1.0, 
		 'alpha': 0.9,  
         'b': 0.2,
		 'TAU' : 1,	  
		 'D' : 0,
		 'v' : 70.0,
		 'sigma' : 0.005 } 							  # change noise strength

noise = {'x': 'D * gwn()', 'y': 'D * gwn()'}

C = params['sigma']

H = [ [C , 0] , [0 , 0] ]

# time delay matrix, distance between nodes over velocity
T_delay = d/params['v']								

max_tau = math.ceil(T_delay.max())

coupling = '-{G:.1f}*{H}*{var}(t-{tau})' #??

# initializing the solver
neuronetz = simnet(eqns, G, H, T_delay, params, coupling, noise)

random.seed()

# initial conditions for the variables / history 
thist = np.linspace(0, max_tau, 1000)
xhist = np.zeros(len(thist))
yhist = np.zeros(len(thist)) + 0.5

# importing initial conditions in dictionary
dic = {'t' : thist}
for i in range(len(G)):
	dic['x'+str(i)] = xhist
	dic['y'+str(i)] = yhist
print dic

# history function from dictionary of arrays
neuronetz.ddeN.hist_from_arrays(dic)

# starting simulation with t=[0,tmax]
tmax = 50
neuronetz.run(tmax)

sampleSol = neuronetz.ddeN.sample(0,tmax,dt=0.1)
adaptiSol = neuronetz.ddeN.sol

t_sample = sampleSol['t']

print dic
print sampleSol
x = {}
y = {}


print len(G[0])  	# column numbers of matrix G

#for i in range(0,len(G[0])):
#	x[i] = sampleSol['x'+str(i)][0:]


#print x






















