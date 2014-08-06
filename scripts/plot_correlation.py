#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# plotting correlation matrixes from various input

import networkx as nx
import numpy as np
from math import factorial 
import matplotlib.pyplot as pl	
import random as rnd
import sys  
import glob
import os


# check the loaded matrix if it is symmetric
def load_matrix(file):
	A  = np.loadtxt(file, unpack=True)
	AT = np.transpose(A)
	# check the symmetry				
	if A.shape[0] != A.shape[1] or not (A == AT).all():
		print "error: loaded matrix is not symmetric"
		raise ValueError
	return AT
	
def plot_corr(corr_matrix, simfile ):
	
	# plots correlation matrix
	
	N_col  = np.shape(corr_matrix)[1]
	extend = (0.5 , N_col+0.5 , N_col+0.5, 0.5 )	
	pl.imshow(corr_matrix, interpolation='nearest', extent=extend)
	cbar = pl.colorbar()
	cbar.set_ticks(np.arange(0, 1.0+0.1, 0.1) )
	cbar.set_ticklabels(np.arange(0, 1.0+0.1, 0.1))
	for t in cbar.ax.get_yticklabels():
		t.set_fontsize(15)
	pl.title('BOLD-fMRI', fontsize=20)
	pl.xticks(fontsize = 20)
	pl.yticks(fontsize = 20)
	#pl.xlabel('Nodes', fontsize = 20)
	#pl.ylabel('Nodes')
	print np.arange(0, 1.0+0.1, 0.1)
	#image_name = simfile[0:-4] + '_CORR.eps'	
	#pl.savefig(image_name, format="eps")
	#pl.show()
	return  	
	
if __name__ == '__main__':
	usage = 'Usage: %s method correlation_matrix [threshold]' % sys.argv[0]
	try:
		input_name = sys.argv[1]
	except:
		print usage
		sys.exit(1)
		
input_data		=		load_matrix(input_name)
plot_corr(input_data , input_name)
pl.show()
