#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import networkx as nx
#from networkx.algorithms import bipartite
import numpy as np
from math import factorial 
import matplotlib.pyplot as pl
import random as rnd
import sys  
import glob
import os

def load_matrix(file):
	a_ij = np.loadtxt(file, unpack=True)
	return a_ij

	

	
	
if __name__ == '__main__':
	usage = 'Usage: %s method correlation_matrix [threshold]' % sys.argv[0]
	try:
		matrix_01 = sys.argv[1]
		matrix_02 = sys.argv[2]
		#input_threshold = sys.argv[3]
	except:
		print usage
		sys.exit(1)
		
A = load_matrix(matrix_01)
B = load_matrix(matrix_02)	
pair_AB = np.zeros(A.shape)

for row in range(A.shape[0]):
		for col in range(A.shape[1]):
			pair_AB[row,col] = A[row,col] * B[row,col]
