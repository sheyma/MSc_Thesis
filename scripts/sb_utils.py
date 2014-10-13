#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np

# load input data as numpy matrix
def load_matrix(file):
	print "reading data ..."
	A  = np.loadtxt(file, unpack=False)
	print "shape of input matrix : " , np.shape(A)
	return A
