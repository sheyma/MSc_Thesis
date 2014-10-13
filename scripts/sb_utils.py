#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np
import subprocess as sp

# load input data as numpy matrix
def load_matrix(infile):
	print "reading data ..."
	
	# handle xz files transparently
	if infile.endswith(".xz"):
		# non-portable but we don't want to depend on pyliblzma module
		xzpipe = sp.Popen(["xzcat", infile], stdout=sp.PIPE)
		x_infile = xzpipe.stdout
	else:
		# in non-xz case we just use the file name instead of a file
		# object, numpy's loadtxt() can deal with this
		x_infile = infile
	
	A  = np.loadtxt(x_infile, unpack=False)
	print "shape of input matrix : " , np.shape(A)
	return A
