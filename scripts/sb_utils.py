#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np
import re
import subprocess as sp

# load input data as numpy matrix
def load_matrix(infile, unpack=False):
	print "reading data ..."
	
	# handle xz files transparently
	if re.search(r'\.xz$', infile, flags=re.IGNORECASE):
		# non-portable but we don't want to depend on pyliblzma module
		xzpipe = sp.Popen(["xzcat", infile], stdout=sp.PIPE)
		x_infile = xzpipe.stdout
	else:
		# in non-xz case we just use the file name instead of a file
		# object, numpy's loadtxt() can deal with this
		x_infile = infile
	
	A  = np.loadtxt(x_infile, unpack=unpack)
	print "shape of input matrix : " , np.shape(A)
	return A


# return file name without extensions like ".dat", ".dat.xz", etc.
def get_dat_basename(infile):
	basename = re.sub(r'\.dat(|\.xz|\.gz|\.bz2)$', '', infile, flags=re.IGNORECASE)
	return basename
