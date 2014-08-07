#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# plotting correlation matrixes from various input

import subprocess as sp
import numpy as np
import matplotlib.pyplot as pl	
import sys 


# user defined input name
if __name__ == '__main__':
	try:
		input_name = sys.argv[1]
	except:
		sys.exit(1)

# handle importing xz files transparently
if input_name.endswith(".xz"):
	xzpipe = sp.Popen(["xzcat", input_name], stdout=sp.PIPE)
	infile = xzpipe.stdout
else:
	infile = input_name
