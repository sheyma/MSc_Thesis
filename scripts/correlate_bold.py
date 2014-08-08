#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import subprocess as sp
import numpy as np
import sys 
import math




def correl(input_name , bold_input):
	# correlation coefficient among the columns of bold_input calculated
	# numpy array must be transposed to get the right corrcoef
	
	transpose_input = np.transpose(bold_input)
	correl_matrix   = np.corrcoef(transpose_input)
	
	#file_name       = str(input_name[:-4] + '_corrcoeff.dat') 	
	#np.savetxt(file_name, correl_matrix, '%.10f',delimiter='\t')
	return correl_matrix

def image(bold_input, simfile):
	
	# plots simulated functional connectivity
	pl.figure(4)
	N_col = np.shape(bold_input)[1]
	extend = (0.5 , N_col+0.5 , N_col+0.5, 0.5 )	
	pl.imshow(bold_input, interpolation='nearest', extent=extend)
	pl.colorbar()
	
	image_name = simfile[0:-4] + '_CORR.eps'	
	#pl.savefig(image_name, format="eps")
	#pl.show()
	return  
	
