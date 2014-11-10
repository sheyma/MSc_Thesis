#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
from math import factorial, sqrt 
import matplotlib.pyplot as pl	
from matplotlib.pyplot import FormatStrFormatter
import random as rnd
import sys  
import glob
import os
import scipy.stats as sistat
import collections

# check the loaded matrix if it is symmetric
def load_matrix(file):
	A  = np.loadtxt(file, unpack=True)
	AT = np.transpose(A)
	# check the symmetry				
	if A.shape[0] != A.shape[1] or not (A == AT).all():
		print "error: loaded matrix is not symmetric"
		raise ValueError
	return AT


def corr_histo(corr_matrix):
		
	#corr_flat = []
	## removing the diagonal elements in both matrices to get
	## more reasonable Pearson correlation coef. between them
	#for i in range(0, np.shape(corr_matrix)[0]):
		#for j in range(0, np.shape(corr_matrix)[1]):
			#if (i == j) :

				#tmp_a  = np.append(corr_matrix[i, 0:j], corr_matrix[i, j+1:])
				#corr_flat  = np.append(corr_flat, tmp_a)

	# merge 2D corr_matrix into 1D numpy array by flatten
	corr_flat = np.ndarray.flatten(corr_matrix) 
	corr_max  = 1.0
	corr_min  = -1.0
	bin_nu    = 100
	# a normalized histogram is obtained
	hist, bin_edges = np.histogram(corr_flat, bins=bin_nu, range=[corr_min, corr_max], normed =True)#, histtype='bar')
	print hist
	#hist = pl.hist(corr_flat, bins=bin_nu, range=[corr_min, corr_max], normed =True, histtype='bar')
	#pl.title(simfile)
	# type(hist) = <type 'tuple'> and len(hist) = 3
	# y_axis : hist[0] , normalized hist values
	# x_axis : hist[1] , start points of bins 
	return hist

# intersection method to compare two histograms
# HA and HB must be normalized histograms
def intersec_hists(HA, HB):
	minsum  = 0
	for i in range(0, len(HA)) :
		minsum = minsum + min(HA[i], HB[i])
	# normalize minsum 
	minsum  = float(minsum) /  sum(HB) 
	return minsum
	
def correl_hists(HA, HB):
	N  = len(HA)
	HA_bar  = sum(HA) / float(N)
	HB_bar  = sum(HB) / float(N)
	tmp1    = 0 
	tmp2    = 0
	tmp3	= 0
	for i in range(0, N) : 
		tmp1 = tmp1 + (HA[i] - HA_bar) * (HB[i] - HB_bar)
		tmp2 = tmp2 + pow((HA[i] - HA_bar) , 2)
		tmp3 = tmp3 + pow((HB[i] - HB_bar) , 2)
	d = float(tmp1) / sqrt(tmp2 * tmp3)
	return d 
	
# Bhattacharyya method to compare two histograms
# HA and HB must be normalized histograms
def bhatta_hists(HA, HB):
	N  = len(HA)
	HA_bar  = sum(HA) / float(N)
	HB_bar  = sum(HB) / float(N)
	S1      = 1./ sqrt(HA_bar*HB_bar*N*N)
	S2      = 0
	for i in range(0, N) :
		S2 = S2 + sqrt(HA[i]*HB[i])
	
	S3  = sqrt(1 - S1*S2)	
	return S3
	

def chi2_hists(HA, HB):
	# compute the chi-squared distance
	squsum  = 0
	eps = 1e-10
	for i in range(0, len(HA)) :
		tmp1 = pow((HA[i] - HB[i]) , 2) 
		tmp2 = HA[i] + HB[i]
		if tmp2 == 0 :
			tmp2 = eps
		squsum = squsum + tmp1 / float(tmp2)
	return squsum
		
#if __name__ == '__main__':
	#usage = 'Usage: %s method correlation_matrix [threshold]' % sys.argv[0]
	#try:
		#input_empiri = sys.argv[1]
		#input_simuli = sys.argv[2]
	#except:
		#print usage
		#sys.exit(1)

local_path   = '../data/jobs_corr/'		

# simulations based on EMPIRICAL brain networks
name_E = 'A_aal_0_ADJ_thr_0.54_sigma=0.5_D=0.05_v=90.0_tmax=45000_FHN_corr.dat'
# simulations based on RANDOMIZED brain networks
name_R = 'A_aal_a_ADJ_thr_0.54_sigma=0.5_D=0.05_v=90.0_tmax=45000_FHN_corr.dat'

thr_array = np.array([ 54,  56,  58,  60,  62,  64, 66])	
vel_array = np.array([40, 50, 60, 70, 80, 90])
sig_array = np.array([0.3, 0.4, 0.5, 0.6, 0.7])

R_thr =  {}

for THR in thr_array :
	R_temp = []
	
	for VEL in vel_array :
 		input_empiri = name_E[0:18] + str(THR) + name_E[20:40] + str(VEL) + name_E[42:]		
		input_simuli = name_R[0:18] + str(THR) + name_R[20:40] + str(VEL) + name_R[42:]
		print str(THR)
		print local_path+input_empiri	
		print local_path+input_simuli

		try:
			mtx_empiri = load_matrix(local_path + input_empiri)
			HistA      = corr_histo(mtx_empiri)
			mtx_simuli = load_matrix(local_path + input_simuli)
			HistB      = corr_histo(mtx_simuli)
		except :
			R_vel      = np.nan
		else :
			R_vel      = intersec_hists(HistA, HistB)
			R_vel      = chi2_hists(HistA, HistB)
			R_vel      = bhatta_hists(HistA, HistB)
			R_vel      = correl_hists(HistA, HistB)
		print R_vel
			
		R_temp     = np.append(R_temp, R_vel)
	R_thr[THR] 	   = np.array(R_temp)	
	
Ordered_R   = collections.OrderedDict(sorted(R_thr.items()))	
#print "Ordered dict"
#print Ordered_R

datam 		= np.array(Ordered_R.values())

#print "check its numpy array version"
#print datam

## PLOTTING BEGINS ! 
fig, ax = pl.subplots()
cmap    = pl.cm.jet
##pl.imshow(np.transpose(datam), interpolation='nearest', cmap='jet', vmin=0.0, vmax=0.48, aspect='auto')
pl.imshow((np.transpose(datam)), interpolation='nearest', cmap='jet', aspect='auto')
cbar = pl.colorbar()
pl.show()
### PLOT PA OVER SIGMA
##a = thr_array
##b = sig_array
### title for fhn....
##pl.title('A_aal_0...' + ' , FHN , ' + '  v = 7 [m/s] '+' T = 450 [s]',
		 ##fontsize=20)
### title for bold...
###pl.title('A_aal_0...' + ' , BOLD , ' + '  v = 7 [m/s]', fontsize=20)
##pl.ylabel('$\sigma$ ', fontsize=20)

## PLOT PA OVER VELOCITY
#a = thr_array
#b = vel_array
## title for fhn....
##pl.title('acp_w_0_...' + ' , FHN , ' + '$\sigma$ = 0.5 '+' T = 450 [s]',
##		 fontsize=20)
#pl.title('A_aal_0...' + ' , FHN , correl. test, bins=100' + '$\sigma$ = 0.5', fontsize=20)
#pl.ylabel('v [m/s]', fontsize=20)

#pl.setp(ax , xticks=np.arange(0,len(a),1), xticklabels = a)
#pl.setp(ax , yticks=np.arange(0,len(b),1), yticklabels = b)
#pl.xlabel('thr', fontsize = 20)
#for t in cbar.ax.get_yticklabels():
	#t.set_fontsize(15)
#pl.xticks(fontsize = 15)
#pl.yticks(fontsize = 15)
#pl.show()		

#mtx_empiri			= 		load_matrix(local_path+input_empiri)		
#mtx_simuli			= 		load_matrix(local_path+input_simuli)
#pl.figure(1)
#HistA = corr_histo(mtx_empiri, input_empiri)
#pl.figure(2)
#HistB = corr_histo(mtx_simuli, input_simuli)
#print intersec_hists(HistA, HistB)

#print chi2_hists(HistA, HistB)
#print "Bahttta : ", bhatta_hists(HistA, HistB)
#print correl_hists(HistA, HistB)
#pl.show()

