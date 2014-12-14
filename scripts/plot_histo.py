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
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata	
from matplotlib import cm

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
	
	#hist = pl.hist(corr_flat, bins=bin_nu, range=[corr_min, corr_max], normed =True, histtype='bar')
	#pl.title(simfile)
	# type(hist) = <type 'tuple'> and len(hist) = 3
	# y_axis : hist[0] , normalized hist values
	# x_axis : hist[1] , start points of bins 
	return hist

def plot_histog(corr_matrix, STRING):
	corr_flat = np.ndarray.flatten(corr_matrix) 
	corr_max  = float(1.0)
	corr_min  = float(-1.0)
	bin_nu    = 100
	# a normalized histogram is obtained
	pl.hist(corr_flat, bins=bin_nu, range=[corr_min, corr_max], normed =True, histtype='bar', align='mid')
	pl.xlim(corr_min-0.01, corr_max+0.01)
	pl.ylim(0.0, 17.0)	
	pl.xticks(fontsize=20)
	pl.yticks(fontsize=20)
	pl.text(-0.5, 15.0, STRING,
				horizontalalignment='center',
				verticalalignment='center', fontsize=30)

	#pl.xlabel('$\\rho$', fontsize=25)

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
		


local_path   = '../data/jobs_corr/'		

O = load_matrix(local_path + 'A_aal_0_ADJ_thr_0.64_sigma=0.018_D=0.05_v=70.0_tmax=45000_FHN_corr.dat')
A = load_matrix(local_path + 'A_aal_a_ADJ_thr_0.64_sigma=0.018_D=0.05_v=70.0_tmax=45000_FHN_corr.dat')
#EMP = load_matrix('A_aal.txt')
#D = load_matrix(local_path + 'A_aal_d_ADJ_thr_0.61_sigma=0.1_D=0.05_v=70.0_tmax=45000_FHN_corr.dat')
#G = load_matrix(local_path + 'A_aal_g_ADJ_thr_0.61_sigma=0.1_D=0.05_v=70.0_tmax=45000_FHN_corr.dat')
#H = load_matrix(local_path + 'A_aal_h_ADJ_thr_0.61_sigma=0.1_D=0.05_v=70.0_tmax=45000_FHN_corr.dat')
#K = load_matrix(local_path + 'A_aal_k_ADJ_thr_0.61_sigma=0.1_D=0.05_v=70.0_tmax=45000_FHN_corr.dat')

#O = load_matrix(local_path + 'acp_w_0_ADJ_thr_0.58_sigma=0.02_D=0.05_v=60.0_tmax=45000_FHN_corr.dat')
#A = load_matrix(local_path + 'acp_w_a_ADJ_thr_0.58_sigma=0.02_D=0.05_v=60.0_tmax=45000_FHN_corr.dat')
#D = load_matrix(local_path + 'acp_w_d_ADJ_thr_0.58_sigma=0.02_D=0.05_v=60.0_tmax=45000_FHN_corr.dat')
#G = load_matrix(local_path + 'acp_w_g_ADJ_thr_0.58_sigma=0.02_D=0.05_v=60.0_tmax=45000_FHN_corr.dat')
#H = load_matrix(local_path + 'acp_w_h_ADJ_thr_0.58_sigma=0.02_D=0.05_v=60.0_tmax=45000_FHN_corr.dat')
#K = load_matrix(local_path + 'acp_w_k_ADJ_thr_0.58_sigma=0.02_D=0.05_v=60.0_tmax=45000_FHN_corr.dat')

w0 = load_matrix(local_path + 'acp_w_0_ADJ_thr_0.42_sigma=0.5_D=0.05_v=60.0_tmax=45000_FHN_corr.dat')

#fig , axes = pl.subplots(5,2,sharex=True,sharey=True,figsize=(12,15))
fig, ax = pl.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(14,7))
#pl.subplots_adjust(left=0.11, right=0.95, top=0.98, bottom=0.06)
#fig.text(0.5, 0.02, '$\\rho$', ha='center', fontsize=30 )
fig.text(0.04, 0.5, 'number of pair of nodes', va='center', rotation='vertical',fontsize=25)

pl.subplot(1,2,1)
plot_histog(O, '$R_{BG}$')
pl.xlabel('$\\rho$', fontsize=30 )
pl.subplot(1,2,2)
plot_histog(A, '$R_{ER}$')
pl.xlabel('$\\rho$', fontsize=30 )
#plot_histog(EMP, '$emp$')

#pl.subplot(3,2,3)
#plot_histog(D, '$R_{DES}$')
#pl.subplot(3,2,4)
#plot_histog(G, '$R_{CM}$')
#pl.subplot(3,2,5)
#plot_histog(H, '$R_{PDD}$')
#pl.subplot(3,2,6)
#plot_histog(K, '$R_{PR}$')

#fig, ax = pl.subplots(figsize=(12,6))
#pl.subplot(1,2,1)
#pl.xlabel('$\\rho$', fontsize = 30)
#pl.ylabel('number of pair of nodes', fontsize = 25)
#plot_histog(O, '$R_{BG}$')
#pl.subplot(1,2,2)
#pl.xlabel('$\\rho$', fontsize = 30)
##pl.ylabel('number of pair of nodes', fontsize = 25)
#plot_histog(w0, '$R_{BG}$')

pl.show()

### simulations based on EMPIRICAL brain networks
name_E = 'A_aal_0_ADJ_thr_0.54_sigma=0.3_D=0.05_v=70.0_tmax=45000_FHN_corr.dat'
###name_E = 'acp_w_0_ADJ_thr_0.54_sigma=0.5_D=0.05_v=30.0_tmax=45000_FHN_corr.dat'
### simulations based on RANDOMIZED brain networks
name_Ra = 'A_aal_a_ADJ_thr_0.54_sigma=0.3_D=0.05_v=70.0_tmax=45000_FHN_corr.dat'
name_Rd = 'A_aal_d_ADJ_thr_0.54_sigma=0.3_D=0.05_v=70.0_tmax=45000_FHN_corr.dat'
name_Rg = 'A_aal_g_ADJ_thr_0.54_sigma=0.3_D=0.05_v=70.0_tmax=45000_FHN_corr.dat'
name_Rh = 'A_aal_h_ADJ_thr_0.54_sigma=0.3_D=0.05_v=70.0_tmax=45000_FHN_corr.dat'
name_Rk = 'A_aal_k_ADJ_thr_0.54_sigma=0.3_D=0.05_v=70.0_tmax=45000_FHN_corr.dat'

###name_R = 'A_aal_a_ADJ_thr_0.54_sigma=0.3_D=0.05_v=70.0_tmax=45000_FHN_corr.dat'
###name_R = 'acp_w_a_ADJ_thr_0.54_sigma=0.5_D=0.05_v=30.0_tmax=45000_FHN_corr.dat'

thr_array = np.array([54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66])	
sig_array = np.array([0.018, 0.02, 0.025, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
sig_array = np.append(sig_array[-1:0:-1] , sig_array[0])
print sig_array

#name_E  = 'acp_w_0_ADJ_thr_0.54_sigma=0.3_D=0.05_v=60.0_tmax=45000_FHN_corr.dat'
#name_Ra = 'acp_w_a_ADJ_thr_0.54_sigma=0.3_D=0.05_v=60.0_tmax=45000_FHN_corr.dat'
#name_Rd = 'acp_w_d_ADJ_thr_0.54_sigma=0.3_D=0.05_v=60.0_tmax=45000_FHN_corr.dat'
#name_Rg = 'acp_w_g_ADJ_thr_0.54_sigma=0.3_D=0.05_v=60.0_tmax=45000_FHN_corr.dat'
#name_Rh = 'acp_w_h_ADJ_thr_0.54_sigma=0.3_D=0.05_v=60.0_tmax=45000_FHN_corr.dat'
#name_Rk = 'acp_w_k_ADJ_thr_0.54_sigma=0.3_D=0.05_v=60.0_tmax=45000_FHN_corr.dat'


#thr_array = np.arange(38, 86, 4)
##vel_array = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
#sig_array = np.array([0.8,   0.5,  0.2, 0.1, 0.05, 0.02])


R_thr_a = {}
R_thr_d = {}
R_thr_g = {}
R_thr_h = {}
R_thr_k = {}

for THR in thr_array :
	R_temp_a = []
	R_temp_d = []
	R_temp_g = []
	R_temp_h = []
	R_temp_k = []

	#for VEL in vel_array :
		#input_empiri = name_E[0:18] + str(THR) + name_E[20:40] + str(VEL) + name_E[42:]		
		#input_simuli = name_Ra[0:18] + str(THR) + name_Ra[20:40] + str(VEL) + name_Ra[42:]

	for SIG in sig_array :
		input_empiri = name_E[0:18] + str(THR) + name_E[20:27] + str(SIG) + name_E[30:]
		input_simuli_a = name_Ra[0:18] + str(THR) + name_Ra[20:27] + str(SIG) + name_Ra[30:]
		input_simuli_d = name_Rd[0:18] + str(THR) + name_Rd[20:27] + str(SIG) + name_Rd[30:]
		input_simuli_g = name_Rg[0:18] + str(THR) + name_Rg[20:27] + str(SIG) + name_Rg[30:]
		input_simuli_h = name_Rh[0:18] + str(THR) + name_Rh[20:27] + str(SIG) + name_Rh[30:]
		input_simuli_k = name_Rk[0:18] + str(THR) + name_Rk[20:27] + str(SIG) + name_Rk[30:]
		
		#print input_empiri
		
		mtx_empiri = load_matrix(local_path + input_empiri)
		HistA      = corr_histo(mtx_empiri)
		
		try:	
			mtx_simuli_a = load_matrix(local_path + input_simuli_a)
			HistBa      = corr_histo(mtx_simuli_a)			
		except :
			R_val_a      = np.nan
		else :
			R_val_a       = bhatta_hists(HistA, HistBa)
		R_temp_a     = np.append(R_temp_a, R_val_a)	
		
		try:			
			mtx_simuli_d = load_matrix(local_path + input_simuli_d)
			HistBd      = corr_histo(mtx_simuli_d)
		except :
			R_val_d      = np.nan
		else :
			R_val_d       = bhatta_hists(HistA, HistBd)
		R_temp_d     = np.append(R_temp_d, R_val_d)	

		try:					
			mtx_simuli_g = load_matrix(local_path + input_simuli_g)
			HistBg      = corr_histo(mtx_simuli_g)
		except :
			R_val_g      = np.nan
		else :
			R_val_g       = bhatta_hists(HistA, HistBg)
		R_temp_g     = np.append(R_temp_g, R_val_g)
		
		try:					
			mtx_simuli_h = load_matrix(local_path + input_simuli_h)
			HistBh      = corr_histo(mtx_simuli_h)
		except :
			R_val_h      = np.nan
		else :
			R_val_h       = bhatta_hists(HistA, HistBh)
		R_temp_h     = np.append(R_temp_h, R_val_h)
		
		try:					
			mtx_simuli_k = load_matrix(local_path + input_simuli_k)
			HistBk      = corr_histo(mtx_simuli_k)
		except :
			R_val_k      = np.nan
		else :
			R_val_k       = bhatta_hists(HistA, HistBk)
		R_temp_k     = np.append(R_temp_k, R_val_k)
		
	R_thr_a[THR] 	   = np.array(R_temp_a)
	R_thr_d[THR] 	   = np.array(R_temp_d)
	R_thr_g[THR] 	   = np.array(R_temp_g)
	R_thr_h[THR] 	   = np.array(R_temp_h)
	R_thr_k[THR] 	   = np.array(R_temp_k)
	
Ordered_R_a   = collections.OrderedDict(sorted(R_thr_a.items()))	
Ordered_R_d   = collections.OrderedDict(sorted(R_thr_d.items()))
Ordered_R_g   = collections.OrderedDict(sorted(R_thr_g.items()))
Ordered_R_h   = collections.OrderedDict(sorted(R_thr_h.items()))
Ordered_R_k   = collections.OrderedDict(sorted(R_thr_k.items()))
#print "Ordered dict"
#print Ordered_R

datam_a 		= np.array(Ordered_R_a.values())
datam_d 		= np.array(Ordered_R_d.values())
datam_g 		= np.array(Ordered_R_g.values())
datam_h 		= np.array(Ordered_R_h.values())
datam_k 		= np.array(Ordered_R_k.values())

#print "check its numpy array version"
#print datam

# PLOTTING BEGINS ! 

thr_array = np.arange(38, 86, 4)
#a = np.array([0.42,  0.50, 0.58, 0.66, 0.74, 0.82])	
a = np.array([0.55,  0.58, 0.61, 0.64])	
b = sig_array

fig , ax = pl.subplots(figsize=(18, 20))
pl.subplots_adjust(left=0.11, right=0.95, top=0.98, bottom=0.06)


pl.subplot(1,1,1)
pl.imshow(np.transpose(datam_a), interpolation='nearest', vmin= 0.05, vmax = 0.8, cmap='jet', aspect='auto')
#pl.imshow(np.transpose(datam_a), interpolation='nearest', vmin= 0.05, vmax = 0.85, cmap='jet', aspect='auto')
cbar = pl.colorbar()

pl.ylabel('$c$', fontsize=50)
pl.xlabel('r', fontsize = 50)

#pl.xticks(np.arange(1,len(thr_array),2),  a)
pl.xticks(np.arange(1,len(thr_array),3),  a)
pl.yticks(np.arange(0,len(b),1),  b)

for t in cbar.ax.get_yticklabels():
	t.set_fontsize(40)
pl.xticks(fontsize = 40)
pl.yticks(fontsize = 40)

#pl.subplot(3,2,2)
#pl.imshow(np.transpose(datam_d), interpolation='nearest', vmin= 0.05, vmax = 1.0, cmap='jet', aspect='auto')
##pl.imshow(np.transpose(datam_d), interpolation='nearest', vmin= 0.05, vmax = 0.85, cmap='jet', aspect='auto')
#cbar = pl.colorbar()

#pl.ylabel('$c$', fontsize=30)
#pl.xlabel('p', fontsize = 25)

#pl.xticks(np.arange(1,len(thr_array),2),  a)
##pl.xticks(np.arange(1,len(thr_array),3),  a)
#pl.yticks(np.arange(0,len(b),1),  b)

#for t in cbar.ax.get_yticklabels():
	#t.set_fontsize(25)
#pl.xticks(fontsize = 25)
#pl.yticks(fontsize = 20)

#pl.subplot(3,2,3)
#pl.imshow(np.transpose(datam_g), interpolation='nearest', vmin= 0.05, vmax = 1.0, cmap='jet', aspect='auto')
##pl.imshow(np.transpose(datam_g), interpolation='nearest', vmin= 0.05, vmax = 0.85, cmap='jet', aspect='auto')
#cbar = pl.colorbar()

#pl.ylabel('$c$', fontsize=30)
#pl.xlabel('p', fontsize = 25)

#pl.xticks(np.arange(1,len(thr_array),2),  a)
##pl.xticks(np.arange(1,len(thr_array),3),  a)
#pl.yticks(np.arange(0,len(b),1),  b)

#for t in cbar.ax.get_yticklabels():
	#t.set_fontsize(25)
#pl.xticks(fontsize = 25)
#pl.yticks(fontsize = 20)

#pl.subplot(3,2,4)
#pl.imshow(np.transpose(datam_h), interpolation='nearest', vmin= 0.05, vmax = 1.0, cmap='jet', aspect='auto')
##pl.imshow(np.transpose(datam_h), interpolation='nearest', vmin= 0.05, vmax = 0.85, cmap='jet', aspect='auto')
#cbar = pl.colorbar()

#pl.ylabel('$c$', fontsize=30)
#pl.xlabel('p', fontsize = 25)

#pl.xticks(np.arange(1,len(thr_array),2),  a)
##pl.xticks(np.arange(1,len(thr_array),3),  a)
#pl.yticks(np.arange(0,len(b),1),  b)

#for t in cbar.ax.get_yticklabels():
	#t.set_fontsize(25)
#pl.xticks(fontsize = 25)
#pl.yticks(fontsize = 20)

#pl.subplot(3,2,5)
#pl.imshow(np.transpose(datam_k), interpolation='nearest', vmin= 0.05, vmax = 1.0, cmap='jet', aspect='auto')
##pl.imshow(np.transpose(datam_k), interpolation='nearest', vmin= 0.05, vmax = 0.85, cmap='jet', aspect='auto')
#cbar = pl.colorbar()

#pl.ylabel('$c$', fontsize=30)
#pl.xlabel('p', fontsize = 25)

#pl.xticks(np.arange(1,len(thr_array),2),  a)
##pl.xticks(np.arange(1,len(thr_array),3),  a)
#pl.yticks(np.arange(0,len(b),1),  b)

#for t in cbar.ax.get_yticklabels():
	#t.set_fontsize(25)
#pl.xticks(fontsize = 25)
#pl.yticks(fontsize = 20)
	
pl.show()

