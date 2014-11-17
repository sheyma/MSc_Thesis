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
#name_E = 'A_aal_0_ADJ_thr_0.54_sigma=0.3_D=0.05_v=30.0_tmax=45000_FHN_corr.dat'
name_E = 'acp_w_0_ADJ_thr_0.54_sigma=0.5_D=0.05_v=30.0_tmax=45000_FHN_corr.dat'
# simulations based on RANDOMIZED brain networks
#name_R = 'A_aal_g_ADJ_thr_0.54_sigma=0.3_D=0.05_v=30.0_tmax=45000_FHN_corr.dat'
name_R = 'acp_w_a_ADJ_thr_0.54_sigma=0.5_D=0.05_v=30.0_tmax=45000_FHN_corr.dat'

#thr_array = np.array([ 54,  56,  58,  60,  62,  64, 66])	
#vel_array = np.array([40, 50, 60, 70, 80, 90])
#sig_array = np.array([0.3, 0.4, 0.5, 0.6, 0.7])

thr_array = np.array([22,  26,  30,   34, 
						 38,  42, 44, 46, 48, 50, 52, 54,
						56, 58,  60,  62,  64,  66, 
						68,  70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
						80, 81, 82, 83])
vel_array = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
sig_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


x = []
y = []
z = []

for THR in thr_array :
	for VEL in vel_array :
		input_empiri = name_E[0:18] + str(THR) + name_E[20:40] + str(VEL) + name_E[42:]		
		input_simuli = name_R[0:18] + str(THR) + name_R[20:40] + str(VEL) + name_R[42:]

	#for SIG in sig_array :
		#input_empiri = name_E[0:18] + str(THR) + name_E[20:27] + str(SIG) + name_E[30:]
		#input_simuli = name_R[0:18] + str(THR) + name_R[20:27] + str(SIG) + name_R[30:]
		
		try:
			mtx_empiri = load_matrix(local_path + input_empiri)
			HistA      = corr_histo(mtx_empiri)
			mtx_simuli = load_matrix(local_path + input_simuli)
			HistB      = corr_histo(mtx_simuli)
		except :
			R_val      = np.nan
		else :
			#R_val      = intersec_hists(HistA, HistB)
			#R_val      = chi2_hists(HistA, HistB)
			R_val       = bhatta_hists(HistA, HistB)
			#R_val      = correl_hists(HistA, HistB)
		
		x = np.append(x, THR)
		y = np.append(y, VEL)
		#y = np.append(y, SIG)
		z = np.append(z, R_val)

xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
zi = griddata( (x,y), z, (xi[None,:], yi[:,None]), method='cubic') 
zi = np.nan_to_num(zi)
xig, yig = np.meshgrid(xi, yi)
fig = pl.figure(2)
ax = fig.add_subplot(1,1,1, projection='3d', azim=210)
surf = ax.plot_surface(xig, yig, zi, rstride=1, cstride=1, cmap=cm.jet ) 
fig.colorbar(surf, ax=ax)
pl.xticks(thr_array)
pl.yticks(vel_array)
pl.autoscale(tight=True)

ax.set_xlabel('r', fontsize=25)
ax.set_ylabel('v', fontsize=25)
ax.set_zlabel('Bhatta', fontsize=25)
#ax.text2D(0.05, 0.95, 'Bhatta comparison (c=0.3) : 0 - g : A_aal', transform=ax.transAxes)
ax.text2D(0.05, 0.95, 'Bhatta comparison (c=0.3) : 0 - a : acp_w', transform=ax.transAxes)

pl.show()


R_thr =  {}

x = []
y = []
z = []
t = []

#for THR in thr_array :
	#R_temp = []
	
	#for SIG in sig_array :		
		#input_empiri = name_E[0:18] + str(THR) + name_E[20:27] + str(SIG) + name_E[30:]
		#input_simuli = name_R[0:18] + str(THR) + name_R[20:27] + str(SIG) + name_R[30:]
		
		#for VEL in vel_array :
			#input_empiri = name_E[0:18] + str(THR) + name_E[20:40] + str(VEL) + name_E[42:]		
			#input_simuli = name_R[0:18] + str(THR) + name_R[20:40] + str(VEL) + name_R[42:]
			
			#try:
				#mtx_empiri = load_matrix(local_path + input_empiri)
				#HistA      = corr_histo(mtx_empiri)
				#mtx_simuli = load_matrix(local_path + input_simuli)
				#HistB      = corr_histo(mtx_simuli)
			#except :
				#R_val      = np.nan
			#else :
				##R_val      = intersec_hists(HistA, HistB)
				##R_val      = chi2_hists(HistA, HistB)
				#R_val       = bhatta_hists(HistA, HistB)
				##R_val      = correl_hists(HistA, HistB)
			
			#x = np.append(x, THR)
			#y = np.append(y, SIG)
			#z = np.append(z, VEL)
			#t = np.append(t, R_val)
##np.savetxt('temp.dat', t,'%.6f',delimiter='\t')
#datam = []
#datam.append([x, y, z, t])			
#datam = zip(*datam)

#fig = pl.figure(1)
#ax  = fig.add_subplot(111,projection='3d')
#p   = ax.scatter3D(datam[0], datam[1], datam[2], c=datam[3], cmap='jet')
#fig.colorbar(p, ax=ax)

#pl.xticks(thr_array)
#pl.yticks(sig_array)
#pl.autoscale(tight=True)

#ax.set_xlabel('r', fontsize=25)
#ax.set_ylabel('c', fontsize=25)
#ax.set_zlabel('v', fontsize=25)
#ax.text2D(0.05, 0.95, 'Bhatta comparison : 0 - a : A_aal', transform=ax.transAxes)
#pl.show()

data1=np.array(x)
data2=np.array(y)
data3=np.array(z)
output=np.array(t)

#fig = pl.figure(2)

#xi = np.linspace(data1.min(),data1.max(),200)
#yi = np.linspace(data2.min(),data2.max(),200)
#wi = np.linspace(data3.min(),data3.max(),200)

## Interpoling unstructured data 
#zi = griddata((data1, data2), output, (xi[None,:], yi[:,None]), method='cubic')
## removing NaNs from the array
#zi = np.nan_to_num(zi)

#ax = fig.add_subplot(1, 1, 1, projection='3d', azim=210)

#xig, yig = np.meshgrid(xi, yi)

##normalizing variable to interval 0-1
#data3col=data3/data3.max()

#surf = ax.plot_surface(xig, yig, zi, rstride=1, cstride=1, facecolor=cm.jet(data3col), cmap='jet', linewidth=0, antialiased=False, shade=False)
##surf = ax.plot_surface(xig, yig, zi, rstride=1, cstride=1, facecolor=cm.jet(data3col), linewidth=0, antialiased=False, shade=False)
#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')
##fig.colorbar(surf, shrink=0.5, aspect=5)
##ax.set_zlim(data3.min(), data3.max())
#fig.colorbar(surf, ax=ax)
#pl.show()

#x = (np.array([x])).T
#y = (np.array([y])).T
#z = (np.array([z])).T
#F = np.array(t)

#points = np.concatenate( (x,y,z), axis=1)

#grid_x, grid_y, grid_z = np.mgrid[54:66:100j, 0.3:0.7:100j, 40:90:100j   ]

#grid_F = griddata(points, F, (grid_x, grid_y, grid_z), method='linear')

#fig = pl.figure(5)
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(grid_x, grid_y, rstride=1, cstride=1, cmap=cm.coolwarm,
        #linewidth=0, antialiased=False)
