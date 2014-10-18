#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

""" 
	main function : degre_pres(R, iter)
	
	input  : R , numpy (adjacency) matrix and iter, iteration number
	
	intermediate processes : loading input, chekck symmetry of input, 
	thresholding input matrix, generating a random network by preser-
	ving the degree distribution. This code is imported from BCT open
	source, original script is "randmio_und_connected.m"
	
	output : a random adjacency matrix  having the same degree 
	distribution as in the original input 
"""


import networkx as nx
import numpy as np
from math import factorial 
import math
import random as rnd
import matplotlib.pyplot as pl	
import sys  
import glob
import os


def load_matrix(file):
	A = np.loadtxt(file, unpack=True)
	AT = np.transpose(A)
	# check the symmetry				
	if A.shape[0] != A.shape[1] or not (A == AT).all():
		print "error: loaded matrix is not symmetric"
		raise ValueError
	return AT

# create adjacency matrix, ones and zeros according to r
def threshold_matrix(A, r):
	B = np.zeros(A.shape)
	for row in range(A.shape[0]):
		for col in range(A.shape[1]):
			if row != col and A[row, col] >= r:
				B[row, col] = 1
	return B

def export_adjacency_matrix(graph, input_mtx, r):  
	#print graph
	hiwi = nx.adjacency_matrix(graph)
	print ('DENEME_' + input_mtx[:-4] + '.dat' )
	f = open('DENEME_' + input_mtx[:-4] + '.dat','w')
	for i in range(len(hiwi)):
		for j in range(len(hiwi)):
			f.write("%d\t" % (hiwi[i,j]))
		f.write("\n")
	f.close()		

def plot_graph(G):
	#G   = nx.from_numpy_matrix(B)
	pos = nx.shell_layout(G)
	nx.draw(G, pos)
	#pl.show()
	
def degre_pres(B , ITER):
	
	n_col   = np.shape(B)[1]		# number of columns in array
	new_B   = np.triu(B)			# upper triangle of array	
	(j , i) = new_B.nonzero()		# (row,col) index of non-zero elem.
	
	i.setflags(write=True)
	j.setflags(write=True)
	
	K       = len(i)				# total number of non-zero elements
	ITER    = K*ITER				# total iteration number 
	
	maxAttempts = int(K/(n_col-1))  # max attempts per iteration
	
	eff     = 0  
	for iter in range(1 , ITER+1 ):
		att = 0
		while att<=maxAttempts:
			rewire = 1
			while 1:
				e1 = int(math.floor(K*rnd.random()))
				e2 = int(math.floor(K*rnd.random()))
				while e1==e2:
					e2 = int(math.floor(K*rnd.random()))
				
				a = i[e1]          # chose a col number from i
				b = j[e1]		   # chose a row number from j		
				c = i[e2]		   # chose another col number from i	
				d = j[e2]		   # chose another row number from j		
				
				
				if ( ( (a!=c) & (a!=d) ) & ( (b!=c) & (b!=d)) ) :
					break          # make sure that a,b,c,d differ
			
			


			# flipping edge c-d with 50% probability	
			if rnd.random() > 0.5 :
				i[e2]  = d
				j[e2]  = c
				c      = i[e2]
				d      = j[e2]		
			
			# rewiring condition
			if int(not(bool( B[a,d] or B[c,b] ))) : 
				
				# connectedness condition	
				if int(not(bool( B[a,c] or B[b,d] ))) :
					
					P = B[(a, d) , : ]
					P[0,b] = 0
					P[1,c] = 0
					PN     = P
					PN[:,d]= 1
					PN[:,a]= 1
			
					while 1:
						
						P[0,:] = (B[(P[0,:]!=0), :]).any(0).astype(int) 
						P[1,:] = (B[(P[1,:]!=0), :]).any(0).astype(int)
						
						P = P* (np.logical_not(PN).astype(int))
							
						if int(not((P.any(1)).all())):
							rewire = 0
							break
						
						elif  (P[:,[b, c]].any(0)).any(0):
							break
							
						PN = PN +1
				
				# reassigning edges
				if rewire :
					B[a,d] = B[a,b]
					B[a,b] = 0			
					B[d,a] = B[b,a]
					B[b,a] = 0		
					B[c,b] = B[c,d]
					B[c,d] = 0		
					B[b,c] = B[d,c]
					B[d,c] = 0
				
					# reassigning edge indices
					j[e1]  = d
					j[e2]  = b
					
					eff = eff+1;
					break
		
			att = att +1
	RG = nx.from_numpy_matrix(B)
	return RG

# taken from bctpy, https://github.com/aestrivex/bctpy
def randomize_graph_partial_und(A,B,maxswap):
    '''
A = RANDOMIZE_GRAPH_PARTIAL_UND(A,B,MAXSWAP) takes adjacency matrices A 
and B and attempts to randomize matrix A by performing MAXSWAP 
rewirings. The rewirings will avoid any spots where matrix B is 
nonzero.

Inputs:       A,      adjacency matrix to randomize
              B,      edges to avoid
        MAXSWAP,      number of rewirings

Outputs:      A,      randomized matrix

Notes:
1. Graph may become disconnected as a result of rewiring. Always
  important to check.
2. A can be weighted, though the weighted degree sequence will not be
  preserved.
3. A must be undirected.
    '''
    A=A.copy()
    i,j=np.where(np.triu(A,1))
    i.setflags(write=True); j.setflags(write=True)
    m=len(i)

    nswap=0
    while nswap < maxswap: 
        while True:
            e1,e2=np.random.randint(m,size=(2,));
            while e1==e2: e2=np.random.randint(m)
            a=i[e1]; b=j[e1]
            c=i[e2]; d=j[e2]
        
            if a!=c and a!=d and b!=c and b!=d:
                break					#all 4 vertices must be different

        if np.random.random()>.5:
            i[e2]=d; j[e2]=c			#flip edge c-d with 50% probability
            c=i[e2]; d=j[e2]			#to explore all potential rewirings
            
        #rewiring condition
        if not (A[a,d] or A[c,b] or B[a,d] or B[c,b]): #avoid specified ixes
            A[a,d]=A[a,b]; A[a,b]=0
            A[d,a]=A[b,a]; A[b,a]=0
            A[c,b]=A[c,d]; A[c,d]=0
            A[b,c]=A[d,c]; A[d,c]=0

            j[e1]=d; j[e2]=b			#reassign edge indices
            nswap+=1
    return A


def plot_graph_2(G):
	pos = nx.shell_layout(G)
	nx.draw(G, pos)
	pl.show()
	
	
	#print maxAttempts

if __name__ == '__main__':
	usage = 'Usage: %s method correlation_matrix [threshold]' % sys.argv[0]
	try:
		input_name   = sys.argv[1]
		input_name_2 = sys.argv[2]
		#input_name_3 = sys.argv[3]
	except:
		print usage
		sys.exit(1)

# if there is no thresholding needed :
A          = load_matrix(input_name)
Graph_A    = nx.from_numpy_matrix(A)
B          = load_matrix(input_name_2)
Graph_B    = nx.from_numpy_matrix(B)
A_rnd      = randomize_graph_partial_und(A, B, maxswap=100)
Graph_rnd  = nx.from_numpy_matrix(A_rnd)
#print "output ::: " 
#print A_rnd

pl.figure(2)
pl.subplot(1,3,1)
plot_graph(Graph_A)
pl.subplot(1,3,2)
plot_graph(Graph_B)
pl.subplot(1,3,3)
plot_graph(Graph_rnd)
pl.show()

## if everything is IMPORTED HERE :
#data_matrix = load_matrix(input_name)
#data_matrix_2 = load_matrix(input_name_2)
#data_matrix_3 = load_matrix(input_name_3)
#pl.figure(3)
#pl.subplot(1,3,1)
#plot_graph(nx.from_numpy_matrix(data_matrix))
#pl.title('acp_w_0.48')
#pl.subplot(1,3,2)
#plot_graph(nx.from_numpy_matrix(data_matrix_2))
#pl.title('A_aal_0.48')
#pl.subplot(1,3,3)
#plot_graph(nx.from_numpy_matrix(data_matrix_3))
#pl.title('swap=150')
#pl.show()

