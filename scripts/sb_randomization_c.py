#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# Creating a random network by preserving the degree distribution of test network (e.g. A.txt)
# use random graph generator : configuration model

import networkx as nx
from networkx.algorithms import bipartite
import numpy as np 
import sys  


# 1. create a random network with method c

def random_graph_c(matrix, r):
	A = np.transpose(np.loadtxt(matrix, unpack=True)) 
	B = np.zeros((len(A),len(A)))

	for row in range(len(A)):
		for item in range(len(A)):
		  if row != item:
			if A[row,item] >= r:
			  B[row,item] = 1
			else:
			  B[row,item] = 0
		#print B	   								   # print thresholded new matrix  
	G=nx.from_numpy_matrix(B,create_using=nx.Graph())  # create graph of thresolded A
  	
	degree_hist = {}
	for node in G:
		
		if G.degree(node) not in degree_hist: # degree dist part
			degree_hist[G.degree(node)] =1
		else:
			degree_hist[G.degree(node)] +=1
	keys = degree_hist.keys()
	degrees = range(0,nx.number_of_nodes(G)+1,1)
	degree_seq = []
	for item in degrees:
		if item in keys:
			degree_seq.append(degree_hist[item])		# degree sequence of nodes	
	Random_Gc = nx.configuration_model(degree_seq)		# random graph generator
	return Random_Gc

def measures_random_Gc(matrix):
	R = 0
	f = open(matrix[:-4]+'_Random_Gc.dat', 'w')
	f.write('r(thres)\tL\tN\tD(dens.)\tcon_comp\tCC(clus.)\tcheck_sum\tave_degr\n')
	for i in range (0,101):
		R = float(i)/100
		Random_Gc = random_graph_c(matrix,R)
		N = nx.number_of_nodes(Random_Gc)
		L = nx.number_of_edges(Random_Gc)
		max_edge = N*(N-1.)/2
		Compon = nx.number_connected_components(Random_Gc)
		
		check_sum = 0.
		degree_hist = {}
		values = []

		for node in Random_Gc:
			if Random_Gc.degree(node) not in degree_hist:
				degree_hist[Random_Gc.degree(node)] = 1
			else:
				degree_hist[Random_Gc.degree(node)] +=1

		values.append(Random_Gc.degree(node))
		ave = float(sum(values))/(nx.number_of_nodes(Random_Gc))

		keys = degree_hist.keys()
		keys.sort()
		for item in keys:
			check_sum +=float(degree_hist[item])/float(N)
		
		#CC = nx.average_clustering(Random_Gc)
		f.write("%f\t%d\t%d\t%f\t%f\t\t%f\t%f\t\n" %(R,L,N,L/max_edge,Compon,check_sum,ave))
	f.close()
	


if __name__ == '__main__':
  
  usage = 'Usage: %s correlation_matrix threshold' % sys.argv[0]
  try:
    input_matrix = sys.argv[1]
    #input_threshold = sys.argv[2]
  except:
    print usage; sys.exit(1)
   
measures_random_Gc(input_matrix)

