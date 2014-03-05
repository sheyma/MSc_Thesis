#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

def get_threshold_matrix(filename, threshold_value):
	import networkx as nx
	import numpy as np
	
	A = np.transpose(np.loadtxt(filename, unpack=True))

	B = np.zeros(len(A),len(A))
	
	for row in range(len(A)):
		for item in range(len(A)):
			if row != item:
				if A[row,item] >= threshold_value:
					B[row,item] = 1
				else:
					B[row,item] = 0 
	
	G=nx.from_numpy_matrix(B,create_using=nx.Graph())
	return G

def print_adjacency_matrix(G):
	import networkx as nx
	print nx.adjacency_matrix(G)


def export_adjacency_matrix(G, filename, threshold_value):
	hiwi = nx.adjacency_matrix(G)
  
	f = open(filename[:-4]+'_r'+str(threshold_value)+'.dat','w')
  	for i in range(len(hiwi)):
    	for j in range(len(hiwi)):
    	  f.write("%d\t" % (hiwi[i,j]))
    	f.write("\n")
  	f.close()



	import sys
	value = sys.argv[2]
	infilename_data = sys.argv[1]

	threshold = float(value)
	network = get_threshold_matrix(infilename_data, threshold)
	print_adjacency_matrix(network)
	export_adjacency_matrix(network, infilename_data, threshold)
