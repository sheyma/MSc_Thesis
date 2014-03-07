#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# Creating a random network by preserving the degree distribution of test network (e.g. A.txt)


import networkx as nx
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
	G=nx.from_numpy_matrix(B,create_using=nx.Graph())  # create graph of thresolded matr.
  
	degree_hist = {}
	for node in G:
		
		if G.degree(node) not in degree_hist: # degree dist part
			degree_hist[G.degree(node)] =1
		else:
			degree_hist[G.degree(node)] +=1
	print degree_hist
	keys = degree_hist.keys()
	print keys
	Random_Gc = nx.configuration_model([2,3,4])
	return keys

if __name__ == '__main__':
  
  usage = 'Usage: %s correlation_matrix threshold' % sys.argv[0]
  try:
    input_matrix = sys.argv[1]
    input_threshold = sys.argv[2]
  except:
    print usage; sys.exit(1)
   
random_graph_c(input_matrix,float(input_threshold))

