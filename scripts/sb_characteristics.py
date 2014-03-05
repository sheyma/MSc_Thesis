#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# sb_thresholding.py : creates a thresholded matrix and saves it

import networkx as nx
import numpy as np 
import sys  

def get_threshold_matrix(input_mtx, r):
  A = np.transpose(np.loadtxt(input_mtx, unpack=True)) 
  B = np.zeros((len(A),len(A)))

  for row in range(len(A)):
    for item in range(len(A)):
      if row != item:
        if A[row,item] >= r:
          B[row,item] = 1
        else:
          B[row,item] = 0

  G=nx.from_numpy_matrix(B,create_using=nx.Graph()) 
  return G
   #result of function is graph G is to be used in all other functions

def characteristics(G, input_mtx):
  print 'calculating characteristics'
  
  num_nodes = nx.number_of_nodes(G)
  num_edges = nx.number_of_edges(G)
  num_components = nx.number_connected_components(G)
  print 'number of nodes in Graph: ', num_nodes
  print 'number of edges in Graph: ', num_edges
  print 'number of components in Graph: ', num_components

def num_edges_r(input_mtx):
  threshold = 0
  f=open(input_mtx[:-4]+'_n_edges.dat','w')
  for i in range (0,101):
	threshold = float(i)/100
	G = get_threshold_matrix(input_mtx,threshold)
	print 'number of edges: ', nx.number_of_edges(G)
	max_n_egdes = nx.number_of_nodes(G)*(nx.number_of_nodes(G)-1.)/2
	f.write("%f\t%d\t%f\n" % threshold, nx.number_of_edges(G), nx.number_of_edges(G)/max_n_egdes)
   
  f.close()


if __name__ == '__main__':
  usage = 'Usage: %s correlation_matrix threshold' % sys.argv[0]
  try:
    infilename_data = sys.argv[1]
    value = sys.argv[2]
  except:
    print usage; sys.exit(1)

  ###manual choice of the threshold value
  threshold = float(value)
  network = get_threshold_matrix(infilename_data, threshold)
  characteristics(network,infilename_data)
  num_edges_r(infilename_data)

