#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# sb_thresholding.py : creates a thresholded matrix and saves it
# prints simple characteristics of the thresholded matrix

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

  G=nx.from_numpy_matrix(B,create_using=nx.Graph())  #result of function is graph G
  return G


def print_adjacency_matrix(G):			# use previous G here, print it
  print nx.adjacency_matrix(G)


def export_adjacency_matrix(G, input_mtx, r):		# save adjacency matrix

  hiwi = nx.adjacency_matrix(G)
  f = open(input_mtx[:-4]+'_r'+str(r)+'.dat','w')
  for i in range(len(hiwi)):
    for j in range(len(hiwi)):
      f.write("%d\t" % (hiwi[i,j]))
    f.write("\n")
  f.close()

def characteristics(G, input_mtx,r):
  print 'calculating characteristics with threshold=',r  
  num_nodes = nx.number_of_nodes(G)
  num_edges = nx.number_of_edges(G)
  num_components = nx.number_connected_components(G)
  print 'number of nodes in Graph: ', num_nodes
  print 'number of edges in Graph: ', num_edges
  print 'number of components in Graph: ', num_components
  print ' '

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
  print_adjacency_matrix(network)
  export_adjacency_matrix(network, infilename_data, threshold)
  characteristics(network, infilename_data, threshold)

