#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# Creating a random network with same node and link numbers of test network 
# use networkx.gnm_random_graph(N,L)


import networkx as nx
import numpy as np 
import sys  


# 1. create a random network with method a

def random_graph_a(matrix, r):
  A = np.transpose(np.loadtxt(matrix, unpack=True)) 
  B = np.zeros((len(A),len(A)))

  for row in range(len(A)):
    for item in range(len(A)):
      if row != item:
        if A[row,item] >= r:
          B[row,item] = 1
        else:
          B[row,item] = 0
  #print B	   										 # print thresholded new matrix  
  G=nx.from_numpy_matrix(B,create_using=nx.Graph())  # create graph of thresolded matr.
  L = nx.number_of_edges(G) 						 # total number of links: L  
  N = nx.number_of_nodes(G) 						 # total number of nodes : N
  #print 'number of links (edges) in G: ' , L
  #print 'number of nodes in G : ' , N	
  Random_Ga = nx.gnm_random_graph(N,L)				 # random graph
  return Random_Ga


def measures_random_Ga(matrix):
  R = 0
  f=open(matrix[:-4]+'_Random_Ga.dat','w')
  f.write('r(thres)\tL\tN\tD(dens.)\tcon_comp\tCC(clus.)\tcheck_sum\tave_degr\n')
  for i in range (0,101):
	R = float(i)/100
	Random_Ga = random_graph_a(matrix,R)
	L = nx.number_of_edges(Random_Ga)  #total number of links : L
	N = nx.number_of_nodes(Random_Ga)  #total number of nodes : N
	max_edge = N*(N-1.)/2     		   # number of maximum edges(links)
	Compon = nx.number_connected_components(Random_Ga)
	CC = nx.average_clustering(Random_Ga)  # clustering coefficient of full network	
	check_sum = 0.     		  # sum of degree distributions for one r
	degree_hist = {}
	values = []	
	for node in Random_Ga:
		
		if Random_Ga.degree(node) not in degree_hist: # degree dist part
			degree_hist[Random_Ga.degree(node)] =1
		else:
			degree_hist[Random_Ga.degree(node)] +=1

		values.append(Random_Ga.degree(node))	# average degree part
	
	ave=float(sum(values))/(nx.number_of_nodes(Random_Ga))	# average degree overall network
	
	keys = degree_hist.keys()
	keys.sort
	for item in keys:		
		check_sum +=float(degree_hist[item])/float(N)
	
	f.write("%f\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t\n" %(R,L,N,L/max_edge,Compon,CC,check_sum,ave))	
  f.close()
  #L/max_edge : network density, D	



if __name__ == '__main__':
  
  usage = 'Usage: %s correlation_matrix threshold' % sys.argv[0]
  try:
    input_matrix = sys.argv[1]
    #input_threshold = sys.argv[2]
  except:
    print usage; sys.exit(1)

  ###manual choice of the threshold value
  #R = float(input_threshold)
  #random_graph_a(input_matrix, R)
  measures_random_Ga(input_matrix)
  #print_adjacency_matrix(network)
