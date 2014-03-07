#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# Creating a random network by preserving the degree of test network (e.g. A.txt)


import networkx as nx
import numpy as np 
import sys  


# 1. create a random network with method b

def random_graph_b(matrix, r):
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
  Random_Gb = nx.double_edge_swap(G,nswap=1,max_tries=100000000)				 # random graph
  return Random_Gb


def measures_random_Gb(matrix):
  R = 0
  f=open(matrix[:-4]+'_Random_Gb.dat','w')
  f.write('r(thres)\tL\tN\tD(dens.)\tcon_comp\tCC(clus.)\tcheck_sum\tave_degr\n')
  for i in range (0,101):
	R = float(i)/100
	Random_Gb = random_graph_b(matrix,R)
	L = nx.number_of_edges(Random_Gb)  #total number of links : L
	N = nx.number_of_nodes(Random_Gb)  #total number of nodes : N
	max_edge = N*(N-1.)/2     		   # number of maximum edges(links)
	Compon = nx.number_connected_components(Random_Gb)
	CC = nx.average_clustering(Random_Gb)  # clustering coefficient of full network	
	check_sum = 0.     		  # sum of degree distributions for one r
	degree_hist = {}
	values = []	
	for node in Random_Gb:
		
		if Random_Gb.degree(node) not in degree_hist: # degree dist part
			degree_hist[Random_Gb.degree(node)] =1
		else:
			degree_hist[Random_Gb.degree(node)] +=1

		values.append(Random_Gb.degree(node))	# average degree part
	
	ave=float(sum(values))/(nx.number_of_nodes(Random_Gb))	# average degree overall network
	
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
  measures_random_Gb(input_matrix)
  #print_adjacency_matrix(network)
