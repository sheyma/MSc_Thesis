#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# sb_thresholding.py : creates a thresholded matrix and Graph G
# in Graph G for many threshold values:
# finds number of edges(links: L)
# finds number of nodes, N
# finds network density D = 2L/(N(N-1))
# finds number of connected components
# finds CC (clustering coefficient) of whole network
# does check_sum exercise for degree distribution
# finds average degree overall network
# find shortest pathway overall network

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
  #result of function graph G is to be used in all other functions

def measures_of_network(input_mtx):
  R = 0
  f=open(input_mtx[:-4]+'_network_measures.dat','w')
  f.write('r(thres)\tL(Edges)\tD(densi.)\tcon_comp\tCC\tcheck_sum\tave_degr\n')
  for i in range (0,101):
	R = float(i)/100
	G = get_threshold_matrix(input_mtx,R)
	L = nx.number_of_edges(G) #total number of links : L
	N = nx.number_of_nodes(G) #total number of nodes : N
	max_edge = N*(N-1.)/2     # number of maximum edges(links)
	Compon = nx.number_connected_components(G)
	CC = nx.average_clustering(G)  # clustering coefficient of full network	
	check_sum = 0.     		  # sum of degree distributions for one r
	degree_hist = {}
	values = []	
	for node in G:
		
		if G.degree(node) not in degree_hist: # degree dist part
			degree_hist[G.degree(node)] =1
		else:
			degree_hist[G.degree(node)] +=1

		values.append(G.degree(node))	# average degree part
	
	ave=float(sum(values))/float(nx.number_of_nodes(G))	
	# average degree overall network

	keys = degree_hist.keys()
	keys.sort
	for item in keys:		
		check_sum +=float(degree_hist[item])/float(N)
	
	f.write("%f\t%d\t%f\t%f\t%f\t%f\t%f\n" %(R,L,L/max_edge,Compon,CC,check_sum,ave))	
  f.close()
  #L/max_edge : network density, D

def shortest_path(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_shortest_path.dat','w')
	f.write('r(thre.)\tshorthest_pathlength\n')
	for i in range(0,101):
		R = float(i)/100
		G = get_threshold_matrix(input_mtx,R)
		Compon = nx.connected_component_subgraphs(G) # components
		values_2 = []
		for i in range(len(Compon)):
			if nx.number_of_nodes(Compon[i])>1:
				values_2.append(nx.average_shortest_path_length(Compon[i]))
		
		if len(values_2) == 0:
			f.write("%f\t0.\n" % (R))

		else:
			f.write("%f\t%f\n" % (R, ( sum(values_2)/len(values_2) ) ) )
	f.close()







  
if __name__ == '__main__':
  usage = 'Usage: %s correlation_matrix threshold' % sys.argv[0]
  try:
    infilename_data = sys.argv[1]
    #value = sys.argv[2]
  except:
    print usage; sys.exit(1)

  ###manual choice of the threshold value
  #threshold = float(value)
  #network = get_threshold_matrix(infilename_data, threshold)
  measures_of_network(infilename_data)
  shortest_path(infilename_data)

