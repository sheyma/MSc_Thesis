#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np 
import sys  

# finds cc(clustering coefficient) of each node for different r's
# (cluster coefficient distributions) 
# finds degree for each node in a network for different r's
# (note : degree = number of links connected to a node)
# find degree distributions (P_k : degree of single node over all nodes)

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

#def node_cc(input_mtx):
#	R = 0 
#	f = open(input_mtx[:-4]+'_node_cc.dat','w')			
#	G = get_threshold_matrix(input_mtx,R)
#	f.write('node\tr(thre.)\tnode_cc\n')
#	for node in G:
#		for i in range(0,101):
#			thre = float(i)/100		
#			G = get_threshold_matrix(input_mtx,thre)
#			f.write("%d\t%f\t%f\n" % (node+1, thre, nx.clustering(G,node)))
#		f.write("\n")
#	f.close()

def node_cc(input_mtx):   # cluster coefficient of each node
	R = 0 
	f = open(input_mtx[:-4]+'_node_cc.dat','w')			
	f.write('node\tr(thre.)\tnode_cc\n')
	for i in range(0,101):
		R = float(i)/100
		G = get_threshold_matrix(input_mtx,R)
		for node in G:
			f.write("%d\t%f\t%f\n" % (node+1, R, nx.clustering(G,node)))
		f.write("\n")
	f.close()

def single_degrees(input_mtx): #degree (links) of each node
	R = 0
	f = open(input_mtx[:-4]+'_single_degrees.dat','w')	
	for i in range(0,101):
		f.write('node\tr(thre.)\tdegree\n')
		R = float(i)/100
		G=get_threshold_matrix(input_mtx,R)
		for node in G:
			degree = G.degree(node)
			f.write('%d\t%f\t%d\n' % ( (node+1), R, degree ) )
		f.write("\n")
	f.close	

def degree_dist(input_mtx):			# degree distribution
	R = 0
	f = open(input_mtx[:-4]+'_degree_dist.dat', 'w')
	f.write('node\tr(thre.)\tdeg_hist\tdeg_dist\n')
	for i in range(0,101):
		R = float(i)/100
		G = get_threshold_matrix(input_mtx,R)
		check_sum = 0.
		degree_hist = {}
		for node in G:
			if G.degree(node) not in degree_hist:	
				degree_hist[G.degree(node)] = 1
			else:
				degree_hist[G.degree(node)] += 1
		degrees = range(0, nx.number_of_nodes(G)+1,1)
		keys = degree_hist.keys()	#keys of block
		keys.sort
		for item in degrees:
			if item in keys:
				P_k=float(degree_hist[item]) / float(nx.number_of_nodes(G))
				check_sum +=P_k				
				f.write('%d\t%f\t%d\t%f\n' % (item,R,degree_hist[item],P_k))			
			else:
				f.write('%d\t%f\t0\t0.\n' % (item, R))
		f.write("\n")
	f.close()

def local_effic(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_local_efficency.dat','w')
	g = open(input_mtx[:-4]+'_node_local_efficency.dat','w')
	g.write('node\tr(thre.)\tlocal_eff')
	for i in range(0,101):
		R = float(i)/100
		G = get_threshold_matrix(input_mtx,R)
		local_effic = 0
		for node_i in G:
			hiwi = 0.	
			if G.degree(node_i)>1:
				neighborhood_i = G.neighbors(node_i)
				for node_j in neighborhood_i:
					for node_h in neighborhood_i:
						if node_j != node_i:
							hiwi +=1./nx.shortest_path_length(G,node_j,node_i)			
				A = G.degree(node_i) * (G.degree(node_i) -1.)					
				local_effic +=hiwi / A				
				g.write('%d\t%f\t%f\n' % ( (node_i+1), R, (hiwi/A) ) )
				 
			else:
				g.write('%d\t%f\t%f\n' % ((node_i+1), R, hiwi))

			g.write("\n")
			local_effic = local_effic / nx.number_of_nodes(G)
			f.write("%f\t%f\n" % ( R, local_effic))
	f.close()
	g.close()



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
  node_cc(infilename_data)
  degree_dist(infilename_data)
  single_degrees(infilename_data)
  local_effic(infilename_data)		
