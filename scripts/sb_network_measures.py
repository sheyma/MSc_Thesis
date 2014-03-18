#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# sb_thresholding.py : firsty binarize the input matrix (thresholding)
# secondy creates the graph (network) of the binarized matrix
# then measures the basic network properties

import networkx as nx
import numpy as np 
import sys  
from math import factorial

# create a thresholded matrix of test matrix and its graph
def get_graph(input_mtx, r):
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
  #result of function : graph G is to be used in all other functions

# printing out the adjacency matrix of G
def get_print_adjacency_matrix(G):
	print nx.adjacency_matrix(G)

# saving the adjacency matrix in as *.dat file
def get_export_adjacency_matrix(G, filename, threshold):
	M = nx.adjacency_matrix(G)
	f = open(filename[:-4]+'r'+str(threshold)+'.dat', 'w')
	for i in range(len(M)):
		for j in range(len(M)):
			f.write("%d\t" % ( M[i,j] ) )
		f.write("\n")
	f.close

# a few characteristic measures of FULL network G with one threshold
def get_characteristics(G,filename,R):

	N = nx.number_of_nodes(G)		#total number of nodes : N
	L = nx.number_of_edges(G)  		#total number of links : L
	Compon = nx.number_connected_components(G) #number of connected components
	cc = nx.average_clustering(G)	# clustering coefficient : cc
	D = nx.density(G)				# network density: Kappa
	check_sum = 0.
	degree_hist = {}
	values = []
	for node in G:
		if G.degree(node) not in degree_hist:
			degree_hist[G.degree(node)] = 1
		else:
			degree_hist[G.degree(node)] += 1
		values.append(G.degree(node))	
	
	ave_degree = float(sum(values)/float(N))	# average degree: <Kappa>
	keys = degree_hist.keys()
	keys.sort()
	
	for item in keys :
		check_sum += float(degree_hist[item])/float(N)
	
	print 'Test matrix: ', filename
	print 'Threshold: ', R
	print 'Number of nodes: ', N
	print 'Number of links: ', L
	print 'Number of connected components: ', Compon
	print 'Clustering coefficient of full network: ', cc
	print 'Check degree distribution sum: ', check_sum
	print 'Network density: ', D

	print 'Average network degree: ', ave_degree 
	return 0	

# get L and D for full network for  different threshold values
def get_number_of_edges_and_density(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_edges_and_density.dat','w')
	for i in range(0,101):
		R = float(i)/100
		G = get_graph(input_mtx,R)
		L = nx.number_of_edges(G)
		D = nx.density(G)
		f.write("%f\t%d\t%f\n" % (R,L,D))
		#1. threshold, 2. edges, 3. density
	f.close()

# get average clustering coefficient of full network for dif.thre.val.
def get_average_cluster_coefficient(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_cluster_coeffi_ave.dat','w')
	for i in range(0,101):
		R = float(i)/100
		G = get_graph(input_mtx,R)
		cc = nx.average_clustering(G)		
		f.write("%f\t%f\n" % (R,cc))
		#1. threshold, 2. clustering coefficient
	f.close()

# get average degree of full network for different threshold values
def get_degrees_ave(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_degree_ave.dat','w')
	for i in range(0,101):
		R = float(i)/100
		G = get_graph(input_mtx,R)
		N = nx.number_of_nodes(G) 
		values = []
		for node in G:
			values.append(G.degree(node))
		ave_degree = float(sum(values)) / float(N)			
		f.write("%f\t%f\n" % (R,ave_degree))
		#1. threshold, 2. average degree
	f.close()

# get number of connected components of full network for dif.thre.val.
def get_connected_components(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_connected_compo.dat','w')
	for i in range(0,101):
		R = float(i)/100
		G = get_graph(input_mtx,R)
		compon = nx.number_connected_components(G)
		f.write("%f\t%f\n" % (R,compon))
		#1. threshold, 2.number of connected components
	f.close()

# get shortest pathway of network
def get_shortest_pathway(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_shortest_path.dat','w')
	#f.write('r(thre.)\tshorthest_pathlength\n')
	for i in range(0,101):
		R = float(i)/100
		G = get_graph(input_mtx,R)
		Compon = nx.connected_component_subgraphs(G) # components
		values_2 = []
		for i in range(len(Compon)):
			if nx.number_of_nodes(Compon[i])>1:
				values_2.append(nx.average_shortest_path_length(Compon[i]))
		
		if len(values_2) == 0:
			f.write("%f\t0.\n" % (R))

		else:
			f.write("%f\t%f\n" % (R, ( sum(values_2)/len(values_2) ) ) )
			# 1.threshold , 2.shortest pathway
	f.close()

# get local efficiency for full network and single nodes separately
def get_local_efficiency(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_local_efficency_ave.dat','w')
	g = open(input_mtx[:-4]+'_local_efficency_node.dat','w')
	#g.write('node\tr(thre.)\tlocal_eff')
	for i in range(0,101):
		R = float(i)/100
		G = get_graph(input_mtx,R)
		local_effic = 0
		for node_i in G:
			hiwi = 0.	
			if G.degree(node_i)>1:
				neighborhood_i = G.neighbors(node_i)
				for node_j in neighborhood_i:
					for node_h in neighborhood_i:
						if node_j != node_h:   #?
							hiwi +=1./nx.shortest_path_length(G,node_j,node_h)			
				A = G.degree(node_i) * (G.degree(node_i) -1.)					
				local_effic +=hiwi / A				
				g.write('%d\t%f\t%f\n' % ( (node_i+1), R, (hiwi/A) ) )

			else:
				g.write('%d\t%f\t%f\n' % ((node_i+1), R, hiwi))
		g.write("\n")
		local_effic = local_effic / nx.number_of_nodes(G)
		f.write("%f\t%f\n" % ( R, local_effic))
		# 1.threshold, 2.local efficiency
	f.close()
	g.close()

# get global efficiency for full network and single nodes separately
def get_global_effic(input_mtx): 
	R = 0
	f = open(input_mtx[:-4]+'_global_efficiency_ave.dat','w')
	g = open(input_mtx[:-4]+'_global_efficiency_node.dat','w')
	for i in range(0,101):
		R = float(i)/100
		G = get_graph(input_mtx,R)
		global_eff = 0.
		for node_i in G:
			sum_inverse_dist = 0.
			for node_j in G:
				if node_i != node_j:
					if nx.has_path(G, node_i, node_j) == True:
						sum_inverse_dist += 1. / nx.shortest_path_length(G, node_i, node_j)
			A = sum_inverse_dist / nx.number_of_nodes(G)  # ?
			g.write('%d\t%f\t%f\n' % ((node_i+1), R, A))
			#1.node, 2,threshold, 3.global efficiency of node
			global_eff += sum_inverse_dist / (nx.number_of_nodes(G) - 1.) 
		g.write("\n")
		global_eff = global_eff / nx.number_of_nodes(G)
		f.write("%f\t%f\n" % (R, global_eff))
		#1.threshold, 2.global efficieny
	f.close()  
	g.close() 

# get degree distribution P(k)
def get_degree_distribution(input_mtx):			# degree distribution
	R = 0
	f = open(input_mtx[:-4]+'_degree_dist.dat', 'w')
	#f.write('node\tr(thre.)\tdeg_hist\tdeg_dist\n')
	for i in range(0,101):
		R = float(i)/100
		G = get_graph(input_mtx,R)
		check_sum = 0.
		degree_hist = {}
		for node in G:
			if G.degree(node) not in degree_hist:	
				degree_hist[G.degree(node)] = 1
			else:
				degree_hist[G.degree(node)] += 1
		#degrees = range(0, nx.number_of_nodes(G)+1,1)
		degrees = range(1, nx.number_of_nodes(G)+1,1)		
		keys = degree_hist.keys()	#keys of block
		keys.sort
		for item in degrees:
			if item in keys:
				P_k=float(degree_hist[item]) / float(nx.number_of_nodes(G))
				check_sum +=P_k				
				f.write('%d\t%f\t%d\t%f\n' % (item,R,degree_hist[item],P_k))
				#1.node, 2.threshold, 3.degree hist, 4.degree distribution			
			else:
				f.write('%d\t%f\t0\t0.\n' % (item, R))
		#f.write("\n")
	f.close()

# get clustering coefficient of each node
def get_node_clustering_coefficient(input_mtx):   # cluster coefficient of each node
	R = 0 
	f = open(input_mtx[:-4]+'_cluster_coeffi_node.dat','w')			
	#f.write('node\tr(thre.)\tnode_cc\n')
	for i in range(0,101):
		R = float(i)/100
		G = get_graph(input_mtx,R)
		for node in G:
			f.write("%d\t%f\t%f\n" % (node+1, R, nx.clustering(G,node)))
			# node, threshold, clustering coefficient of node			
		#f.write("\n")
	f.close()

# get number of connected components of each node
def get_connected_components_nodes(input_mtx):		# connected components of nodes
	R =0
	f = open(input_mtx[:-4]+'_connected_compo_node.dat','w')
	#f.write('node\tr(thre.)\tcount\n')
	for i in range(0,101):
		R = float(i)/100
		G = get_graph(input_mtx,R)
		comps = nx.connected_component_subgraphs(G)
		count = 0
		for graph in comps:
			count +=1
			liste = graph.nodes()
			for node in liste:
				f.write("%d\t%f\t%d\n" % (node,R,count))
				# 1.node, 2.threshold, 3. connected components		
		#f.write("\n")
	f.close

# get degree of each node
def get_degrees_node(input_mtx): #degree (links) of each node
	R = 0
	f = open(input_mtx[:-4]+'_degree_node.dat','w')	
	for i in range(0,101):
		#f.write('node\tr(thre.)\tdegree\n')
		R = float(i)/100
		G=get_graph(input_mtx,R)
		for node in G:
			degree = G.degree(node)
			f.write('%d\t%f\t%d\n' % ( (node+1), R, degree ) )
			# 1.node, 2.threshold, 3.degree			
		#f.write("\n")
	f.close	

# get small worldness of network 
def get_small_worldness(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_small_worldness.dat','w')
	g = open(input_mtx[:-4]+'_cc_trans_ER.dat','w')	
	#g.write('r(thre.)\t\cc_A\tcc_ER\ttran_A\ttran_ER\n')	
	for i in range(0,101):
		R = float(i)/100
		G = get_graph(input_mtx, R)
		ER_graph = nx.erdos_renyi_graph(nx.number_of_nodes(G), nx.density(G))
		# erdos-renyi, binomial random graph generator ...(N,D:density)	
		cluster = nx.average_clustering(G)   # clustering coef. of whole network
		ER_cluster = nx.average_clustering(ER_graph)	#cc of random graph
		
		transi = nx.transitivity(G)
		ER_transi = nx.transitivity(ER_graph)
	
		g.write("%f\t%f\t%f\t%f\t%f\n" % (R,cluster,ER_cluster,transi,ER_transi ))
		
		f.write("%f\t%f\t%f" % (R, cluster, ER_cluster))
		components = nx.connected_component_subgraphs(G)
		ER_components = nx.connected_component_subgraphs(ER_graph)

		values = []
		ER_values = []
		for i in range(len(components)):
			if nx.number_of_nodes(components[i]) > 1:
				values.append(nx.average_shortest_path_length(components[i]))
		for i in range(len(ER_components)):
			if nx.number_of_nodes(ER_components[i]) > 1:
				ER_values.append(nx.average_shortest_path_length(ER_components[i]))
		if len(values) == 0:
			f.write("\t0.")
		else:
			f.write("\t%f" % (sum(values)/len(values))) # pathlenght

		if len(ER_values) == 0:
			f.write("\t0.")
		else:
			f.write("\t%f" % (sum(ER_values)/len(ER_values)))

		f.write("\t%f\t%f" % (transi, ER_transi))  

		if (ER_cluster*sum(values)*len(values)*sum(ER_values)*len(ER_values)) >0 :
			S_WS = (cluster/ER_cluster) / ((sum(values)/len(values)) / (sum(ER_values)/len(ER_values)))  
		else:
			S_WS = 0.
		if (ER_transi*sum(values)*len(values)*sum(ER_values)*len(ER_values)) >0 :
			S_Delta = (transi/ER_transi) / ((sum(values)/len(values)) / (sum(ER_values)/len(ER_values)))
		else:
			S_Delta = 0.

		f.write("\t%f\t%f" % (S_WS, S_Delta)) # S_WS ~ small worldness 
		f.write("\n")

	f.close() 
	g.close()	 
  #1:threshold 2:cluster-coefficient 3:random-cluster-coefficient 4:shortest-pathlength 5:random-shortest-pathlength 6:transitivity 7:random-transitivity 8:S-Watts-Strogatz 9:S-transitivity" 


def binomialCoefficient(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))
  
def get_motifs(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_motifs.dat','w')
	for i in range(0,101):
		R = float(i)/100
		G = get_graph(input_mtx, R)
		tri_dict = nx.triangles(G)   #number of triangles around nodes in G
		summe = 0
		for node in tri_dict:
			summe += tri_dict[node] # summing up all triangle numbers over nodes

		N = nx.number_of_nodes(G)
		ratio = summe / (3. * binomialCoefficient(N,3)) # ratio to porential tria.

		transi = nx.transitivity(G)
		if transi > 0:
			triads = summe / transi 	# triads
			ratio_triads = triads / (3 * binomialCoefficient(N,3)) #ratio to pot.
		else:
			triads = 0.
			ratio_triads = 0.
    		f.write("%f\t%d\t%f\t%f\t%f\n" % (R, summe/3, ratio, triads, ratio_triads))
	f.close()
    # 1:threshold 2:triangles 3:ratio-to-potential-triangles 4:triads 5:ratio-to-potential-triads



if __name__ == '__main__':
  usage = 'Usage: %s correlation_matrix threshold' % sys.argv[0]
  try:
    input_name = sys.argv[1]
    #value = sys.argv[2]
  except:
    print usage; sys.exit(1)

  ###manual choice of the threshold value
  #threshold = float(value)
  #network = get_graph(input_name, threshold)
  #get_print_adjacency_matrix(network)
  #get_export_adjacency_matrix(network, input_name, threshold)
  #get_characteristics(network, input_name, threshold)
  get_number_of_edges_and_density(input_name)
  get_average_cluster_coefficient(input_name)	
  get_degrees_ave(input_name)	
  get_connected_components(input_name)		
  get_local_efficiency(input_name)
  get_global_effic(input_name)
  get_degree_distribution(input_name)
  get_node_clustering_coefficient(input_name)
  get_connected_components_nodes(input_name)
  get_degrees_node(input_name)  
  get_shortest_pathway(input_name)
  get_small_worldness(input_name)	
  get_motifs(input_name)	
