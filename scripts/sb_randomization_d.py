#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# Creating a random network by preserving degree distribution but swapping edges
# use nx.double_edge_swap(G) == method d)
# Problem here : threshold range must be [0.2,0.7] !!!

import networkx as nx
import numpy as np 
import sys  
from math import factorial

def get_random_graph_d(input_mtx, r):
	A = np.transpose(np.loadtxt(input_mtx, unpack=True)) 
	B = np.zeros((len(A),len(A)))

	for row in range(len(A)):
		for item in range(len(A)):
			if row != item:
				if A[row,item] >= r:
					B[row,item] = 1
				else:
					B[row,item] = 0

	G=nx.from_numpy_matrix(B,create_using=nx.Graph())  #undirected graph G
	Random_Gd = nx.double_edge_swap(G)  
	#print nx.adjacency_matrix(Random_Gd)
	return Random_Gd

# a few characteristic measures of FULL network G with one threshold
def get_characteristics(filename,R):
	Random_Gd = get_random_graph_d(filename,R)
	N = nx.number_of_nodes(Random_Gd)		#total number of nodes : N
	L = nx.number_of_edges(Random_Gd)  		#total number of links : L
	Compon = nx.number_connected_components(Random_Gd) #number of connected components
	cc = nx.average_clustering(Random_Gd)	# clustering coefficient : cc
	D = nx.density(Random_Gd)				# network density: Kappa
	check_sum = 0.
	degree_hist = {}
	values = []
	for node in Random_Gd:
		if Random_Gd.degree(node) not in degree_hist:
			degree_hist[Random_Gd.degree(node)] = 1
		else:
			degree_hist[Random_Gd.degree(node)] += 1
		values.append(Random_Gd.degree(node))	
	
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
	f = open(input_mtx[:-4]+'_Rd_edges_and_density.dat','w')
	for i in range(20,71):
		R = float(i)/100
		Random_Gd = get_random_graph_d(input_mtx,R)
		L = nx.number_of_edges(Random_Gd)
		D = nx.density(Random_Gd)
		f.write("%f\t%d\t%f\n" % (R,L,D))
		#1. threshold, 2. edges, 3. density
	f.close()

# get average clustering coefficient of full network for dif.thre.val.
def get_average_cluster_coefficient(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_Rd_cluster_coeffi_ave.dat','w')
	for i in range(20,71):
		R = float(i)/100
		Random_Gd = get_random_graph_d(input_mtx,R)
		cc = nx.average_clustering(Random_Gd)		
		f.write("%f\t%f\n" % (R,cc))
		#1. threshold, 2. clustering coefficient
	f.close()

# get average degree of full network for different threshold values
def get_degrees_ave(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_Rd_degree_ave.dat','w')
	for i in range(20,71):
		R = float(i)/100
		Random_Gd = get_random_graph_d(input_mtx,R)
		N = nx.number_of_nodes(Random_Gd) 
		values = []
		for node in Random_Gd:
			values.append(Random_Gd.degree(node))
		ave_degree = float(sum(values)) / float(N)			
		f.write("%f\t%f\n" % (R,ave_degree))
		#1. threshold, 2. average degree
	f.close()

# get number of connected components of full network for dif.thre.val.
def get_connected_components(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_Rd_connected_compo.dat','w')
	for i in range(20,71):
		R = float(i)/100
		Random_Gd = get_random_graph_d(input_mtx,R)
		compon = nx.number_connected_components(Random_Gd)
		f.write("%f\t%f\n" % (R,compon))
		#1. threshold, 2.number of connected components
	f.close()

# get shortest pathway of network
def get_shortest_pathway(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_Rd_shortest_path.dat','w')
	#f.write('r(thre.)\tshorthest_pathlength\n')
	for i in range(20,71):
		R = float(i)/100
		Random_Gd = get_random_graph_d(input_mtx,R)
		Compon = nx.connected_component_subgraphs(Random_Gd) # components
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
	f = open(input_mtx[:-4]+'_Rd_local_efficency_ave.dat','w')
	g = open(input_mtx[:-4]+'_Rd_local_efficency_node.dat','w')
	#g.write('node\tr(thre.)\tlocal_eff')
	for i in range(20,71):
		R = float(i)/100
		Random_Gd = get_random_graph_d(input_mtx,R)
		local_effic = 0
		for node_i in Random_Gd:
			hiwi = 0.	
			if Random_Gd.degree(node_i)>1:
				neighborhood_i = Random_Gd.neighbors(node_i)
				for node_j in neighborhood_i:
					for node_h in neighborhood_i:
						if node_j != node_h:   #?
							hiwi +=1./nx.shortest_path_length(Random_Gd,node_j,node_h)			
				A = Random_Gd.degree(node_i) * (Random_Gd.degree(node_i) -1.)					
				local_effic +=hiwi / A				
				g.write('%d\t%f\t%f\n' % ( (node_i+1), R, (hiwi/A) ) )

			else:
				g.write('%d\t%f\t%f\n' % ((node_i+1), R, hiwi))
		g.write("\n")
		local_effic = local_effic / nx.number_of_nodes(Random_Gd)
		f.write("%f\t%f\n" % ( R, local_effic))
		# 1.threshold, 2.local efficiency
	f.close()
	g.close()

# get global efficiency for full network and single nodes separately
def get_global_effic(input_mtx): 
	R = 0
	f = open(input_mtx[:-4]+'_Rd_global_efficiency_ave.dat','w')
	g = open(input_mtx[:-4]+'_Rd_global_efficiency_node.dat','w')
	for i in range(20,71):
		R = float(i)/100
		Random_Gd = get_random_graph_d(input_mtx,R)
		global_eff = 0.
		for node_i in Random_Gd:
			sum_inverse_dist = 0.
			for node_j in Random_Gd:
				if node_i != node_j:
					if nx.has_path(Random_Gd, node_i, node_j) == True:
						sum_inverse_dist += 1. / nx.shortest_path_length(Random_Gd, node_i, node_j)
			A = sum_inverse_dist / nx.number_of_nodes(Random_Gd)  # ?
			g.write('%d\t%f\t%f\n' % ((node_i+1), R, A))
			#1.node, 2,threshold, 3.global efficiency of node
			global_eff += sum_inverse_dist / (nx.number_of_nodes(Random_Gd) - 1.) 
		g.write("\n")
		global_eff = global_eff / nx.number_of_nodes(Random_Gd)
		f.write("%f\t%f\n" % (R, global_eff))
		#1.threshold, 2.global efficieny
	f.close()  
	g.close() 

# get degree distribution P(k)
def get_degree_distribution(input_mtx):			# degree distribution
	R = 0
	f = open(input_mtx[:-4]+'_Rd_degree_dist.dat', 'w')
	#f.write('node\tr(thre.)\tdeg_hist\tdeg_dist\n')
	for i in range(20,71):
		R = float(i)/100
		Random_Gd = get_random_graph_d(input_mtx,R)
		check_sum = 0.
		degree_hist = {}
		for node in Random_Gd:
			if Random_Gd.degree(node) not in degree_hist:	
				degree_hist[Random_Gd.degree(node)] = 1
			else:
				degree_hist[Random_Gd.degree(node)] += 1
		#degrees = range(0, nx.number_of_nodes(Random_Gd)+1,1)
		degrees = range(1, nx.number_of_nodes(Random_Gd)+1,1)		
		keys = degree_hist.keys()	#keys of block
		keys.sort
		for item in degrees:
			if item in keys:
				P_k=float(degree_hist[item]) / float(nx.number_of_nodes(Random_Gd))
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
	f = open(input_mtx[:-4]+'_Rd_cluster_coeffi_node.dat','w')			
	#f.write('node\tr(thre.)\tnode_cc\n')
	for i in range(20,71):
		R = float(i)/100
		Random_Gd = get_random_graph_d(input_mtx,R)
		for node in Random_Gd:
			f.write("%d\t%f\t%f\n" % (node+1, R, nx.clustering(Random_Gd,node)))
			# node, threshold, clustering coefficient of node			
		#f.write("\n")
	f.close()

# get number of connected components of each node
def get_connected_components_nodes(input_mtx):		# connected components of nodes
	R =0
	f = open(input_mtx[:-4]+'_Rd_connected_compo_node.dat','w')
	#f.write('node\tr(thre.)\tcount\n')
	for i in range(20,71):
		R = float(i)/100
		Random_Gd = get_random_graph_d(input_mtx,R)
		comps = nx.connected_component_subgraphs(Random_Gd)
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
	f = open(input_mtx[:-4]+'_Rd_degree_node.dat','w')	
	for i in range(20,71):
		#f.write('node\tr(thre.)\tdegree\n')
		R = float(i)/100
		Random_Gd=get_random_graph_d(input_mtx,R)
		for node in Random_Gd:
			degree = Random_Gd.degree(node)
			f.write('%d\t%f\t%d\n' % ( (node+1), R, degree ) )
			# 1.node, 2.threshold, 3.degree			
		#f.write("\n")
	f.close	

def get_small_worldness(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_Rd_small_worldness.dat','w')
	g = open(input_mtx[:-4]+'_Rd_cc_trans_ER.dat','w')	
	#g.write('r(thre.)\t\cc_A\tcc_ER\ttran_A\ttran_ER\n')	
	for i in range(20,71):
		R = float(i)/100
		Random_Gd = get_random_graph_d(input_mtx,R)
		ER_graph = nx.erdos_renyi_graph(nx.number_of_nodes(Random_Gd), nx.density(Random_Gd))
		# erdos-renyi, binomial random graph generator ...(N,D:density)	
		cluster = nx.average_clustering(Random_Gd)   # clustering coef. of whole network
		ER_cluster = nx.average_clustering(ER_graph)	#cc of random graph
		
		transi = nx.transitivity(Random_Gd)
		ER_transi = nx.transitivity(ER_graph)
	
		g.write("%f\t%f\t%f\t%f\t%f\n" % (R,cluster,ER_cluster,transi,ER_transi ))
		
		f.write("%f\t%f\t%f" % (R, cluster, ER_cluster))
		components = nx.connected_component_subgraphs(Random_Gd)
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
  #f.write = 1:threshold 2:cluster-coefficient 3:random-cluster-coefficient 4:shortest-pathlength 5:random-shortest-pathlength 6:transitivity 7:random-transitivity 8:S-Watts-Strogatz 9:S-transitivity" 





def binomialCoefficient(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))
  
def get_motifs(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_Rd_motifs.dat','w')
	for i in range(20,71):
		R = float(i)/100
		Random_Gd = get_random_graph_d(input_mtx,R)
		tri_dict = nx.triangles(Random_Gd)   #number of triangles around nodes in Random_Gd
		summe = 0
		for node in tri_dict:
			summe += tri_dict[node] # summing up all triangle numbers over nodes

		N = nx.number_of_nodes(Random_Gd)
		ratio = summe / (3. * binomialCoefficient(N,3)) # ratio to porential tria.

		transi = nx.transitivity(Random_Gd)
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
    #input_threshold = sys.argv[2]
  except:
    print usage; sys.exit(1)

###manual choice of the threshold value
#threshold = float(input_threshold)
#network = get_random_graph_d(input_name, threshold)
#get_characteristics(input_name, threshold)

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
