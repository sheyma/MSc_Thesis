#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# Creating a random network by preserving degree distribution but swapping edges
# use nx.double_edge_swap(G) == method d)
# Problem here : threshold range must be [0.2,0.7] !!!

import networkx as nx
import numpy as np 
import sys  
from math import factorial

def random_graph_d(input_mtx, r):
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

def measures_random_Gd(input_mtx):
  R = 0
  f=open(input_mtx[:-4]+'_Random_Gd_network_measures.dat','w')
  #f.write('r(thres)\tL\tN\tD(dens.)\tcon_comp\tCC(clus.)\tcheck_sum\tave_degr\n')
  for i in range (20,71):
	R = float(i)/100
	Random_Gd = random_graph_d(input_mtx,R)
	L = nx.number_of_edges(Random_Gd)  #total number of links : L
	N = nx.number_of_nodes(Random_Gd)  #total number of nodes : N
	max_edge = N*(N-1.)/2     		   # number of maximum edges(links)
	Compon = nx.number_connected_components(Random_Gd)
	CC = nx.average_clustering(Random_Gd)  # clustering coefficient of full network	
	check_sum = 0.     		  # sum of degree distributions for one r
	degree_hist = {}
	values = []	
	for node in Random_Gd:
		
		if Random_Gd.degree(node) not in degree_hist: # degree dist part
			degree_hist[Random_Gd.degree(node)] =1
		else:
			degree_hist[Random_Gd.degree(node)] +=1

		values.append(Random_Gd.degree(node))	# average degree part
	
	ave=float(sum(values))/(nx.number_of_nodes(Random_Gd))	# average degree overall network
	
	keys = degree_hist.keys()
	keys.sort
	for item in keys:		
		check_sum +=float(degree_hist[item])/float(N)
	
	f.write("%f\t%d\t%f\t%f\t%f\t%f\t%f\t\n" %(R,L,L/max_edge,Compon,CC,check_sum,ave))	
  	#1:threshold 2:L 3:Density 4:connected components 5: clus.coef. 6. check_sum 7:ave
  f.close()
 

def shortest_path(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_Random_Gd_shortest_path.dat','w')
	#f.write('r(thre.)\tshorthest_pathlength\n')
	for i in range(20,71):
		R = float(i)/100
		Random_Gd = random_graph_d(input_mtx,R)
		Compon = nx.connected_component_subgraphs(Random_Gd) # components
		values_2 = []
		for i in range(len(Compon)):
			if nx.number_of_nodes(Compon[i])>1:
				values_2.append(nx.average_shortest_path_length(Compon[i]))
		
		if len(values_2) == 0:
			f.write("%f\t0.\n" % (R))

		else:
			f.write("%f\t%f\n" % (R, ( sum(values_2)/len(values_2) ) ) )
			#1.threshold , 2.shortest pathway
	f.close()

def global_effic(input_mtx): 
	R = 0
	f = open(input_mtx[:-4]+'_Random_Gd_global_efficiency.dat','w')
	g = open(input_mtx[:-4]+'_Random_Gd_node_global_efficiency.dat','w')
	for i in range(20,71):
		R = float(i)/100
		Random_Gd = random_graph_d(input_mtx,R)
		global_eff = 0.
		for node_i in Random_Gd:
			sum_inverse_dist = 0.
			for node_j in Random_Gd:
				if node_i != node_j:
					if nx.has_path(Random_Gd, node_i, node_j) == True:
						sum_inverse_dist += 1. / nx.shortest_path_length(Random_Gd, node_i, node_j)
			A = sum_inverse_dist / nx.number_of_nodes(Random_Gd)  # ?
			g.write('%d\t%f\t%f\n' % ((node_i+1), R, A))
			global_eff += sum_inverse_dist / (nx.number_of_nodes(Random_Gd) - 1.) 
		g.write("\n")
		global_eff = global_eff / nx.number_of_nodes(Random_Gd)
		f.write("%f\t%f\n" % (R, global_eff))
		#1.threshold, 2.global efficieny
	f.close()  
	g.close() 

def local_effic(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_Random_Gd_local_efficency.dat','w')
	g = open(input_mtx[:-4]+'_Random_Gd_node_local_efficency.dat','w')
	#g.write('node\tr(thre.)\tlocal_eff')
	for i in range(20,71):
		R = float(i)/100
		Random_Gd = random_graph_d(input_mtx,R)
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

def small_worldness(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_Random_Gd_small_worldness.dat','w')
	g = open(input_mtx[:-4]+'_Random_Gd_cc_trans_ER.dat','w')	
	#g.write('r(thre.)\t\cc_A\tcc_ER\ttran_A\ttran_ER\n')	
	for i in range(20,71):
		R = float(i)/100
		Random_Gd = random_graph_d(input_mtx,R)
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

def degree_dist(input_mtx):			# degree distribution
	R = 0
	f = open(input_mtx[:-4]+'_Random_Gd_degree_dist.dat', 'w')
	#f.write('node\tr(thre.)\tdeg_hist\tdeg_dist\n')
	for i in range(20,71):
		R = float(i)/100
		Random_Gd = random_graph_d(input_mtx,R)
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

def node_cc(input_mtx):   # cluster coefficient of each node
	R = 0 
	f = open(input_mtx[:-4]+'_Random_Gd_node_cc.dat','w')			
	#f.write('node\tr(thre.)\tnode_cc\n')
	for i in range(20,71):
		R = float(i)/100
		Random_Gd = random_graph_d(input_mtx,R)
		for node in Random_Gd:
			f.write("%d\t%f\t%f\n" % (node+1, R, nx.clustering(Random_Gd,node)))
			# node, threshold, clustering coefficient of node			
		#f.write("\n")
	f.close()

def nodes_of_comp(input_mtx):		# connected components of nodes
	R =0
	f = open(input_mtx[:-4]+'_Random_Gd_nodes_comp_.dat','w')
	#f.write('node\tr(thre.)\tcount\n')
	for i in range(20,71):
		R = float(i)/100
		Random_Gd = random_graph_d(input_mtx,R)
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

def single_degrees(input_mtx): #degree (links) of each node
	R = 0
	f = open(input_mtx[:-4]+'_Random_Gd_single_degrees.dat','w')	
	for i in range(20,71):
		#f.write('node\tr(thre.)\tdegree\n')
		R = float(i)/100
		Random_Gd=random_graph_d(input_mtx,R)
		for node in Random_Gd:
			degree = Random_Gd.degree(node)
			f.write('%d\t%f\t%d\n' % ( (node+1), R, degree ) )
			# 1.node, 2.threshold, 3.degree			
		#f.write("\n")
	f.close	

def binomialCoefficient(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))
  
def motifs(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_Random_Gd_motifs.dat','w')
	for i in range(20,71):
		R = float(i)/100
		Random_Gd = random_graph_d(input_mtx,R)
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
    input_matrix = sys.argv[1]
    input_threshold = sys.argv[2]
  except:
    print usage; sys.exit(1)

R = float(input_threshold)	
network = random_graph_d(input_matrix, R)
#measures_random_Gd(input_matrix)
#shortest_path(input_matrix)
#global_effic(input_matrix)
#local_effic(input_matrix)
#small_worldness(input_matrix)
#degree_dist(input_matrix)
#node_cc(input_matrix)
#nodes_of_comp(input_matrix)
#single_degrees(input_matrix)
#motifs(input_matrix)
