#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# Creating a random network with same node and link numbers of test network 
# use networkx.gnm_random_graph(N,L)  == method a)


import networkx as nx
import numpy as np 
import sys  
from math import factorial


# 1. create a random network 

def get_random_graph_a(matrix, r):
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
  G=nx.from_numpy_matrix(B,create_using=nx.Graph())  
  L = nx.number_of_edges(G) 						 # total number of links: L  
  N = nx.number_of_nodes(G) 						 # total number of nodes : N	
  Random_Ga = nx.gnm_random_graph(N,L)				 # random graph
  return Random_Ga

# a few characteristic measures of FULL network G with one threshold
def get_characteristics(filename,R):
	Random_Ga = get_random_graph_a(filename,R)
	N = nx.number_of_nodes(Random_Ga)		#total number of nodes : N
	L = nx.number_of_edges(Random_Ga)  		#total number of links : L
	Compon = nx.number_connected_components(Random_Ga) #number of connected components
	cc = nx.average_clustering(Random_Ga)	# clustering coefficient : cc
	D = nx.density(Random_Ga)				# network density: Kappa
	check_sum = 0.
	degree_hist = {}
	values = []
	for node in Random_Ga:
		if Random_Ga.degree(node) not in degree_hist:
			degree_hist[Random_Ga.degree(node)] = 1
		else:
			degree_hist[Random_Ga.degree(node)] += 1
		values.append(Random_Ga.degree(node))	
	
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
	print 'Assortativity : ', nx.degree_assortativity_coefficient(Random_Ga)
	return 0	

# Single (Average) Network Measures : 
# get L and D for full network for  different threshold values
# get average clustering coefficient of full network for dif.thre.val.
# get average degree of full network for different threshold values
# get number of connected components of full network for dif.thre.val.
# get shortest pathway of network
def get_single_network_measures(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_Ra_single_network_measures.dat','w')
	for i in range(0,101):
		R = float(i)/100
		Random_Ga = get_random_graph_a(input_mtx,R)
		N = nx.number_of_nodes(Random_Ga)
		L = nx.number_of_edges(Random_Ga)
		D = nx.density(Random_Ga)
		cc = nx.average_clustering(Random_Ga)
		compon = nx.number_connected_components(Random_Ga)
		Con_sub = nx.connected_component_subgraphs(Random_Ga)		
	      
		values = []
		values_2 =[]

		for node in Random_Ga:
			values.append(Random_Ga.degree(node))
		ave_deg = float(sum(values)) / float(N)
	
		f.write("%f\t%d\t%f\t%f\t%f\t%f\t" % (R,L,D,cc,ave_deg,compon))
		#1. threshold, 2. edges, 3. density 4.clustering coefficient
		#5. average degree, 6. number of connected components
	
		for i in range(len(Con_sub)):
			if nx.number_of_nodes(Con_sub[i])>1:
				values_2.append(nx.average_shortest_path_length(Con_sub[i]))

		if len(values_2)==0:
			f.write("0.\n")
		else:
			f.write("%f\n" % (sum(values_2)/len(values_2)))
		#7. shortest pathway
		
	f.close()


# assortativity
def get_assort(input_mtx):
	R = 0.09
	f = open(input_mtx[:-4]+'_Ra_assortativity.dat','w')
	Random_Ga = get_random_graph_a(input_mtx,R)
	N = nx.number_of_nodes(Random_Ga)
	L = nx.number_of_edges(Random_Ga)
	A = 0
	B = 0
	C = 0
	D = 0
	for node_i in Random_Ga:
	  for node_j in Random_Ga:
	     k_i = Random_Ga.degree(node_i)
	     k_j = Random_Ga.degree(node_j)
	     A += float(  k_i * k_j)
	     B +=float(0.5 * (k_i+k_j))
	     C += float(0.5 * (pow(k_i,2) + pow(k_j,2)))
	     D += float(0.5 * k_i + k_j) 
	print "L : " , L
	print A, pow(B,2), C, pow(D,2) 
	
# get local efficiency for full network and single nodes separately
def get_local_efficiency(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_Ra_local_efficency_ave.dat','w')
	g = open(input_mtx[:-4]+'_Ra_local_efficency_node.dat','w')
	#g.write('node\tr(thre.)\tlocal_eff')
	for i in range(0,101):
		R = float(i)/100
		Random_Ga = get_random_graph_a(input_mtx,R)
		local_effic = 0
		for node_i in Random_Ga:
			hiwi = 0.	
			if Random_Ga.degree(node_i)>1:
				neighborhood_i = Random_Ga.neighbors(node_i)
				for node_j in neighborhood_i:
					for node_h in neighborhood_i:
						if node_j != node_h:   #?
							hiwi +=1./nx.shortest_path_length(Random_Ga,node_j,node_h)			
				A = Random_Ga.degree(node_i) * (Random_Ga.degree(node_i) -1.)					
				local_effic +=hiwi / A				
				g.write('%d\t%f\t%f\n' % ( (node_i+1), R, (hiwi/A) ) )

			else:
				g.write('%d\t%f\t%f\n' % ((node_i+1), R, hiwi))
		#g.write("\n")
		local_effic = local_effic / nx.number_of_nodes(Random_Ga)
		f.write("%f\t%f\n" % ( R, local_effic))
		# 1.threshold, 2.local efficiency
	f.close()
	g.close()

# get global efficiency for full network and single nodes separately
def get_global_effic(input_mtx): 
	R = 0
	f = open(input_mtx[:-4]+'_Ra_global_efficiency_ave.dat','w')
	g = open(input_mtx[:-4]+'_Ra_global_efficiency_node.dat','w')
	for i in range(0,101):
		R = float(i)/100
		Random_Ga = get_random_graph_a(input_mtx,R)
		global_eff = 0.
		for node_i in Random_Ga:
			sum_inverse_dist = 0.
			for node_j in Random_Ga:
				if node_i != node_j:
					if nx.has_path(Random_Ga, node_i, node_j) == True:
						sum_inverse_dist += 1. / nx.shortest_path_length(Random_Ga, node_i, node_j)
			A = sum_inverse_dist / nx.number_of_nodes(Random_Ga)  # ?
			g.write('%d\t%f\t%f\n' % ((node_i+1), R, A))
			#1.node, 2,threshold, 3.global efficiency of node
			global_eff += sum_inverse_dist / (nx.number_of_nodes(Random_Ga) - 1.) 
		#g.write("\n")
		global_eff = global_eff / nx.number_of_nodes(Random_Ga)
		f.write("%f\t%f\n" % (R, global_eff))
		#1.threshold, 2.global efficieny
	f.close()  
	g.close() 

# get degree distribution P(k)
def get_degree_distribution(input_mtx):			# degree distribution
	R = 0
	f = open(input_mtx[:-4]+'_Ra_degree_dist.dat', 'w')
	#f.write('node\tr(thre.)\tdeg_hist\tdeg_dist\n')
	for i in range(0,101):
		R = float(i)/100
		Random_Ga = get_random_graph_a(input_mtx,R)
		check_sum = 0.
		degree_hist = {}
		for node in Random_Ga:
			if Random_Ga.degree(node) not in degree_hist:	
				degree_hist[Random_Ga.degree(node)] = 1
			else:
				degree_hist[Random_Ga.degree(node)] += 1
		#degrees = range(0, nx.number_of_nodes(Random_Ga)+1,1)
		degrees = range(1, nx.number_of_nodes(Random_Ga)+1,1)		
		keys = degree_hist.keys()	#keys of block
		keys.sort
		for item in degrees:
			if item in keys:
				P_k=float(degree_hist[item]) / float(nx.number_of_nodes(Random_Ga))
				check_sum +=P_k				
				f.write('%d\t%f\t%d\t%f\n' % (item,R,degree_hist[item],P_k))
				#1.node, 2.threshold, 3.degree hist, 4.degree distribution			
			else:
				f.write('%d\t%f\t0\t0.\n' % (item, R))
		#f.write("\n")
	f.close()

# get clustering coefficient and degree of each node
def get_node_cc_and_degree(input_mtx):  
	R = 0 
	f = open(input_mtx[:-4]+'_Ra_cc_and_degree_node.dat','w')			
	#f.write('node\tr(thre.)\tnode_cc\n')
	for i in range(0,101):
		R = float(i)/100
		Random_Ga = get_random_graph_a(input_mtx,R)
		for node in Random_Ga:
			cc_node = nx.clustering(Random_Ga,node)
			deg_node = Random_Ga.degree(node)

			f.write("%d\t%f\t%f\t%f\n" % (node+1, R, cc_node, deg_node))
			#1. node, 2. threshold, 3. clustering coefficient of node 
			#4. degree of node			
	f.close()

# get number of connected components of each node
def get_connected_components_nodes(input_mtx):		# connected components of nodes
	R =0
	f = open(input_mtx[:-4]+'_Ra_connected_compo_node.dat','w')
	#f.write('node\tr(thre.)\tcount\n')
	for i in range(0,101):
		R = float(i)/100
		Random_Ga = get_random_graph_a(input_mtx,R)
		comps = nx.connected_component_subgraphs(Random_Ga)
		count = 0
		for graph in comps:
			count +=1
			liste = graph.nodes()
			for node in liste:
				f.write("%d\t%f\t%d\n" % (node,R,count))
				# 1.node, 2.threshold, 3. connected components		
		#f.write("\n")
	f.close

def get_small_worldness(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_Ra_small_worldness.dat','w')
	g = open(input_mtx[:-4]+'_Ra_cc_trans_ER.dat','w')	
	#g.write('r(thre.)\t\cc_A\tcc_ER\ttran_A\ttran_ER\n')	
	for i in range(0,101):
		R = float(i)/100
		Random_Ga = get_random_graph_a(input_mtx,R)
		ER_graph = nx.erdos_renyi_graph(nx.number_of_nodes(Random_Ga), nx.density(Random_Ga))
		# erdos-renyi, binomial random graph generator ...(N,D:density)	
		cluster = nx.average_clustering(Random_Ga)   # clustering coef. of whole network
		ER_cluster = nx.average_clustering(ER_graph)	#cc of random graph
		
		transi = nx.transitivity(Random_Ga)
		ER_transi = nx.transitivity(ER_graph)
	
		g.write("%f\t%f\t%f\t%f\t%f\n" % (R,cluster,ER_cluster,transi,ER_transi ))
		
		f.write("%f\t%f\t%f" % (R, cluster, ER_cluster))
		components = nx.connected_component_subgraphs(Random_Ga)
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
	f = open(input_mtx[:-4]+'_Ra_motifs.dat','w')
	for i in range(0,101):
		R = float(i)/100
		Random_Ga = get_random_graph_a(input_mtx,R)
		tri_dict = nx.triangles(Random_Ga)   #number of triangles around nodes in Random_Ga
		summe = 0
		for node in tri_dict:
			summe += tri_dict[node] # summing up all triangle numbers over nodes

		N = nx.number_of_nodes(Random_Ga)
		ratio = summe / (3. * binomialCoefficient(N,3)) # ratio to porential tria.

		transi = nx.transitivity(Random_Ga)
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
    input_threshold = sys.argv[2]
  except:
    print usage; sys.exit(1)

###manual choice of the threshold value
threshold = float(input_threshold)
network = get_random_graph_a(input_name, threshold)
get_characteristics(input_name, threshold)
#get_single_network_measures(input_name)
#get_local_efficiency(input_name)
#get_global_effic(input_name)
#get_degree_distribution(input_name)
#get_node_cc_and_degree(input_name)  
#get_connected_components_nodes(input_name)
#get_small_worldness(input_name)	
#get_motifs(input_name)	
get_assort(input_name)