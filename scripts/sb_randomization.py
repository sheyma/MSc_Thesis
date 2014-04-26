#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# Creating a random network by preserving the degree distribution of test network (e.g. A.txt)
# use nx.configuration_model(degree_seq[integers]) == method c)

import networkx as nx
#from networkx.algorithms import bipartite
import numpy as np
from math import factorial 
import matplotlib.pyplot as pl
import random as rnd
import sys  
import glob
import os

# create a random network with method a
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

# 1. create a random network with method b
def get_random_graph_b(matrix, r):
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
  N = nx.number_of_nodes(G)							 # number of nodes in G	
  d = nx.density(G)									 # network density of G
  Random_Gb = nx.erdos_renyi_graph(N,d)	 # random graph
  return Random_Gb

# create a random network with method c
def get_random_graph_c(matrix, r):
	A = np.transpose(np.loadtxt(matrix, unpack=True))
	B = np.zeros((len(A),len(A)))

	for row in range(len(A)):
		for item in range(len(A)):
		  if row != item:
			if A[row,item] >= r:
			  B[row,item] = 1
			else:
			  B[row,item] = 0
		#print B	   								   # print binarized matrix
	G=nx.from_numpy_matrix(B,create_using=nx.Graph())  # create graph of thresolded A
	# G is now non-directed graph
	degree_hist = {}
	
	
	for node in G:
		if G.degree(node) not in degree_hist: # degree dist part
			degree_hist[G.degree(node)] =1
		else:
			degree_hist[G.degree(node)] +=1
	keys = degree_hist.keys()
	values = degree_hist.values()
	degree_seq = []
	
	for j in range(0,len(keys)):
		for i in range(0,(values[j])):
			degree_seq.append(keys[j])
	
	#Random_Gc = nx.configuration_model(degree_seq,create_using=nx.Graph())	
	
	Random_Gc = nx.random_degree_sequence_graph(degree_seq,tries=100)
	
	#pos = nx.shell_layout(Random_Gc)
	#nx.draw(Random_Gc, pos)
	#pl.show()
	
	return Random_Gc

# create a random network with method d
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
	L = nx.number_of_edges(G)	
	trial = L*(L-1.)/2
	swap_num = L;
	if L >2:
		Random_Gd = nx.double_edge_swap(G,nswap=swap_num,max_tries=trial)
		return Random_Gd
	else:
		print "No swap possible for R=", float(r), "number of edges", L
		return G



# a few characteristic measures of FULL network G with one threshold
def get_characteristics(G, thr, input_name):
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
	
	print 'Test matrix: ', input_name
	print 'Threshold: ', thr
	print 'Number of nodes: ', N
	print 'Number of links: ', L
	print 'Number of connected components: ', Compon
	print 'Clustering coefficient of full network: ', cc
	print 'Check degree distribution sum: ', check_sum
	print 'Network density: ', D

	print 'Average network degree: ', ave_degree 
	return 0	

# get L and D for full network for  different threshold values
# get average clustering coefficient of full network for dif.thre.val.
# get average degree of full network for different threshold values
# get number of connected components of full network for dif.thre.val.
# get shortest pathway of network
def get_single_network_measures(G, thr):
	f = open(out_prfx + 'single_network_measures.dat', 'a')
	N = nx.number_of_nodes(G)
	L = nx.number_of_edges(G)
	D = nx.density(G)
	cc = nx.average_clustering(G)
	compon = nx.number_connected_components(G)
	Con_sub = nx.connected_component_subgraphs(G)

	values = []
	values_2 =[]

	for node in G:
		values.append(G.degree(node))
	ave_deg = float(sum(values)) / float(N)
	
	f.write("%f\t%d\t%f\t%f\t%f\t%f\t" % (thr, L, D, cc, ave_deg, compon))
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

def get_assortativity(G, thr):
	f = open(out_prfx + 'assortativity.dat', 'a')
	
	print "get_assortativity", thr
	degrees = G.degree()
	m = float(nx.number_of_edges(G))
	num1, num2, den1 = 0, 0, 0
	for source, target in G.edges():
		
		num1 += degrees[source] * degrees[target]
		num2 += degrees[source] + degrees[target]
		den1 += degrees[source] **2 + degrees[target] **2
	if m!=0:
		num1 /= m
		den1 /= 2*m
		num2 = (num2 / (2*m)) ** 2
		#assort_coeff_1 = nx.degree_assortativity_coefficient(G)
		#print 'Assortativity : ', assort_coeff_1 
		if ((den1-num2)!=0):
			assort_coeff = (num1 - num2) / (den1 - num2)
			f.write("%f\t%f\n" % (thr, assort_coeff))
		#print "Assortativity manual :", assort_coeff
	f.close()


# get local efficiency for full network and single nodes separately
def get_local_efficiency(G, thr):
	f = open(out_prfx + 'local_efficency_ave.dat', 'a')
	g = open(out_prfx + 'local_efficency_node.dat', 'a')
	#g.write('node\tr(thre.)\tlocal_eff')
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
			g.write('%d\t%f\t%f\n' % ( (node_i+1), thr, (hiwi/A) ) )

		else:
			g.write('%d\t%f\t%f\n' % ((node_i+1), thr, hiwi))
	#g.write("\n")
	local_effic = local_effic / nx.number_of_nodes(G)
	f.write("%f\t%f\n" % ( thr, local_effic))
	# 1.threshold, 2.local efficiency
	f.close()
	g.close()

# get global efficiency for full network and single nodes separately
def get_global_effic(G, thr):
	f = open(out_prfx + 'global_efficiency_ave.dat', 'a')
	g = open(out_prfx + 'global_efficiency_node.dat', 'a')
	global_eff = 0.
	for node_i in G:
		sum_inverse_dist = 0.
		for node_j in G:
			if node_i != node_j:
				if nx.has_path(G, node_i, node_j) == True:
					sum_inverse_dist += 1. / nx.shortest_path_length(G, node_i, node_j)
		A = sum_inverse_dist / nx.number_of_nodes(G)  # ?
		g.write('%d\t%f\t%f\n' % ((node_i+1), thr, A))
		#1.node, 2,threshold, 3.global efficiency of node
		global_eff += sum_inverse_dist / (nx.number_of_nodes(G) - 1.)
	#g.write("\n")
	global_eff = global_eff / nx.number_of_nodes(G)
	f.write("%f\t%f\n" % (thr, global_eff))
	#1.threshold, 2.global efficieny
	f.close()  
	g.close() 

# get degree distribution P(k)
def get_degree_distribution(G, thr):
	f = open(out_prfx + 'degree_dist.dat', 'a')
	#f.write('node\tr(thre.)\tdeg_hist\tdeg_dist\n')
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
			f.write('%d\t%f\t%d\t%f\n' % (item,thr ,degree_hist[item],P_k))
			#1.node, 2.threshold, 3.degree hist, 4.degree distribution			
		else:
			f.write('%d\t%f\t0\t0.\n' % (item, thr))
	#f.write("\n")
	f.close()

# get clustering coefficient and degree of each node
def get_node_cc_and_degree(G, thr):
	f = open(out_prfx + 'cc_and_degree_node.dat', 'a')
	#f.write('node\tr(thre.)\tnode_cc\n')
	for node in G:
		cc_node = nx.clustering(G,node)
		deg_node = G.degree(node)

		f.write("%d\t%f\t%f\t%f\n" % (node+1, thr, cc_node, deg_node))
		#1. node, 2. threshold, 3. clustering coefficient of node 
		#4. degree of node			
	f.close()

# get number of connected components of each node
def get_connected_components_nodes(G, thr):
	f = open(out_prfx + 'connected_compo_node.dat', 'a')
	#f.write('node\tr(thre.)\tcount\n')
	comps = nx.connected_component_subgraphs(G)
	count = 0
	for graph in comps:
		count +=1
		liste = graph.nodes()
		for node in liste:
			f.write("%d\t%f\t%d\n" % (node, thr, count))
			# 1.node, 2.threshold, 3. connected components		
	#f.write("\n")
	f.close

def get_small_worldness(G, thr):
	f = open(out_prfx + 'small_worldness.dat', 'a')
	g = open(out_prfx + 'cc_trans_ER.dat', 'a')
	#g.write('r(thre.)\t\cc_A\tcc_ER\ttran_A\ttran_ER\n')
	ER_graph = nx.erdos_renyi_graph(nx.number_of_nodes(G), nx.density(G))
	# erdos-renyi, binomial random graph generator ...(N,D:density)	
	cluster = nx.average_clustering(G)   # clustering coef. of whole network
	ER_cluster = nx.average_clustering(ER_graph)	#cc of random graph
	
	transi = nx.transitivity(G)
	ER_transi = nx.transitivity(ER_graph)

	g.write("%f\t%f\t%f\t%f\t%f\n" % (thr, cluster,ER_cluster,transi,ER_transi ))
	
	f.write("%f\t%f\t%f" % (thr, cluster, ER_cluster))
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
  #f.write = 1:threshold 2:cluster-coefficient 3:random-cluster-coefficient 4:shortest-pathlength 5:random-shortest-pathlength 6:transitivity 7:random-transitivity 8:S-Watts-Strogatz 9:S-transitivity" 





def binomialCoefficient(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))
  
def get_motifs(G, thr):
	f = open(out_prfx + 'motifs.dat', 'a')
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
	f.write("%f\t%d\t%f\t%f\t%f\n" % (thr, summe/3, ratio, triads, ratio_triads))
	f.close()
    # 1:threshold 2:triangles 3:ratio-to-potential-triangles 4:triads 5:ratio-to-potential-triads




if __name__ == '__main__':
	usage = 'Usage: %s method correlation_matrix [threshold]' % sys.argv[0]
	try:
		method = sys.argv[1]
		input_name = sys.argv[2]
		#input_threshold = sys.argv[3]
	except:
		print usage
		sys.exit(1)

random_graph_methods = {
	"a" : get_random_graph_a,
	"b" : get_random_graph_b,
	"c" : get_random_graph_c,
	"d" : get_random_graph_d,
}

out_prfx = input_name[:-4]+'_R'+method+'_'

# remove old out files if exist
filelist = glob.glob(out_prfx + "*.dat")
for f in filelist:
	os.remove(f)

for i in range(0, 101):
	thr = float(i) / 100.0
	print "loop", i, thr
	try:
		#Random_G = nx.random_degree_sequence_graph([1,1],tries=100)
		Random_G = random_graph_methods[method](input_name, thr)
	except:
		print "couldn't find a random graph"
		continue
		
	get_characteristics(Random_G, thr, input_name)
	get_single_network_measures(Random_G, thr)
	get_assortativity(Random_G, thr)
	get_local_efficiency(Random_G, thr)
	get_global_effic(Random_G, thr)
	get_degree_distribution(Random_G, thr)
	get_node_cc_and_degree(Random_G, thr)
	get_connected_components_nodes(Random_G, thr)
	get_small_worldness(Random_G, thr)
	get_motifs(Random_G, thr)
