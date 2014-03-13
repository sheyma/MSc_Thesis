#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# Creating a random network by preserving the degree distribution of test network (e.g. A.txt)
# use nx.configuration_model(degree_seq[integers]) == method c)

import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
from math import factorial 
import sys  


# 1. create a random network with method c

def random_graph_c(matrix, r):
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
  	
	degree_hist = {}
	
	for node in G:
		
		if G.degree(node) not in degree_hist: # degree dist part
			degree_hist[G.degree(node)] =1
		else:
			degree_hist[G.degree(node)] +=1	
	keys = degree_hist.keys()
	degrees = range(0,nx.number_of_nodes(G)+1,1)
	degree_seq = []
	for item in degrees:
		if item in keys:
			degree_seq.append(degree_hist[item])		# degree sequence of nodes	
	Random_Gc_1 = nx.configuration_model(degree_seq)    # returns MULTIGRAPH
	Random_Gc = nx.Graph(Random_Gc_1)					# convert into graph
	return Random_Gc 

def random_graph_c_1(matrix, r):
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
  	
	degree_hist = {}
	
	for node in G:
		
		if G.degree(node) not in degree_hist: # degree dist part
			degree_hist[G.degree(node)] =1
		else:
			degree_hist[G.degree(node)] +=1	
	keys = degree_hist.keys()
	degrees = range(0,nx.number_of_nodes(G)+1,1)
	degree_seq = []
	for item in degrees:
		if item in keys:
			degree_seq.append(degree_hist[item])		# degree sequence of nodes	
	Random_Gc_1 = nx.configuration_model(degree_seq)    # returns MULTIGRAPH
						# convert into graph
	return Random_Gc_1 

# compares some basic network properties before and after collapse
def comparison(matrix):
	f=open(matrix[:-4]+'CONF_multigraph.dat','w')
	g=open(matrix[:-4]+'CONF_singlegraph.dat','w')	
	R = 0
	for i in range(0,101):
		R = float(i)/100	

		Random_Gc = random_graph_c(matrix,R)
		Random_Gc_1 = random_graph_c_1(matrix,R)
		N = nx.number_of_nodes(Random_Gc)
		L = nx.number_of_edges(Random_Gc)
		d = nx.density(Random_Gc)
		Compon = nx.number_connected_components(Random_Gc)
		n = nx.number_of_nodes(Random_Gc)
		L_1 = nx.number_of_edges(Random_Gc)
		d_1 = nx.density(Random_Gc)
		Compon_1 = nx.number_connected_components(Random_Gc)
		
		
		check_sum = 0.
		degree_hist = {}
		values = []

		cs = 0.
		degree_h = {}
		values_1 = []
		for node in Random_Gc:
			if Random_Gc.degree(node) not in degree_hist:
				degree_hist[Random_Gc.degree(node)] = 1
			else:
				degree_hist[Random_Gc.degree(node)] +=1

		values.append(Random_Gc.degree(node))
		ave = float(sum(values))/(nx.number_of_nodes(Random_Gc))
		keys = degree_hist.keys()
		keys.sort()

		for item in keys:
			check_sum +=float(degree_hist[item])/float(N)

		for node in Random_Gc_1:
			if Random_Gc_1.degree(node) not in degree_h:
				degree_h[Random_Gc_1.degree(node)] = 1
			else:
				degree_h[Random_Gc_1.degree(node)] +=1

		values_1.append(Random_Gc_1.degree(node))
		ave_1 = float(sum(values_1))/(nx.number_of_nodes(Random_Gc_1))

		keys_1 = degree_h.keys()
		keys_1.sort()
		for item in keys_1:
			cs +=float(degree_h[item])/float(n)

		f.write("%f\t%d\t%f\t%f\t%f\t%f\t\n" %(R,L,d,Compon,check_sum,ave)) 
		g.write("%f\t%d\t%f\t%f\t%f\t%f\t\n" %(R,L_1,d_1,Compon_1,cs,ave_1))
	f.close() 

def measures_random_Gc(matrix):
	R = 0
	f = open(matrix[:-4]+'_Random_Gc_network_measures.dat', 'w')
	#f.write('r(thres)\tL\tN\tD(dens.)\tcon_comp\tCC(clus.)\tcheck_sum\tave_degr\n')
	for i in range (0,101):
		R = float(i)/100
		Random_Gc = random_graph_c(matrix,R)
		N = nx.number_of_nodes(Random_Gc)
		L = nx.number_of_edges(Random_Gc)
		d = nx.density(Random_Gc)
		Compon = nx.number_connected_components(Random_Gc)
		
		check_sum = 0.
		degree_hist = {}
		values = []


		for node in Random_Gc:
			if Random_Gc.degree(node) not in degree_hist:
				degree_hist[Random_Gc.degree(node)] = 1
			else:
				degree_hist[Random_Gc.degree(node)] +=1

		values.append(Random_Gc.degree(node))
		ave = float(sum(values))/(nx.number_of_nodes(Random_Gc))

		keys = degree_hist.keys()
		keys.sort()
		for item in keys:
			check_sum +=float(degree_hist[item])/float(N)
		CC = nx.average_clustering(Random_Gc)
		f.write("%f\t%d\t%f\t%f\t%f\t%f\t%f\t\n" %(R,L,d,Compon,CC,check_sum,ave)) 
		#1:threshold 2:L 3:Density 4:connected components 5: clus.coef. 6. check_sum 7:ave
	f.close()

def shortest_path(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_Random_Gc_shortest_path.dat','w')
	#f.write('r(thre.)\tshorthest_pathlength\n')
	for i in range(0,101):
		R = float(i)/100
		Random_Gc = random_graph_c(input_mtx,R)
		Compon = nx.connected_component_subgraphs(Random_Gc) # components
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
	
def global_effic(input_mtx): 
	R = 0
	f = open(input_mtx[:-4]+'_Random_Gc_global_efficiency.dat','w')
	g = open(input_mtx[:-4]+'_Random_Gc_node_global_efficiency.dat','w')
	for i in range(0,101):
		R = float(i)/100
		Random_Gc = random_graph_c(input_mtx,R)
		global_eff = 0.
		for node_i in Random_Gc:
			sum_inverse_dist = 0.
			for node_j in Random_Gc:
				if node_i != node_j:
					if nx.has_path(Random_Gc, node_i, node_j) == True:
						sum_inverse_dist += 1. / nx.shortest_path_length(Random_Gc, node_i, node_j)
			A = sum_inverse_dist / nx.number_of_nodes(Random_Gc)  # ? >> Trick
			g.write('%d\t%f\t%f\n' % ((node_i+1), R, A))
			global_eff += sum_inverse_dist / (nx.number_of_nodes(Random_Gc) - 0.999999) 
		g.write("\n")
		global_eff = global_eff / nx.number_of_nodes(Random_Gc)
		f.write("%f\t%f\n" % (R, global_eff))
		#1.threshold, 2.global efficieny
	f.close()  
	g.close() 

def local_effic(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_Random_Gc_local_efficency.dat','w')
	g = open(input_mtx[:-4]+'_Random_Gc_node_local_efficency.dat','w')
	#g.write('node\tr(thre.)\tlocal_eff')
	for i in range(0,101):
		R = float(i)/100
		Random_Gc = random_graph_c(input_mtx,R)
		local_effic = 0
		for node_i in Random_Gc:
			hiwi = 0.	
			if Random_Gc.degree(node_i)>1:
				neighborhood_i = Random_Gc.neighbors(node_i)
				for node_j in neighborhood_i:
					for node_h in neighborhood_i:
						if node_j != node_h:   #?
							hiwi +=1./nx.shortest_path_length(Random_Gc,node_j,node_h)			
				A = Random_Gc.degree(node_i) * (Random_Gc.degree(node_i) -1.)  					
				local_effic +=hiwi / A				
				g.write('%d\t%f\t%f\n' % ( (node_i+1), R, (hiwi/A) ) )

			else:
				g.write('%d\t%f\t%f\n' % ((node_i+1), R, hiwi))
		g.write("\n")
		local_effic = local_effic / nx.number_of_nodes(Random_Gc)
		f.write("%f\t%f\n" % ( R, local_effic))
		# 1.threshold, 2.local efficiency
	f.close()
	g.close()

def small_worldness(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_Random_Gc_small_worldness.dat','w')
	g = open(input_mtx[:-4]+'_Random_Gc_cc_trans_ER.dat','w')	
	#g.write('r(thre.)\t\cc_A\tcc_ER\ttran_A\ttran_ER\n')	
	for i in range(0,101):
		R = float(i)/100
		Random_Gc = random_graph_c(input_mtx,R)
		ER_graph = nx.erdos_renyi_graph(nx.number_of_nodes(Random_Gc), nx.density(Random_Gc))
		# erdos-renyi, binomial random graph generator ...(N,D:density)	

				
		cluster = nx.average_clustering(Random_Gc)   # clustering coef. of whole network
		ER_cluster = nx.average_clustering(ER_graph)	#cc of random graph
		
		transi = nx.transitivity(Random_Gc) 
		ER_transi = nx.transitivity(ER_graph)
	
		g.write("%f\t%f\t%f\t%f\t%f\n" % (R,cluster,ER_cluster,transi,ER_transi ))
		
		f.write("%f\t%f\t%f" % (R, cluster, ER_cluster))
		components = nx.connected_component_subgraphs(Random_Gc)
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
	f = open(input_mtx[:-4]+'_Random_Gc_degree_dist.dat', 'w')
	#f.write('node\tr(thre.)\tdeg_hist\tdeg_dist\n')
	for i in range(0,101):
		R = float(i)/100
		Random_Gc = random_graph_c(input_mtx,R)
		check_sum = 0.
		degree_hist = {}
		for node in Random_Gc:
			if Random_Gc.degree(node) not in degree_hist:	
				degree_hist[Random_Gc.degree(node)] = 1
			else:
				degree_hist[Random_Gc.degree(node)] += 1
		#degrees = range(0, nx.number_of_nodes(Random_Gc)+1,1)
		degrees = range(1, nx.number_of_nodes(Random_Gc)+1,1)		
		keys = degree_hist.keys()	#keys of block
		keys.sort
		for item in degrees:
			if item in keys:
				P_k=float(degree_hist[item]) / float(nx.number_of_nodes(Random_Gc))
				check_sum +=P_k				
				f.write('%d\t%f\t%d\t%f\n' % (item,R,degree_hist[item],P_k))
				#1.node, 2.threshold, 3.degree hist, 4.degree distribution			
			else:
				f.write('%d\t%f\t0\t0.\n' % (item, R))
		#f.write("\n")
	f.close()

def compare_degree_dist(input_mtx):			# degree distribution
	R = 0
	f = open(input_mtx[:-4]+'CONF_multigraph_dd.dat', 'w')
	g = open(input_mtx[:-4]+'CONF_singlegraph_dd.dat', 'w')
	#f.write('node\tr(thre.)\tdeg_hist\tdeg_dist\n')
	for i in range(0,101):
		R = float(i)/100
		Random_Gc = random_graph_c(input_mtx,R)
		check_sum = 0.
		degree_hist = {}
		for node in Random_Gc:
			if Random_Gc.degree(node) not in degree_hist:	
				degree_hist[Random_Gc.degree(node)] = 1
			else:
				degree_hist[Random_Gc.degree(node)] += 1
		#degrees = range(0, nx.number_of_nodes(Random_Gc)+1,1)
		degrees = range(1, nx.number_of_nodes(Random_Gc)+1,1)		
		keys = degree_hist.keys()	#keys of block
		keys.sort
		for item in degrees:
			if item in keys:
				P_k=float(degree_hist[item]) / float(nx.number_of_nodes(Random_Gc))
				check_sum +=P_k				
				f.write('%d\t%f\t%d\t%f\n' % (item,R,degree_hist[item],P_k))
				#1.node, 2.threshold, 3.degree hist, 4.degree distribution			
			else:
				f.write('%d\t%f\t0\t0.\n' % (item, R))
		#f.write("\n")

		Random_Gc_1 = random_graph_c_1(input_mtx,R)
		check_sum1 = 0.
		degree_hist1 = {}
		for node in Random_Gc_1:
			if Random_Gc_1.degree(node) not in degree_hist1:	
				degree_hist1[Random_Gc_1.degree(node)] = 1
			else:
				degree_hist1[Random_Gc_1.degree(node)] += 1
		#degrees = range(0, nx.number_of_nodes(Random_Gc_1)+1,1)
		degrees = range(1, nx.number_of_nodes(Random_Gc_1)+1,1)		
		keys1 = degree_hist1.keys()	#keys of block
		keys1.sort
		for item in degrees:
			if item in keys1:
				P_k1=float(degree_hist1[item]) / float(nx.number_of_nodes(Random_Gc_1))
				check_sum1 +=P_k1				
				g.write('%d\t%f\t%d\t%f\n' % (item,R,degree_hist1[item],P_k1))
				#1.node, 2.threshold, 3.degree hist, 4.degree distribution			
			else:
				g.write('%d\t%f\t0\t0.\n' % (item, R))
		#f.write("\n")


	f.close()
	g.close()









def node_cc(input_mtx):   # cluster coefficient of each node
	R = 0 
	f = open(input_mtx[:-4]+'_Random_Gc_node_cc.dat','w')			
	#f.write('node\tr(thre.)\tnode_cc\n')
	for i in range(0,101):
		R = float(i)/100
		Random_Gc= random_graph_c(input_mtx,R)
		
		for node in Random_Gc:
			f.write("%d\t%f\t%f\n" % (node+1, R, nx.clustering(Random_Gc,node)))
			# node, threshold, clustering coefficient of node			
		#f.write("\n")
	f.close()

def nodes_of_comp(input_mtx):		# connected components of nodes
	R =0
	f = open(input_mtx[:-4]+'_Random_Gc_nodes_comp_.dat','w')
	#f.write('node\tr(thre.)\tcount\n')
	for i in range(0,101):
		R = float(i)/100
		Random_Gc = random_graph_c(input_mtx,R)
		comps = nx.connected_component_subgraphs(Random_Gc)
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
	f = open(input_mtx[:-4]+'_Random_Gc_single_degrees.dat','w')	
	for i in range(0,101):
		#f.write('node\tr(thre.)\tdegree\n')
		R = float(i)/100
		Random_Gc=random_graph_c(input_mtx,R)
		for node in Random_Gc:
			degree = Random_Gc.degree(node)
			f.write('%d\t%f\t%d\n' % ( (node+1), R, degree ) )
			# 1.node, 2.threshold, 3.degree			
		#f.write("\n")
	f.close	


def binomialCoefficient(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))
  
# there is no motifs defined here!



if __name__ == '__main__':
  
  usage = 'Usage: %s correlation_matrix threshold' % sys.argv[0]
  try:
    input_matrix = sys.argv[1]
    #input_threshold = sys.argv[2]
  except:
    print usage; sys.exit(1)

#random_graph_c(input_matrix,float(input_threshold))
#measures_random_Gc(input_matrix)
#shortest_path(input_matrix)
#global_effic(input_matrix)
#local_effic(input_matrix)
#small_worldness(input_matrix)
degree_dist(input_matrix)
#node_cc(input_matrix)
#nodes_of_comp(input_matrix)
#single_degrees(input_matrix)
#comparison(input_matrix)
#compare_degree_dist(input_matrix)
