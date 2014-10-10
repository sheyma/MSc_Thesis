#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# Creating a random network by different methhods

import networkx as nx
import numpy as np
from math import factorial 
import math
import matplotlib.pyplot as pl	
import random as rnd
import sys  
import glob
import os

# increase recursion limit for our recursive random_graph function
sys.setrecursionlimit(10000)
# global debug variable
deep = 0

# check the loaded matrix if it is symmetric
def load_matrix(file):
	A = np.loadtxt(file, unpack=True)
	AT = np.transpose(A)
	# check the symmetry				
	if A.shape[0] != A.shape[1] or not (A == AT).all():
		print "error: loaded matrix is not symmetric"
		raise ValueError
	return AT

# create adjacency matrix, ones and zeros according to r
def threshold_matrix(A, r):
	B = np.zeros(A.shape)
	for row in range(A.shape[0]):
		for col in range(A.shape[1]):
			if row != col and A[row, col] >= r:
				B[row, col] = 1
	return B

def plot_graph(G):
	pos = nx.shell_layout(G)
	nx.draw(G, pos)
	pl.show()

def print_adjacency_matrix(B):			
	G = nx.from_numpy_matrix(B)
	hiwi = nx.adjacency_matrix(G)
	#print nx.adjacency_matrix(G)
	print hiwi
	
# create a random network with method a:
# networkx.gnm_random_graph : random graph with given N and L
def get_random_graph_a(B):
	G = nx.from_numpy_matrix(B)
	L = nx.number_of_edges(G)
	N = nx.number_of_nodes(G)
	RG = nx.gnm_random_graph(N, L)
	return RG

# create a random network with method d:
# networkx.double_edge_swap : random graph by swaping two edges 
def get_random_graph_d(B):
	G = nx.from_numpy_matrix(B)
	L = nx.number_of_edges(G)	
	trial = L*(L-1.)/2
	swap_num = L;
	if L >2:
		RG = nx.double_edge_swap(G,nswap=swap_num,max_tries=trial)
		return RG
	else:
		print "No swap possible for number of edges", L
		return G

# create a random network with method g
# networkx.expected_degree_graph : 
# random graph with given degree sequence - a probabilistic approach
def get_random_graph_g(B):
	G = nx.from_numpy_matrix(B)
	degree_seq = nx.degree(G).values()
	RG = nx.expected_degree_graph(degree_seq, seed=None, selfloops=False)
 	return RG

# create a random network with method h
# generating a random network by preserving degree distribution
# reference : Brain Connectivity Toolbox, Rubinov & Sporns, 2009
# ported script : "randmio_und_connected.m"
def get_random_graph_h(B):
	ITER = 1000						# ITERATION CAN BE CHANGED!
	n_col   = np.shape(B)[1]		# number of columns in array
	new_B   = np.triu(B)			# upper triangle of array	
	(j , i) = new_B.nonzero()		# (row,col) index of non-zero elem.
	i.setflags(write=True)
	j.setflags(write=True)	
	K       = len(i)				# total number of non-zero elements
	ITER    = K*ITER				# total iteration number 
	maxAttempts = int(K/(n_col-1))  # max attempts per iteration
	eff     = 0  
	for iter in range(1 , ITER+1 ):
		att = 0
		while att<=maxAttempts:
			rewire = 1
			while 1:
				e1 = int(math.floor(K*rnd.random()))
				e2 = int(math.floor(K*rnd.random()))
				while e1==e2:
					e2 = int(math.floor(K*rnd.random()))
				
				a = i[e1]          # chose a col number from i
				b = j[e1]		   # chose a row number from j		
				c = i[e2]		   # chose another col number from i	
				d = j[e2]		   # chose another row number from j		
								
				if ( ( (a!=c) & (a!=d) ) & ( (b!=c) & (b!=d)) ) :
					break          # make sure that a,b,c,d differ
			
			# flipping edge c-d with 50% probability	
			if rnd.random() > 0.5 :
				i[e2]  = d
				j[e2]  = c
				c      = i[e2]
				d      = j[e2]		
			
			# rewiring condition
			if int(not(bool( B[a,d] or B[c,b] ))) : 
				
				# connectedness condition	
				if int(not(bool( B[a,c] or B[b,d] ))) :
					
					P = B[(a, d) , : ]
					P[0,b] = 0
					P[1,c] = 0
					PN     = P
					PN[:,d]= 1
					PN[:,a]= 1
			
					while 1:
						
						P[0,:] = (B[(P[0,:]!=0), :]).any(0).astype(int) 
						P[1,:] = (B[(P[1,:]!=0), :]).any(0).astype(int)
						
						P = P* (np.logical_not(PN).astype(int))
							
						if int(not((P.any(1)).all())):
							rewire = 0
							break
						
						elif  (P[:,[b, c]].any(0)).any(0):
							break
							
						PN = PN +1
				
				# reassigning edges
				if rewire :
					B[a,d] = B[a,b]
					B[a,b] = 0			
					B[d,a] = B[b,a]
					B[b,a] = 0		
					B[c,b] = B[c,d]
					B[c,d] = 0		
					B[b,c] = B[d,c]
					B[d,c] = 0
				
					# reassigning edge indices
					j[e1]  = d
					j[e2]  = b
					
					eff = eff+1;
					break
		
			att = att +1
	RG = nx.from_numpy_matrix(B)
	return RG

def get_random_graph_k(A , B , maxswap):

	new_A   =  np.triu(A,1)
	(i , j) = new_A.nonzero()
	m		= len(i)
	
	i.setflags(write=True)
	j.setflags(write=True)
	
	nswap   = 0
	while nswap < maxswap :
		while 1: 
			e1  =  rnd.randint(0,m-1)
			e2  =  rnd.randint(0,m-1)
			while e2 == e1 :
				e2  = rnd.randint(0,m-1)
		
			a = i[e1]          # chose a row number from i
			b = j[e1]		   # chose a col number from j		
			c = i[e2]		   # chose another row number from i	
			d = j[e2]		   # chose another col number from j
			if ( ( (a!=c) & (a!=d) ) & ( (b!=c) & (b!=d)) ) :
				break          # make sure that a,b,c,d differ
			
		# flipping edge c-d with 50% probability	
		if rnd.random() > 0.5 :
			i[e2]  = d
			j[e2]  = c
			c      = i[e2]
			d      = j[e2]		

		if int(not(bool( A[a,d] or A[c,b] or B[a,d] or B[c,b] ))):
			A[a,d] = A[a,b]
			A[a,b] = 0
			A[d,a] = A[b,a]
			A[b,a] = 0
			A[c,b] = A[c,d]
			A[c,d] = 0
			A[b,c] = A[d,c]
			A[d,c] = 0
			j[e1]  = d
			j[e2]  = b
			nswap = +1	
	RG = nx.from_numpy_matrix(A)
	return RG	

def export_adjacency_matrix(graph , method, input_mtx, r):		# save adjacency matrix
	#print graph
	hiwi = nx.adjacency_matrix(graph)
	f = open(input_mtx[:-4] + '_' + method + '_ADJ_thr_'+str('%.2f' % (r))+'.dat','w')
	for i in range(len(hiwi)):
		for j in range(len(hiwi)):
			f.write("%d\t" % (hiwi[i,j]))
		f.write("\n")
	f.close()		
	#print str('%.2f' % (r))
	
# a few characteristic measures of FULL network G with one GIVEN threshold
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

# assortativity coefficient of full network
def get_assortativity(G, thr):
	f = open(out_prfx + 'assortativity.dat', 'a')
	
	#print "get_assortativity", thr
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
		if ((den1-num2)!=0):
			assort_coeff = (num1 - num2) / (den1 - num2)
			f.write("%f\t%f\n" % (thr, assort_coeff))
		else:
			assort_coeff = float('NaN')
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



# method : corresponding randomization abbreviation
# input_name : given data matrix
if __name__ == '__main__':
	usage = 'Usage: %s method correlation_matrix [threshold]' % sys.argv[0]
	try:
		method = sys.argv[1]
		input_name = sys.argv[2]
		# new input name required for the randomizatin method k
		name_k     = sys.argv[3]
		
		#input_threshold = sys.argv[3]
	except:
		print usage
		sys.exit(1)

# assign the metdhod abbreviations to the methhods
random_graph_methods = {
	"0" : nx.from_numpy_matrix,
	"a" : get_random_graph_a,
	"d" : get_random_graph_d,
	"g" : get_random_graph_g,
	"h" : get_random_graph_h,
	"k" : get_random_graph_k,
}

if not method in random_graph_methods:
	print "unknown method", method
	sys.exit(1)

# global variable contains the prefix used for output files
out_prfx = input_name[:-4]+'_R'+method+'_'

# remove old out files if exist
filelist = glob.glob(out_prfx + "*.dat")
for f in filelist:
	os.remove(f)


data_matrix     = load_matrix(input_name)
print "input data is loaded! "
method_k_matrix = load_matrix(name_k)

thr = 0.1
maxswap = 2
Random_G = random_graph_methods[method](data_matrix,method_k_matrix, maxswap)
export_adjacency_matrix(Random_G, method, input_name, thr)
print (input_name[:-4] + '_' + method + '_ADJ_thr_'+str('%.2f' % (thr))+'.dat','w')



#for i in range(1,2):
	#thr = float(i) / 100.0
	#print "loop", i, thr
	
	#A = threshold_matrix(data_matrix, thr)
	#try:
		#if method == 'k':
			
			##if thr < 0.09:
				##maxswap = 0
			##else:
			#maxswap = 2				
			
			#B        = threshold_matrix(method_k_matrix, thr) 
			#Random_G = random_graph_methods[method](A,B,maxswap)
		#else:
			#Random_G = random_graph_methods[method](A)
	#except:
		#print "couldn't find a random graph", method, sys.exc_info()[0]
		#continue
	
	#plot_graph(Random_G)
	#print_adjacency_matrix(A)
	#export_adjacency_matrix(Random_G, method, input_name, thr)
	#get_single_network_measures(Random_G, thr)
	#get_assortativity(Random_G, thr)
	#get_local_efficiency(Random_G, thr)
	#get_global_effic(Random_G, thr)
	#get_degree_distribution(Random_G, thr)
	#get_node_cc_and_degree(Random_G, thr)
	#get_connected_components_nodes(Random_G, thr)
	#get_small_worldness(Random_G, thr)
	#get_motifs(Random_G, thr)
