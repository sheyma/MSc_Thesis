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

def get_threshold_matrix(input_mtx,R):
  A = np.transpose(np.loadtxt(input_mtx, unpack=True)) 
  B = np.zeros((len(A),len(A)))

  for row in range(len(A)):
    for item in range(len(A)):
      if row != item:
        if A[row,item] >= R:
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



def nodes_of_comp(input_mtx):
	R =0
	f = open(input_mtx[:-4]+'_nodes_comp_.dat','w')
	f.write('node\tr(thre.)\tcount\n')
	for i in range(0,101):
		R = float(i)/100
		G = get_threshold_matrix(input_mtx,R)
		comps = nx.connected_component_subgraphs(G)
		count = 0
		for graph in comps:
			count +=1
			liste = graph.nodes()
			for node in liste:
				f.write("%d\t%f\t%d\n" % (node,R,count))
		f.write("\n")
	f.close

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
	f.close()
	g.close()


def global_effic(input_mtx): 
	R = 0
	f = open(input_mtx[:-4]+'_global_efficiency.dat','w')
	g = open(input_mtx[:-4]+'_node_global_efficiency.dat','w')
	for i in range(0,101):
		R = float(i)/100
		G = get_threshold_matrix(input_mtx,R)
		global_eff = 0.
		for node_i in G:
			sum_inverse_dist = 0.
			for node_j in G:
				if node_i != node_j:
					if nx.has_path(G, node_i, node_j) == True:
						sum_inverse_dist += 1. / nx.shortest_path_length(G, node_i, node_j)
			A = sum_inverse_dist / nx.number_of_nodes(G)  # ?
			g.write('%d\t%f\t%f\n' % ((node_i+1), R, A))
			global_eff += sum_inverse_dist / (nx.number_of_nodes(G) - 1.) 
		g.write("\n")
		global_eff = global_eff / nx.number_of_nodes(G)
		f.write("%f\t%f\n" % (R, global_eff))
	f.close()  
	g.close()  


def small_worldness(input_mtx):
	R = 0
	f = open(input_mtx[:-4]+'_small_worldness.dat','w')
	g = open(input_mtx[:-4]+'_cc_trans_ER.dat','w')	
	g.write('r(thre.)\t\cc_A\tcc_ER\ttran_A\ttran_ER\n')	
	for i in range(0,101):
		R = float(i)/100
		G = get_threshold_matrix(input_mtx, R)
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
  #print "1:threshold 2:cluster-coefficient 3:random-cluster-coefficient 4:shortest-pathlength 5:random-shortest-pathlength 6:transitivity 7:random-transitivity 8:S-Watts-Strogatz 9:S-transitivity" 

  
def binomialCoefficient(n, k):
    from math import factorial
    return factorial(n) // (factorial(k) * factorial(n - k))
  
def motifs(input_mtx):
	from math import factorial
	R = 0
	f = open(input_mtx[:-4]+'_motifs.dat','w')
	for i in range(0,101):
		R = float(i)/100
		G = get_threshold_matrix(input_mtx, R)
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

    #print 'threshold: %f, number of triangles: %f, ratio: %f, triads: %f, ratio: %f' %(threshold, summe/3, ratio, triads, ratio_triads)
    		f.write("%f\t%d\t%f\t%f\t%f\n" % (R, summe/3, ratio, triads, ratio_triads))
	f.close()
  #print "1:threshold 2:#triangles 3:ratio-to-potential-triangles 4:triads 5:ratio-to-potential-triads"
  


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
  node_cc(infilename_data)
  degree_dist(infilename_data)
  single_degrees(infilename_data)
  local_effic(infilename_data)		
  global_effic(infilename_data)	
  nodes_of_comp(infilename_data)		
  small_worldness(infilename_data)
  motifs(infilename_data)
