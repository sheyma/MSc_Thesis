#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

def get_threshold_matrix(filename, threshold_value):
  import networkx as nx
  import numpy as np 
  A = np.transpose(np.loadtxt(filename, unpack=True)) 

  B = np.zeros((len(A),len(A)))

  for row in range(len(A)):
    for item in range(len(A)):
      if row != item:
        if A[row,item] >= threshold_value:
          B[row,item] = 1
        else:
          B[row,item] = 0

  G=nx.from_numpy_matrix(B,create_using=nx.Graph())
  return G

def print_adjacency_matrix(G):
  import networkx as nx
  print nx.adjacency_matrix(G)

def export_adjacency_matrix(G, filename, threshold_value):
  import networkx as nx

  hiwi = nx.adjacency_matrix(G)
  
  f = open(filename[:-4]+'_r'+str(threshold_value)+'.dat','w')
  for i in range(len(hiwi)):
    for j in range(len(hiwi)):
      f.write("%d\t" % (hiwi[i,j]))
    f.write("\n")
  f.close()

def get_characteristics(G, filename):
  import networkx as nx
  print 'calculating characteristics'
    
  n_nodes = nx.number_of_nodes(G)
  n_edges = nx.number_of_edges(G)
  n_components = nx.number_connected_components(G)
  print 'number of nodes:', n_nodes
  print 'number of edges:', n_edges
  print 'number of components:', n_components
 
  print 'degree histogram'
  check_sum = 0.
  degree_hist = {}
  for node in G:
    if G.degree(node) not in degree_hist:
      degree_hist[G.degree(node)] = 1
    else:
      degree_hist[G.degree(node)] += 1
    
  keys = degree_hist.keys()
  keys.sort()
  for item in keys:
    print item, degree_hist[item]
    check_sum += float(degree_hist[item])/float(n_nodes)
    
  print "check sum: %f" % check_sum
            
  #print 'clustering coefficient'
  print 'clustering coefficient of full network', nx.average_clustering(G)
  return 0

def get_number_of_edges(filename):
  import networkx as nx
  threshold = 0
  f = open(filename[:-4]+'_edges.dat','w')
  for i in range(0,101):
    threshold = float(i)/100
    G = get_threshold_matrix(filename, threshold)
    print 'number of edges:', nx.number_of_edges(G)
    max_number_edges = nx.number_of_nodes(G) * (nx.number_of_nodes(G) - 1.) / 2
    f.write("%f\t%d\t%f\n" % (threshold, nx.number_of_edges(G), nx.number_of_edges(G)/max_number_edges))
  f.close()

def get_cluster_coefficients(filename):
  import networkx as nx
  threshold = 0
  f = open(filename[:-4]+'_cc.dat','w')
  for i in range(0,101):
    threshold = float(i)/100
    G = get_threshold_matrix(filename, threshold)
    for node in G:
      f.write('%d\t%f\t%f\n' % (node, threshold, nx.clustering(G, node)))
    f.write("\n")
    print 'clustering coefficients for threshold: %f' % threshold
  f.close()
  
def get_average_cluster_coefficient(filename):
  import networkx as nx
  threshold = 0
  f = open(filename[:-4]+'_average_cc.dat','w')
  for i in range(0,101):
    threshold = float(i)/100
    G = get_threshold_matrix(filename, threshold)
    print 'threshold: %f, average cluster coefficient: %f' %(threshold, nx.average_clustering(G))
    f.write("%f\t%f\n" % (threshold, nx.average_clustering(G)))
  f.close()

def get_degree_distr(filename):
  import networkx as nx
  threshold = 0
  f = open(filename[:-4]+'_degreedistr.dat','w')
  for i in range(0,101):
    threshold = float(i)/100
    G = get_threshold_matrix(filename, threshold)
    check_sum = 0.
    degree_hist = {}
    for node in G:
      if G.degree(node) not in degree_hist:
	degree_hist[G.degree(node)] = 1
      else:
	degree_hist[G.degree(node)] += 1
    degrees = range(0, nx.number_of_nodes(G)+1, 1)
    keys = degree_hist.keys()
    keys.sort()
    for item in degrees:
      if item in keys:
        check_sum += float(degree_hist[item])/float(nx.number_of_nodes(G))
        f.write('%d\t%f\t%d\t%f\n' % (item, threshold, degree_hist[item], float(degree_hist[item])/float(nx.number_of_nodes(G))))
        #print item, degree_hist[item], float(degree_hist[item])/float(nx.number_of_nodes(G)))
      else:
        f.write('%d\t%f\t0\t0.\n' % (item, threshold))
    f.write("\n")
    print 'degree distribution for threshold: %f, check sum: %f' % (threshold, check_sum)
  f.close()

def get_degrees(filename):
  import networkx as nx
  threshold = 0
  f = open(filename[:-4]+'_degrees.dat','w')
  for i in range(0,101):
    threshold = float(i)/100
    G = get_threshold_matrix(filename, threshold)
    print 'threshold: %f' % threshold
    for node in G:
        f.write('%d\t%f\t%d\n' % ((node+1), threshold, G.degree(node)))
    f.write("\n")
  f.close()

def get_average_degree(filename):
  import networkx as nx
  threshold = 0
  f = open(filename[:-4]+'_average_degree.dat','w')
  for i in range(0,101):
    threshold = float(i)/100
    G = get_threshold_matrix(filename, threshold)
    values = []
    for node in G:
       values.append(G.degree(node))
    f.write('%f\t%f\n' % (threshold, float(sum(values))/float(nx.number_of_nodes(G))))
    #f.write('%f\t%f\n' % (threshold, float(sum(values))/float(len(values))))
    print 'threshold: %f, average degree %f' % (threshold, float(sum(values))/float(len(values)))
  f.close()

def get_number_of_components(filename):
  import networkx as nx
  threshold = 0
  f = open(filename[:-4]+'_components.dat','w')
  for i in range(0,101):
    threshold = float(i)/100
    G = get_threshold_matrix(filename, threshold)
    print 'number of connected components:', nx.number_connected_components(G)
    f.write("%f\t%d\n" % (threshold, nx.number_connected_components(G)))
  f.close()

def get_local_efficiency(filename):
  import networkx as nx
  threshold = 0
  f = open(filename[:-4]+'_local_efficiency.dat','w')
  g = open(filename[:-4]+'_node_local_efficiency.dat','w')
  for i in range(0,101):
    threshold = float(i)/100
    G = get_threshold_matrix(filename, threshold)
    local_efficiency = 0.
    for node_i in G:
      hiwi = 0.
      if G.degree(node_i) > 1:
        neighborhood_i = G.neighbors(node_i)
        for node_j in neighborhood_i:
	  for node_h in neighborhood_i:
	    if node_j != node_h:
	      hiwi += 1. / nx.shortest_path_length(G, node_j, node_h)
        g.write('%d\t%f\t%f\n' % ((node_i+1), threshold, (hiwi/ (G.degree(node_i) * (G.degree(node_i) -1.)) ) ))
        local_efficiency += (hiwi/ (G.degree(node_i) * (G.degree(node_i) -1.)) )
      else:
        g.write('%d\t%f\t%f\n' % ((node_i+1), threshold, hiwi))
    g.write("\n")
    local_efficiency = local_efficiency / nx.number_of_nodes(G)
    f.write("%f\t%f\n" % (threshold, local_efficiency))
    print 'local efficiency for threshold %f: %f ' % (threshold, local_efficiency)
      
  f.close()  
  g.close()  

def get_global_efficiency(filename): 
  import networkx as nx
  threshold = 0
  f = open(filename[:-4]+'_global_efficiency.dat','w')
  g = open(filename[:-4]+'_node_global_efficiency.dat','w')
  for i in range(0,101):
    threshold = float(i)/100
    G = get_threshold_matrix(filename, threshold)
    global_efficiency = 0.
    for node_i in G:
      sum_inverse_dist = 0.
      for node_j in G:
	if node_i != node_j:
	  if nx.has_path(G, node_i, node_j) == True:
	    sum_inverse_dist += 1. / nx.shortest_path_length(G, node_i, node_j)
      g.write('%d\t%f\t%f\n' % ((node_i+1), threshold, (sum_inverse_dist / nx.number_of_nodes(G)) ))
      global_efficiency += sum_inverse_dist / (nx.number_of_nodes(G) - 1.) 
    g.write("\n")
    global_efficiency = global_efficiency / nx.number_of_nodes(G)
    f.write("%f\t%f\n" % (threshold, global_efficiency))
    print 'global efficiency for threshold %f: %f ' % (threshold, global_efficiency)
      
  f.close()  
  g.close()  
  
def get_shortest_pathlength(filename): 
  import networkx as nx
  threshold = 0
  f = open(filename[:-4]+'_shortest_pathlength.dat','w')
  for i in range(0,101):
    threshold = float(i)/100
    G = get_threshold_matrix(filename, threshold)
    components = nx.connected_component_subgraphs(G)
    values = []
    for i in range(len(components)):
      if nx.number_of_nodes(components[i]) > 1:
        values.append(nx.average_shortest_path_length(components[i]))
    if len(values) == 0:
      f.write("%f\t0.\n" % (threshold))
      print 'average shortest pathlength: 0'
    else:
      f.write("%f\t%f\n" % (threshold, (sum(values)/len(values))))
      print 'average shortest pathlength: %f ' % (sum(values)/len(values))
  f.close()

def get_harmonic_pathlength(filename):
  import networkx as nx
  threshold = 0
  f = open(filename[:-4]+'_harmonic_pathlength.dat','w')
  for i in range(0,101):
    threshold = float(i)/100
    G = get_threshold_matrix(filename, threshold)
    components = nx.connected_component_subgraphs(G)
    values =[]
    for i in range(len(components)):
      adjacency = nx.adjacency_matrix(components[i])
      hiwi = 0
      values_indi = []
      for row in adjacency:
        if row.sum() > 0:
	  hiwi += 1./row.sum()
        values_indi.append(hiwi)
      if len(values_indi) > 0:
        values.append(sum(values_indi)/len(values_indi))
    #the following holds only for a connected network
    #adjacency = nx.adjacency_matrix(G)
    #hiwi = 0
    #values = []
    #for row in adjacency:
      #if row.sum() > 0:
	#hiwi += 1./row.sum()
      #values.append(hiwi)
    if len(values) == 0:
      f.write("%f\t0.\n" % (threshold))
      print 'harmonic pathlength: 0'
    else:
      print 'harmonic pathlength: %f' % (sum(values)/len(values))
      f.write("%f\t%f\n" % (threshold, (sum(values)/len(values))))
  f.close()

def get_nodes_of_components(filename, value):
  import networkx as nx
  threshold = value
  f = open(filename[:-4]+'_nodes_components_r'+str(threshold)+'.dat','w')
  G = get_threshold_matrix(filename, threshold)
  print 'number of connected components:', nx.number_connected_components(G)
  comps = nx.connected_component_subgraphs(G)
  counter = 0
  for graph in comps:
    counter += 1
    print counter, nx.number_of_nodes(graph), graph.nodes()
    liste = graph.nodes()
    for node in graph.nodes():
      f.write("%d\t%d\n" % (counter, node))
    #f.write("%f\t%d\t%d\n" % (threshold, counter, nx.number_of_nodes(graph)))
  f.close()

def get_small_worldness(filename):
  import networkx as nx
  threshold = 0
  f = open(filename[:-4]+'_small_worldness.dat','w')
  for i in range(0,101):
    threshold = float(i)/100
    G = get_threshold_matrix(filename, threshold)
    ER_graph = nx.erdos_renyi_graph(nx.number_of_nodes(G), nx.density(G))

    cluster = nx.average_clustering(G)
    ER_cluster = nx.average_clustering(ER_graph)
    
    transi = nx.transitivity(G)
    ER_transi = nx.transitivity(ER_graph)

    print 'threshold: %f, average cluster coefficient: %f, random nw: %f, transitivity: %f, random nw: %f' %(threshold, cluster, ER_cluster, transi, ER_transi)

    f.write("%f\t%f\t%f" % (threshold, cluster, ER_cluster))
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
      f.write("\t%f" % (sum(values)/len(values)))

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
    
    f.write("\t%f\t%f" % (S_WS, S_Delta))  
    f.write("\n")
    
  f.close()  
  print "1:threshold 2:cluster-coefficient 3:random-cluster-coefficient 4:shortest-pathlength 5:random-shortest-pathlength 6:transitivity 7:random-transitivity 8:S-Watts-Strogatz 9:S-transitivity" 

def binomialCoefficient(n, k):
    from math import factorial
    return factorial(n) // (factorial(k) * factorial(n - k))
  
def get_motifs(filename):
  import networkx as nx
  from math import factorial
  threshold = 0
  f = open(filename[:-4]+'_motifs.dat','w')
  for i in range(0,101):
    threshold = float(i)/100
    G = get_threshold_matrix(filename, threshold)
    tri_dict = nx.triangles(G)
    summe = 0
    for node in tri_dict:
      summe += tri_dict[node]
    
    N = nx.number_of_nodes(G)
    ratio = summe / (3. * binomialCoefficient(N,3))
    
    transi = nx.transitivity(G)
    if transi > 0:
      triads = summe / transi 
      ratio_triads = triads / (3 * binomialCoefficient(N,3))
    else:
      triads = 0.
      ratio_triads = 0.
    
    print 'threshold: %f, number of triangles: %f, ratio: %f, triads: %f, ratio: %f' %(threshold, summe/3, ratio, triads, ratio_triads)
    f.write("%f\t%d\t%f\t%f\t%f\n" % (threshold, summe/3, ratio, triads, ratio_triads))
  f.close()
  print "1:threshold 2:#triangles 3:ratio-to-potential-triangles 4:triads 5:ratio-to-potential-triads"
  
if __name__ == '__main__':
  import sys
  import networkx as nx
  usage = 'Usage: %s correlation_matrix threshold' % sys.argv[0]
  try:
    infilename_data = sys.argv[1]
    value = sys.argv[2]
  except:
    print usage; sys.exit(1)

  ###manual choice of the threshold value
  threshold = float(value)
  network = get_threshold_matrix(infilename_data, threshold)
  print_adjacency_matrix(network)
  export_adjacency_matrix(network, infilename_data, threshold)

  #threshold = 0.51
  #get_characteristics(network, infilename_data)
  #get_number_of_edges(infilename_data)
  #get_cluster_coefficients(infilename_data)
  #get_average_cluster_coefficient(infilename_data)
  #get_degree_distr(infilename_data)
  #get_degrees(infilename_data)
  #get_average_degree(infilename_data)
  #get_number_of_components(infilename_data)
  #get_nodes_of_components(infilename_data, threshold)
  #get_shortest_pathlength(infilename_data)
  #get_harmonic_pathlength(infilename_data)  
  #get_global_efficiency(infilename_data)
  ####automated choice of the the threshold value
  #get_local_efficiency(infilename_data)
  #get_small_worldness(infilename_data)
