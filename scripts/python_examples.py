#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# Small python examples

import numpy as np
import networkx as nx

# create a matrix

a = np.matrix('0 1 1 1 1; 1 0 1 1 1; 1 1 0 0 0; 1 1 0 0 1; 1 1 0 0 0')

G = nx.from_numpy_matrix(a, create_using=nx.Graph())
N = nx.number_of_nodes(G);
L = nx.number_of_edges(G);
degrees_of_nodes = G.degree()

print "number of nodes", N
print "number of edges", L
print "degrees of nodes", degrees_of_nodes

histo = {} 
#Each key in dictionary has a value
# histo[key]=value

for node in G:
	if G.degree(node) not in histo:
		histo[G.degree(node)] = 1
	else:
		histo[G.degree(node)] += 1 

print "histogram of degrees: ", histo
print "keys of histo: ", histo.keys()
print "values of histo: ", histo.values()

my_keys = histo.keys()
my_values = histo.values()

A = []

for j in range (0,len(my_keys)):
	for i in range(0,my_values[j]):
		A.append(my_keys[j]) 
print "degree sequence: ", A
