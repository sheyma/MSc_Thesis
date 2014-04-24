#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np


nodes = ['A','B','C','D']
deg_dis = np.array([3,2,1,2])

G = nx.Graph()
G.add_nodes_from(nodes)

print G.nodes()
print G.edges()

for j in nodes:
	print j	
for i in range(0,(len(deg_dis))):
	print nodes[i], G.degree([nodes[i]]).values()[0], deg_dis[i]

#available_nodes = nodes

for i in range(0,(len(deg_dis))):
	cn = nodes[i]
	cn_dst_degree = deg_dis[i]
	start_k = i+1
	for j in range(0 , cn_dst_degree):
		print cn, "add", j, G.degree([cn]).values()[0]
		# select get a random node from available_nodes
		for k in range(start_k, (len(deg_dis))):
			print cn, "try", nodes[k]
			if G.degree([nodes[k]]).values()[0] < deg_dis[k]:
				print cn, "got", nodes[k]
				G.add_edge(cn , nodes[k])
				start_k += 1
				break

for i in range(0,(len(deg_dis))):
	print nodes[i], G.degree([nodes[i]]).values()[0], deg_dis[i]
