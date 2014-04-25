#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import matplotlib.pyplot as pl
import random as rnd

nodes = ['A','B','C','D','E','F']
deg_dis = np.array([1,3,2,2,1,1])
nodis = dictionary = dict(zip(nodes, deg_dis))

G = nx.Graph()
G.add_nodes_from(nodes)

print G.nodes()
print G.edges()

for j in nodes:
	print j	
for i in range(0,(len(nodes))):
	print nodes[i], G.degree([nodes[i]]).values()[0], deg_dis[i]


for i in range(0,(len(nodes))):
	cn = nodes[i]
	cn_degree = G.degree([cn]).values()[0]
	cn_dst_degree = deg_dis[i]
	
	# we maintain a list of available nodes
	avlbl_nodes = nodes[i+1:]
	print "try ", cn, avlbl_nodes
	# we have to iterate over a _copy_ of the avlbl_nodes when we modify it
	for node in list(avlbl_nodes):
		if G.degree([node]).values()[0] >= nodis[node]:
			print "remove full node", node
			avlbl_nodes.remove(node)
	print "free neighbours", cn, avlbl_nodes
	
	# now solve the free degree of current node
	for j in range( cn_degree, cn_dst_degree):
		max_rnd = len(avlbl_nodes) - 1
		if max_rnd < 0:
			print "FUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUCK, can't solve"
			break
		wtf = rnd.randint(0, max_rnd)
		addnode = avlbl_nodes[wtf]
		avlbl_nodes.remove(addnode)
		print "connect", cn , addnode
		G.add_edge(cn , addnode)


for i in range(0,(len(nodes))):
	print nodes[i], G.degree([nodes[i]]).values()[0], deg_dis[i]

pos = nx.shell_layout(G)
nx.draw(G, pos)

pl.show()
