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
for i in range(0,(len(deg_dis))):
	print nodes[i], G.degree([nodes[i]]).values()[0], deg_dis[i]


for i in range(0,(len(deg_dis))):
	cn = nodes[i]
	cn_dst_degree = deg_dis[i]
	
	avlbl_nodes = nodes[i+1:]
	print "xxxx", avlbl_nodes
	for node in avlbl_nodes:
		print "what about", node, G.degree([node]).values()[0], nodis[node]
		if G.degree([node]).values()[0] >= nodis[node]:
			print "remove", node
			avlbl_nodes.remove(node)
	print "yyyy", avlbl_nodes
	
	for j in range( G.degree([cn]).values()[0], cn_dst_degree):
		wtf = rnd.randint(0,len(avlbl_nodes) - 1)
		addnode = avlbl_nodes[wtf]
		avlbl_nodes.remove(addnode)
		print "add", cn , addnode
		G.add_edge(cn , addnode)


for i in range(0,(len(deg_dis))):
	print nodes[i], G.degree([nodes[i]]).values()[0], deg_dis[i]

pos = nx.shell_layout(G)
nx.draw(G, pos)

pl.show()
