#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import matplotlib.pyplot as pl
import random as rnd

nodes = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63']
deg_dis = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,5,5,5,6,7,7,7])
nodis = dictionary = dict(zip(nodes, deg_dis))


while 1:
	broken = 0
	G = nx.Graph()
	G.add_nodes_from(nodes)
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
				broken = 1
				break
			wtf = rnd.randint(0, max_rnd)
			addnode = avlbl_nodes[wtf]
			avlbl_nodes.remove(addnode)
			print "connect", cn , addnode
			G.add_edge(cn , addnode)
		if broken == 1:
			break
	if broken == 0:
		break


for i in range(0,(len(nodes))):
	print nodes[i], G.degree([nodes[i]]).values()[0], deg_dis[i]

pos = nx.shell_layout(G)
nx.draw(G, pos)

pl.show()
