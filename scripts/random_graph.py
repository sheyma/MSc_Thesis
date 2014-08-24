#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import matplotlib.pyplot as pl
import random as rnd

#nodes = ['A','B','C','D','E','F']
#deg_dis = np.array([1,3,2,2,1,1])
nodes = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63']
deg_dis = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,5,5,5,6,7,7,7])
nodis = dictionary = dict(zip(nodes, deg_dis))

deep = 0

def print_graph(G):
	nodes = G.nodes()
	nodes.sort()
	for n in nodes:
		print n, G.degree([n]).values()[0], nodis[n]


def add_edge(G, node1, avlbl_nodes):
	max_rnd = len(avlbl_nodes) - 1
	if max_rnd < 0:
		print "FUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUCK, can't solve"
		return 1
	#wtf = 0
	wtf = rnd.randint(0, max_rnd)
	node2 = avlbl_nodes[wtf]
	print "connect", node1, node2
	G.add_edge(node1 , node2)
	avlbl_nodes.remove(node2)
	return 0

## return -1 not solvable
##         0 added
#def add_edge2(G, nodis)
	#fuck

def get_todo(G):
	nodes = G.nodes()
	#nodes.sort()
	for n in nodes:
		if G.degree([n]).values()[0] < nodis[n]:
			return n
	return False

def available_nodes(G, cn, avl):
	for n in nx.non_neighbors(G, cn):
		if G.degree([n]).values()[0] < nodis[n]:
			avl.append(n)
	avl.sort()
	return len(avl)

## return -1 not solvable
##         0 ready
def random_graph(G):
	global deep
	deep += 1
	cn = get_todo(G)
	if not cn:
		deep -= 1
		return 0
	
	avl = list()
	ret = available_nodes(G, cn, avl)
	print "go", cn, deep, ret
	if ret <= 0:
		deep -= 1
		return -1

	rnd.shuffle(avl)
	for n in avl:
		G.add_edge(cn, n)
		ret = random_graph(G)
		if ret >= 0:
			deep -= 1
			return 0
		G.remove_edge(cn, n)
	deep -= 1
	return -1


#broken = 1
#while broken:
	#broken = 0
	#G = nx.Graph()
	#G.add_nodes_from(nodes)
	#for i in range(0,(len(nodes))):
		#cn = nodes[i]
		#cn_degree = G.degree([cn]).values()[0]
		#cn_dst_degree = deg_dis[i]
		
		#we maintain a list of available nodes
		#avlbl_nodes = nodes[i+1:]
		#print "try ", cn, avlbl_nodes
		#we have to iterate over a _copy_ of the avlbl_nodes when we modify it
		#for node in list(avlbl_nodes):
			#if G.degree([node]).values()[0] >= nodis[node]:
				#print "remove full node", node
				#avlbl_nodes.remove(node)
		#print "free neighbours", cn, avlbl_nodes
		
		#now solve the free degree of current node
		#for j in range( cn_degree, cn_dst_degree):
			#broken = add_edge(G, cn, avlbl_nodes)
			#if broken == 1:
				#break
		#if broken == 1:
			#break

G = nx.Graph()
G.add_nodes_from(nodes)

random_graph(G)

print_graph(G)

#pos = nx.shell_layout(G)
#nx.draw(G, pos)

#pl.show()
