#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import re
import math
from cmath import log, sqrt
from pydelay import dde23
import sys

""" Was es schon kann, sieht man am besten unten oder mit "pydoc netpy".

    Am Ende gibt's ein Beispiel.

"""

class simnet:
    """Simuliert ein Netzwerk.
    """
    def __init__(self, eqns, G, H, T, params, coupling, noise):
        self.G = G
        self.H = H
        self.T = T
        self.params = params
        self.coupling = coupling
        self.eqns = eqns
        self.noise = noise
        self.sol = dict()

        self.eqnsN = neteqns(self.G, self.H, self.T, self.eqns, params=self.params, coupling=self.coupling)
        self.noiseN = netnoise(self.eqnsN, self.noise)

        print >> sys.stderr, 'Gleichungen fuers ganze Netz:'
        print >> sys.stderr, self.eqnsN  #Debug
        print >> sys.stderr, 'Noise fuers ganze Netz:'
        print >> sys.stderr, self.noiseN #Debug

        self.ddeN = dde23(eqns=self.eqnsN, params=self.params, noise=self.noiseN)

    def transient(self, initial_conditions, tmax):
        """Zunaechst die History mit der Zeitserie eines Systems fuellen.
           Aber nur, wenn noch nichts gepickled und nichts an sys.argv[0]
           (das File mit den Gleichungen) geändert wurde.
        """
        import cPickle as pickle
        import hashlib

        file1 = hashlib.md5(open(sys.argv[0]).read()).hexdigest()+'.pickle'

        try:
            sol1 = pickle.load(open(file1,'r'))
            print >> sys.stderr, 'Transiente wird aus {0} genommen'.format(file1)
        except:
            eqns1 = neteqns([[rowsum(self.G,0)]], self.H, self.T, self.eqns, params=self.params, coupling=self.coupling)

            print >> sys.stderr, 'rechne Transiente für 1 System mit Gleichungen:'
            print >> sys.stderr, eqns1 #Debug

            dde1 = dde23(eqns=eqns1, params=self.params)
            dde1.set_sim_params(tfinal=tmax)

            for v in initial_conditions:
                for w in dde1.hist:
                    if re.match(v,w):
                        dde1.hist[w] += initial_conditions[v]

            dde1.run()

            sol1 = dde1.sol
            del(dde1)

            pickle.dump(sol1,open(file1,'w'))

        return sol1

    def hist_from_1(self, sol1):
        n = dict()
        for v in self.ddeN.hist:
            for w in sol1:
                if re.match(w[:1],v):
                    n[v] = sol1[w]
        self.ddeN.hist_from_arrays(n)

    def hist_from_arrays(self, arrays):
        """Übergibt das dict aus arrays an pydelay für ddeN.
        """
        self.ddeN.hist_from_arrays(arrays)

    def hist_from_funcs(self, funcs):
        """Übergibt das dict aus funcs an pydelay für ddeN.
        """
        self.ddeN.hist_from_funcs(funcs)

    def run(self, tmax):
        """Jetzt das ganze Netz.
        """
        self.ddeN.set_sim_params(tfinal=tmax)
        self.ddeN.run()

        """Zeitserien mit linearer Zeit speichern.
        """
        self.sol = self.ddeN.sol_spl(np.linspace(0,tmax,10000))

class msf:
    """Nützliches, um die MSF auszurechnen.
       TODO: Im Moment kommen sigma, alpha und beta noch aus params.
             D.h., diese Parameter dürfen nicht in den Gln. vorkommen.
             Das sollte aber so nicht sein.
    """

    def __init__(self, eqns, vareqns, params, groups, H, coupling):
        self.resolution = 1000
        self.params = params
        self.resarray = np.linspace(0, self.params['tau'], self.resolution)
        self.eqns = eqns
        self.vareqns = vareqns
        self.groups = groups
        self.H = H
        self.transient = dict()
        self.T = np.ones_like(self.groups) * self.params['tau']
        self.coupling = coupling

    def run_transient(self, inits, ttransients=20):
        ttime = ttransients * self.params['tau']
        tsystem = neteqns(self.groups, self.H, self.T, self.eqns, params=self.params, coupling=self.coupling)
        tdde = dde23(eqns=tsystem, params=self.params)
        tdde.set_sim_params(tfinal=ttime)
        for v in inits:
            for w in tdde.hist:
                if re.match(v,w):
                    tdde.hist[w][-1] = inits[v]
        tdde.run()
        self.transient = tdde.sol_spl(self.resarray)

    def __fillinitdict(self, d, h):
        """Macht die erste History für Manifold und Variationen.
           Variationen sind immer 1e-15. Vielleicht Parameter zulassen?
        """
        if len(self.transient) == 0:
            self.run_transient(dict())
        n = dict()
        for v in h:
            if re.match('var',v):
                n[v] = np.linspace(1e-15, 1e-15, len(self.resarray))
            else:
                for w in d:
                    if re.match(w,v):
                        n[v] = d[w]
        return n

    def __renormalized(self, d, norm):
        """Renormiert alle Variationen im Dictionary.
        """
        n = dict()
        for v in d:
            if re.match('var',v):
                n[v] = d[v]/norm
            else:
                n[v] = d[v]
        return n

    def __calcnorm(self, linsol):
        """Norm der Variationen.
        """
        norm = 0
        for t in range(len(linsol['t'])):
            for v in linsol:
                if re.match('var',v):
                    norm += linsol[v][t]**2
        return sqrt(norm)/self.resolution
        
    def calcmsf(self, alpha, beta, transients=5, final=4000):
        """MSF ausrechnen.
        """
        le = 0
        self.params['alpha'] = alpha
        self.params['beta'] = beta

        system = neteqns(self.groups, self.H, self.T, self.eqns, params=self.params, coupling=self.coupling)
        system.update(neteqns(self.groups, self.H, self.T, self.vareqns, params=self.params, coupling=self.coupling, var=True))

#        print system # debug

        for i in range(final):
            sdde = dde23(eqns=system, params=self.params, debug=False)
            sdde.set_sim_params(tfinal=self.params['tau'])
            if i==0:
                sdde.hist_from_arrays(self.__fillinitdict(self.transient, sdde.hist))
            else:
                norm = self.__calcnorm(linsol)
                if i > transients:
                    le += log(norm)
                sdde.hist_from_arrays(self.__renormalized(linsol, norm))
            sdde.run()
            sol = sdde.sol
            linsol = sdde.sol_spl(self.resarray)
        exponent = le/((i-transients)*self.params['tau'])
        return exponent.real


class network:
    """Ein network Objekt nimmt als Argument eine quadratische Matrix.
       Zusaetzlich zu u.g. Methoden gibt es noch diese Variablen

       matrix         Die Adjacency-Matrix des Netzwerks
       dim            Die Dimension des Netzwerks
       allpathes      Liste aller Pfade im Netzwerk
       pathesbylength Liste der Listen der Pfade einer Pfadlaenge, beginnend bei Pfadlaenge = 1 (intern 0)
       TODO:
       - Cachen der globalen Properties in network fuer mehr Speed (fast fertig)
       - Pfadlaenge inf wenn kein Pfad der Laenge <= N-1 zu finden ist
       - Normierung des lokalen Clustering Koeff. checken (in localcc)
    """

    def __init__(self, matrix):
        self.epsilon = 1e-10
        """Um zu entscheiden, wann eine Verbindung da ist, das ist ein workaround fuer !=0 bei floats
        """
        self.matrix = matrix
        self.dim = len(self.matrix)
        """Die Dimension des Netzwerks.
        """
        self.allpathes = []
        self.pathesbylength = self.__findpathesbylength()
        self.pathesbyindex = []
        self.components = []
        self.traces = []

    def __findpathesbylength(self, alle=False):
        """Findet alle Pfade der Laenge nach sortiert. Wichtig fuer alles andere.
           TODO: Cachen von pathesbylength, so dass nur diese Funktion gebraucht wird.
        """
        self.allpathes = []
        lange = [ [] for i in range(self.dim) ]
        ## TODO: in lange(N-1) sollen die Pfade der Laenge inf gespeichert werden!
        for l in range(self.dim): ## N ist die maximale Pfadlaenge fuer N=1, sonst N-1, aber das sollte keine Rolle spielen.
            ## Erstmal Pfade der Laenge 1 finden, das ist leicht
            if l==0:
                for k in range(self.dim):
                    for i in range(self.dim):
                        ## Wenn k ein Nachbar von i ist...
                        if k in self.outneighbors(i):
                            ## ... dann ist das ein Pfad der Laenge 1
                            lange[0].append([i,k])
            ## Fuer Pfade der Laenge >1
            else:
                for k in range(self.dim):
                    for i in range(self.dim):
                        ## Nur wenn es noch keinen Pfad der Laenge <l gibt, soll auch nach Pfaden der Laenge l gesucht werden.
                        if alle or [i,k] not in self.allpathes:
                             ## m sind die Nachbarn von i
                            for m in self.outneighbors(i):
                                ## Wenn zwischen den Nachbarn von i (also m) und k einen Pfad der Laenge l-1 gibt, so gibt es einen Pfad der Laenge l zwischen k und i. Ausserdem ausschliessen, dass mehrere Pfade gespeichert werden.
                                if alle or [m,k] in lange[l-1] and [i,k] not in lange[l]:
                                    lange[l].append([i,k])
            for i in range(len(lange[l])): ## Pfade, die schon gefunden wurden, updaten.
                self.allpathes.append(lange[l][i])
        ## Zurueckgegeben wird die Liste der Pfade
        return lange

    def allpathesbylength(self):
        """Alle Pfade, nicht nur die kuerzesten.
        """
        return self.__findpathesbylength(alle=True)

    def __findpathlengthdist(self):
        """Die Verteilung der Pfadlaengen.
        """
        return [ len(self.pathesbylength[i]) for i in range(self.dim) ]
    
    def __findpathesbyindex(self):
        """Die Pfade nach Index sortiert.
           TODO: Funktioniert der Algorithmus, wenn Pfade der Länge inf vorkommen,
                 die ja dann nicht in pathesbylength stehen?
        """
        allepfade = []
        for i in range(self.dim):
            for k in range(len(self.pathesbylength[i])):
                allepfade.append([self.pathesbylength[i][k],i])
        self.pathesbyindex = sorted(allepfade)

    def outneighbors(self, i):
        """Alle Nachbarn von i, d.h. alle k, fuer die es einen Link i->k gibt.
           Zurueckgegeben als Liste.
        """
        liste = []
        for k in range(self.dim):
            if abs(self.matrix[i][k]) > self.epsilon:
                liste.append(k)
        return liste

    def inneighbors(self, i):
        """Knoten, von denen i Nachbar ist in einer Liste,
           d.h. alle k, fuer die es einen Link k->i gibt.
        """
        liste = []
        for k in range(self.dim):
            if abs(self.matrix[k][i]) > self.epsilon:
                liste.append(k)
        return liste

    def allneighbors(self, i):
        """Die Vereinigung von In- und Out-Nachbarn eines Knotens.
        """
        return sorted(np.union1d(self.outneighbors(i),self.inneighbors(i)))

    def meanpathlength(self, alternative=False):
        """Die mittlere Pfadlaenge im ganzen Netz.
           Bei altenative=True wird die alternative reziproke Definition benutzt.
           Sinnvoll, wenn Pfadlängen unendlich vorkommen.
        """
        mittlerepfadlaenge = 0.0
        pfadlaengen = self.__findpathlengthdist()
        for i in range(len(pfadlaengen)):
            if alternative:
                mittlerepfadlaenge += 1.0/(pfadlaengen[i]*(i+1))
            else:
                mittlerepfadlaenge += pfadlaengen[i]*(i+1)                
        mittlerepfadlaenge /= self.dim**2
        if alternative:
            return 1.0/mittlerepfadlaenge
        else:
            return mittlerepfadlaenge

    def pathlength(self, i, k):
        """Pfadlaenge von i nach k.
        """
        if self.pathesbyindex == []:
            self.__findpathesbyindex()
        if [i,k] not in self.allpathes:
            return np.inf
        for m in range(len(self.pathesbyindex)):
            if self.pathesbyindex[m][0]==[i,k]:
                return self.pathesbyindex[m][1]+1

    def closeness(self, i):
        """Closeness des Knotens i
           TODO: Macht Closeness Sinn, wenn Pfadlängen inf auftauchen?
        """
        abstandsumme = 0
        if self.pathesbyindex == []:
            self.__findpathesbyindex()
        for k in range(self.dim):
            # if i!=k: ## Wird nur gebraucht, wenn Feedback hier keinen Beitrag liefern soll!
            if [i,k] not in self.allpathes:
                abstandsumme += np.inf
                break
            for m in range(len(self.pathesbyindex)):
                if self.pathesbyindex[m][0]==[i,k]:# or allepfade[m][0]==[k,i]:
                    abstandsumme += self.pathesbyindex[m][1]+1 # plus 1 weil die kuerzeste Pfadlaenge hier 0 ist, nicht 1.
        return 1.0/(abstandsumme/float(self.dim)) # Die Normierung ist noch falsch

    def outdegree(self, i):
        """Out-Degree des Knotens i.
        """
        grad = 0
        for k in range(self.dim):
            if abs(self.matrix[i][k]) > self.epsilon:
                grad += 1
        return grad

    def indegree(self, i):
        """In-Degree des Knotens i
        """
        grad = 0
        for k in range(self.dim):
            if abs(self.matrix[k][i]) > self.epsilon:
                grad += 1
        return grad

    def degree(self, i):
        """Mittel aus In- und Out-Degree des Knotens i.
           Fuer ungerichtete Netze ist indegree(i) = outdegree(i) = degree(i).
        """
        return np.average([self.indegree(i),self.outdegree(i)])

    def localcc(self, i):
        """Clustering coefficient des Knotens i.
        """
        allenachbarn = self.allneighbors(i)
        linkszwischennb = 0
        for k in range(len(allenachbarn)):
            for m in range(len(allenachbarn)):
                if abs(self.matrix[k][m]) > self.epsilon:
                    linkszwischennb += 1
        # Wenn es nur einen oder keinen Nachbarn gibt, muss der CC null sein. Das muss nochmal ueberprueft werden.
        if len(allenachbarn) <= 1:
            return 0.
        else:
            return linkszwischennb/float(len(allenachbarn)*(len(allenachbarn)-1))

    def cc(self):
        """Mittlerer Clustering coefficient des Netzwerks.
           Mittel ueber alle lokalen Clustering Koeffizienten.
        """
        return np.average([ self.localcc(i) for i in range(self.dim) ])

    def __findcomponents(self):
        """Liste der Listen der Elemente aller Komponenten des Netzwerks.
        """
        cluster = [ [0] ]
        for i in range(1,self.dim):
            gefunden = False
            for k in range(len(cluster)):
                for m in range(len(cluster[k])):
                    if [i,cluster[k][m]] in self.allpathes:# or [cluster[k][m],i] in self.allpathes: #Brauchen wir das?
                        print [i,cluster[k][m]] #Debug
                        cluster[k].append(i)
                        gefunden = True
                        break
                if gefunden:
                    break
            if not gefunden:
                cluster.append([i])
        self.components = cluster

    def giantcomp(self):
        """Liste der Elemente der groessten Komponente des Netzwerks.
        """
        if self.components == []:
            self.__findcomponents()
        groessen = map(len,self.components)
        return self.components[np.argmax(groessen)]

    def giantcompsize(self):
        """Anzahl der Elemente in der groessten Komponente des Netzwerks.
        """
        return len(self.giantcomp())

    def giantmatrix(self):
        """Adjacency Matrix fuer die groesste Komponente des Netzwerks.
           Praktisch, um damit ein neues Netzwerk zu erzeugen, z.B.
           unternetzwerk = netpy.netzwerk(netpy.netzwerk(G).giantmatrix())
        """
        if self.components == []:
            self.__findcomponents()
        elem = self.giantcomp()
        N = len(elem)
        K = np.zeros((N,N))
        for i in range(N):
            for k in range(N):
                K[i][k] = self.matrix[elem[i]][elem[k]]
        return K

    def __findtraces(self):
        """Findet alle Pfade inkl. Zwischenstationen.
           Nützlich für die Betweenness.
           Ähnlich zu __findpathesbylength(), aber doch zu anders,
           also hier in einer neuen Funktion.
           TODO: Dokumentieren, bevor wir nicht mehr wissen, wie es geht.
        """
        allepfade = []
        lange = [ [] for i in range(self.dim) ]
        traces =  [ [] for i in range(self.dim) ]
        for l in range(self.dim):
            if l==0:
                for k in range(self.dim):
                    for i in range(self.dim):
                        if k in self.outneighbors(i):
                            lange[0].append([i,k])
                            traces[0].append(np.concatenate(([i,],[k,])))
            else:
                for k in range(self.dim):
                    for i in range(self.dim):
                        if [i,k] not in allepfade:
                            for m in self.outneighbors(i):
                                if [m,k] in lange[l-1]:
                                    if [i,k] not in lange[l]:
                                        lange[l].append([i,k])
                                    for p in range(len(traces[l-1])):
                                        if traces[l-1][p][0]==m and traces[l-1][p][-1]==k:
                                            traces[l].append(np.concatenate(([i,], traces[l-1][p])))
            for i in range(len(lange[l])):
                allepfade.append(lange[l][i])
        return traces

    def betweenness(self, i):
        """Betweenness des i-ten Knotens.

           Geht nach folgender Definition:
           - Sei $n_{jk}$ die Zahl der kürzesten Pfade, die
             die Knoten $j$ und $k$ verbindet.
           - Sei $n_{jk}(i)$ die Zahl dieser Pfade, die durch
             den Knoten $i$ laufen.
           Dann ist die Betweenness $b_i$ des $i$-ten Knotens:
           \begin{equation*}
             b_i = \sum_{j \neq k} \frac{ n_{jk}(i) }{ n_{jk} }
           \end{equation*}
        
           TODO: Dokumentieren, bevor wir nicht mehr wissen, wie es geht.
        """
        if self.traces == []:
            self.traces = self.__findtraces()
#        print self.traces # Debug
        summe = 0.0
        for k in range(self.dim):
            for m in range(self.dim):
                if k!=m:# and k!=i and m!=i: #Einkommentieren, wenn Feedback-Pfade keine Rolle spielen sollen.
                    zaehler = 0
                    nenner = 0
                    match = False
                    for n in range(len(self.traces)):
                        for p in range(len(self.traces[n])):
                            if (self.traces[n][p][0]==k and self.traces[n][p][-1]==m):# or (self.traces[n][p][0]==m and self.traces[n][p][-1]==k): #Noch unklar, ob wir das brauchen.
#                                print('{i} {m} {k} {trace}'.format(i=i,m=m,k=k,trace=self.traces[n][p])) # Debug
                                match = True
                                nenner += 1
                                if i in self.traces[n][p]:
                                    zaehler += 1
                        if match:
                            break
#                    print('{z} {n}'.format(z=zaehler,n=nenner)) # Debug
                    if nenner != 0:
                        summe += float(zaehler)/float(nenner)
        return summe


def inhibitory_part(G):
    """Ersetzt in einer Matrix alle exhibitorischen Elemente (G[i][k]>0) durch Nullen.
       Praktisch, um nur daraus ein Unternetzwerk zu betrachten, z.B.
       netzwerk = netpy.network(inhibitory_part(G))
    """
    N = len(G)
    new = np.zeros((N,N))
    for i in range(N):
        for k in range(N):
            if G[i][k] < 0:
                new[i][k] = G[i][k]
    return new

def rowsum(G,i):
    """Zeilensumme der i-ten Zeile einer Matrix
    """
    N = len(G)
    zs = 0.0
    for k in range(N):
        zs += G[i][k]
    return zs

def scalefree_matrix_noself(N, directed=True):
    """Standard Albert-Barabasi scale-free Netzwerk
    """
    def sf_probability(G,i):
        N = len(G)
        summe = 0.0
        for k in range(N):
            summe += rowsum(G,k)
        return rowsum(G,i)/summe

    import random
    random.seed()

    G = np.zeros((N,N))
    """Initial nodes: erstmal nur einer ungleich 0
    """
    probability = [ 0. for i in range(N) ]
    G[0][1] = 1.
    for i in range(1,N):
        for k in range(N):
          if i != k:
            probability[k] = sf_probability(G,k)
            if random.random() < probability[k]:
                G[i][k]=1.0;
                if not directed:
                    G[k][i]=1.0
            if directed and random.random() < probability[k]:
                G[k][i]=1.0
    return G

def scalefree_matrix(N, directed=True):
    """Standard Albert-Barabasi scale-free Netzwerk
    """
    def sf_probability(G,i):
        N = len(G)
        summe = 0.0
        for k in range(N):
            summe += rowsum(G,k)
        return rowsum(G,i)/summe

    import random
    random.seed()

    G = np.zeros((N,N))
    """Initial nodes: erstmal nur einer ungleich 0
    """
    probability = [ 0. for i in range(N) ]
    G[0][0] = 1.
    for i in range(1,N):
        for k in range(N):
            probability[k] = sf_probability(G,k)
            if random.random() < probability[k]:
                G[i][k]=1.0;
                if not directed:
                    G[k][i]=1.0
            if directed and random.random() < probability[k]:
                G[k][i]=1.0
    return G

def random_matrix_noself(N, p, directed=True):
    """Erdos-Renyi random Netzwerk
    """
    import random
    random.seed()
    G = np.zeros((N,N))
    for i in range(N):
        if directed:
            start = 0
        else:
            start = i+1
        """ Will man Feedback ausschliessen, muss start = i+1 sein.
        """
        for k in range(start,N):
          if i != k:
            if random.random() < p:
                G[i][k]=1.0
		if not directed:
                    G[k,i]=1.0
    return G
    
def random_matrix(N, p, directed=True):
    """Erdos-Renyi random Netzwerk
    """
    import random
    random.seed()
    G = np.zeros((N,N))
    for i in range(N):
        if directed:
            start = 0
        else:
            start = i
        """ Will man Feedback ausschliessen, muss start = i+1 sein.
        """
        for k in range(start,N):
            if random.random() < p:
                G[i][k]=1.0
		if not directed:
                    G[k,i]=1.0
    return G

def exponential(N, kappa):
    """ Exponential decay with A normalized to 1 in interval [0:1]
        G(i-j)=A*exp(-kappa|i-j|)
    """    
    G = np.zeros((N,N))
    for i in range(N):
        for l in range(i,i+N/2+1):
            m= l % N
            G[i][m]=kappa/(2.*(1.-math.exp(-kappa/2.)))*math.exp(-kappa*math.fabs((l-i)/N))
            #G[i][m]=kappa/(2.*(1.-math.exp(-kappa/2.)))*math.exp(-kappa*math.fabs(float(l-i)/N))
        for l in range(i-N/2,i):
            n= l % N
            G[i][n]=kappa/(2.*(1.-math.exp(-kappa/2.)))*math.exp(-kappa*math.fabs((l-i)/N))
            #G[i][n]=kappa/(2.*(1.-math.exp(-kappa/2.)))*math.exp(-kappa*math.fabs(float(l-i)/N))
        #for i in range(N):
	  #G[i][i] = 0
    return G

def neighborhood_normalized(N, k, unidirectional=0):
    """Die Basis eines Small-World Netzwerks: Ring mit k nächsten
       Nachbarn nach rechts und links.

       - feedback sagt, was auf der Diagonalen steht (default 0).
    """
    G = np.zeros((N,N))
    for i in range(N):
        G[i][i] = 1/(2*k+1)
        for l in range(i+1,i+k+1):
            m= l % N
            n=(l+(N-1)-k) % (N)
            if not unidirectional == 1:
	        G[i][m]=1/(2*k+1)
	    if not unidirectional == -1:
                G[i][n]=1/(2*k+1)
    return G


def neighborhood(N, k, feedback=0, unidirectional=0):
    """Die Basis eines Small-World Netzwerks: Ring mit k nächsten
       Nachbarn nach rechts und links.

       - feedback sagt, was auf der Diagonalen steht (default 0).
    """
    G = np.zeros((N,N))
    for i in range(N):
        G[i][i] = feedback
        for l in range(i+1,i+k+1):
            m= l % N
            n=(l+(N-1)-k) % (N)
            if not unidirectional == 1:
	        G[i][m]=1
	    if not unidirectional == -1:
                G[i][n]=1
    return G

def addwires(G, p, k, inh=True):
    """Zusätzliche Links ins Netz einbauen. Gleiches kann aber rewire(add=True).
       Wenn rewire() gleiche Ergebnisse liefert, können wir addwires() löschen.
    """
    import random
    random.seed()

    N = len(G)
    Lang = int(round(p*k*N))  #Anzahl der zusaetzlichen Verbindungen, die eingefuegt werden sollen  
    for i in range(Lang): #Also ueber all zuverteilenden Verbindungen
     	nichtgefunden = True #passende Verbindung gefunden
	while nichtgefunden: # Ist das nicht langsam für große k?
		von=random.randint(0, N-1)  # welche Knoten verknuepfen
		bis=random.randint(0, N-1)  
		if von != bis and  G[von][bis] == 0:
		    nichtgefunden = False
	   	    if inh:
	      	       G[von][bis] = -1 
	               G[bis][von] = -1
	   	    else:
	      		G[von][bis] = 1 
	      		G[bis][von] = 1   
    return G

def rewire(G, p, inh=True, directed=False, add=False, excludedoubles=True):
    """Vorhandene Reguläre Links rewiren.  Dabei soll hier k nicht
       nötig sein und Feedback egal sein. Also: erst die Links, die
       schon da sind, zählen und speichern, dann mit p die selektieren
       die rewired werden sollen und dies tun.

       Achtung: Unterschied zu vorher bzw. zu addwires: Dort werden
       immer genau p*k*N Links rewired, hier gilt das nur im Mittel
       über viele Realisationen.

       - inh            macht -1, sonst 1
       - directed       macht ein gerichtete Rewires.
       - add            löscht nicht die alten Links.
       - excludedoubles schaut, dass nicht zwei neue Links gleich sind.
    """
    import random
    random.seed()

    if inh:
        vz = -1
    else:
        vz = 1

    # Erstmal schauen, was da ist.
    vorher = []
    N = len(G)
    for i in range(N):
        if directed:
            krange = range(N) # die ganze Matrix
        else:
            krange = range(i+1,N) # nur ein Dreieck der Matrix
        for k in krange:
            if G[i][k] != 0:
                vorher.append([i,k]) # vorhandene Links sammeln
    
    # Und rewiren.
    nachher = []
    for link in vorher:
        if random.random() < p:
            von = 0
            bis = 0
            while von == bis or [von,bis] in vorher or (not directed and [bis,von] in vorher) or [von,bis] in nachher: # solange es den schon gibt, weiter nach nicht vorhandenen Links suchen.
                von = random.randint(0, N-1)
                bis = random.randint(0, N-1)
            if not add: # Alte Links löschen (rewire) oder nicht.
                G[link[0]][link[1]] = 0
                if not directed:
                    G[link[1]][link[0]] = 0
            G[von][bis] += vz
            if excludedoubles:
                nachher.append([von,bis]) # Den kennen wir dann schon.
            if not directed:
                G[bis][von] += vz
                if excludedoubles:
                    nachher.append([bis,von]) # Und den auch.
    return G

def small_world(N,k,p, inh=True, directed=False, add=True, excludedoubles=True):
    """Small-world Netzwerke, default-maessig zusaetzliche langreichweitige Verbindungen inhibitorisch.
       Kein Rewiren. Netzwerk ist per default nicht directed.
    """
    G = neighborhood(N,k)
    #G = addwires(G, p, k, inh=inh) # geht auch noch, aber rewire() kann mehr.
    G = rewire(G, p, inh=inh, directed=directed, add=add, excludedoubles=excludedoubles)
    return G

def renormalized(G):
    """Macht konstante Zeilensumme, indem es in jeder Zeile durch diese teilt.
    """
    N = len(G)
    for i in range(N):
        G[i] /= rowsum(G,i)
    return G

def renormalized_bysubtraction(G, rest=0):
    """Macht konstante Zeilensumme, indem es in jeder Zeile (Zeilensumme - rest) abzieht.
       rest ist dann die Zeilensumme, die uebrig bleibt.
       Nuetzlich z.B. bei scale-free Netzen, in denen eine Zeile schon mal komplett null
       sein kann und daher eine Normierung durch Division nicht moeglich ist.
    """
    N = len(G)
    for i in range(N):
        G[i][i] -= rowsum(G,i) - rest
    return G

def prettyprint(G):
    """Gibt eine Matrix als durch Leerzeichen separierte Tabelle aus.
       Praktisch zur besseren Lesbarkeit und zur weiteren Verwendung in anderen Programmen.
    """
    N = len(G)
    for i in range(N):
        for k in range(N):
            print "%g" % G[i][k],
        print

def delaymatrix_linear(N, tau, delta=0., periodic=False):
    """Gibt eine Matrix von verschiedenen Delays aus, wobei eine lineare Chain-
       (periodic=False) oder Ring-Struktur (periodic=True) des Raums auf dem die Knoten
       leben, vorausgesetzt wird.
       - tau skaliert alle.
       - delta ist ein Offset (Latency) für alle.
       (siehe moredelays.tex in /home/dahms/hg/netzwerk)
    """
    T = np.zeros((N,N))
    for i in range(N):
        for k in range(N):
            if periodic:
                T[i][k] = tau * min(abs(k-i), N-abs(k-i)) + delta
            else:
                T[i][k] = tau * abs(k-i) + delta
    return T

def delaymatrix_from_positions(P, tau, delta=0.):
    """Gibt eine Matrix mit Delays, wobei diese aus den Abständen der Knoten berechnet
       werden, gegeben durch einen N-dim. Vektor P, in dem 1-, 2-, oder 3-dim. Positionen
       stecken.
       - tau skaliert alle.
       - delta ist ein Offset (Latency) für alle.
    """
    N = len(P)
    T = np.zeros((N,N))
    for i in range(N):
        for k in range(N):
            T[i][k] = tau * np.linalg.norm(P[i]-P[k]) + delta
    return T

def neteqns(G, H, T, eqns_solitary, params, coupling, var=False):
    """Macht aus einer Gleichung ganz viele.
       Zum Simulieren.
    """
    eqns_net=dict()
    N = len(G)
    vars = sorted(eqns_solitary.keys())
    for i in range(N):
        for v,V in enumerate(vars):
            eqns_net[V.format(i=str(i))] = eqns_solitary[V].format(i=str(i))
            for k in range(N):
                if G[i][k] != 0:
                    for w,W in enumerate(vars):
                        if H[v][w] != 0:
                            if var == False:
                                fmt = { 'G' : float(G[i][k]),
                                        'H' : float(H[v][w]),
                                        'sigma' : float(params['sigma']),
                                        'var' : re.sub(':[cC]','',W.format(i=str(k))),
                                        'tau' : float(T[i][k]),
					'self' : re.sub(':[cC]','',W.format(i=str(i))),
                                        }

                            else:
                                fmt = { 'G' : '({alpha}+ii*{beta})'.format(alpha=float(params['alpha']),
                                                                           beta=float(params['beta'])),
                                        'H' : float(H[v][w]),
                                        'sigma' : 1.0,
                                        'var' : re.sub(':[cC]','',W.format(i=str(k))),
                                        'tau' : float(T[i][k]),
					 'self' : re.sub(':[cC]','',W.format(i=str(i))),
                                        }
                            eqns_net[V.format(i=str(i))] += coupling.format(**fmt)
    return eqns_net

def netnoise(eqns_net, noise_solitary):
    """Noise für alle!
       Zum Simulieren.
    """
    noise_net = dict()
    for v in eqns_net:
        for w in noise_solitary:
            if re.match(w,v):
                noise_net[re.sub(':[cC]','',v)] = noise_solitary[w]
    return noise_net


if __name__ == '__main__':

    """Ein Beispiel. Wird gerechnet, falls die Datei direkt aufgerufen wird.
    """

    """
    import sys

    try:
        G = np.loadtxt(sys.argv[1])
    except:
        print("Matrix file not given or not good.")
        sys.exit(1)
    netzwerk = network(network(G).giantmatrix())
    """

    directed=False
    netzwerk = network(scalefree_matrix(25,directed))
    print netzwerk.giantcomp()

#    directed=True
#    netzwerk = network(network(random_matrix(25,0.8,directed)).giantmatrix())

#    prettyprint(netzwerk.matrix)
#    laengen = [ [ netzwerk.pathlength(i,k) for i in range(netzwerk.dim) ] for k in range(netzwerk.dim) ]
#    prettyprint(laengen)
#    print netzwerk.allpathes

    netzwerk=network(netzwerk.giantmatrix())

    prettyprint(netzwerk.matrix)
#    laengen = [ [ netzwerk.pathlength(i,k) for i in range(netzwerk.dim) ] for k in range(netzwerk.dim) ]
#    prettyprint(laengen)
#    print netzwerk.allpathes



#    print netzwerk.pathlength(4,0)
    for i in range(netzwerk.dim):
        print("{0} {1}".format(netzwerk.betweenness(i), netzwerk.degree(i)))


"""    #netzwerk = network(M)
    netzwerk = small_world(10, 3, 0.1)
    prettyprint(netzwerk)
    
    #print netzwerk.giantcomp()
"""

"""
    print("# (Matrix) (Network's dimension) (Mean path length) (Mean Clustering coefficient) (Mean closeness)")
    print("{0} {1:4} {2:6f} {3:6f} {4:6f}"
          .format(sys.argv[1],
                  netzwerk.dim,
                  netzwerk.meanpathlength(),
                  netzwerk.cc(),
                  np.average([ netzwerk.closeness(i)
                               for i in range(netzwerk.dim) ])))
"""



