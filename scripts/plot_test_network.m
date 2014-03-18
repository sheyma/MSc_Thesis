% plot network measures of test network

% A_edges_and_density.dat : #1. threshold, 2. edges, 3. density
% A_degree_ave.dat : #1. threshold, 2. average degree
% A_cluster_coeffi_ave.dat : #1. threshold, 2. clustering coefficient
% A_connected_compo.dat : #1. threshold, 2.number of connected components
% A_shortest_path.dat : # 1.threshold , 2.shortest pathway
% A_global_efficiency_ave.dat : #1.threshold, 2.global efficieny
% A_local_efficency_ave.dat : # 1.threshold, 2.local efficiency
% A_small_worldnes.dat : #1:threshold 2:cluster-coefficient... 
              %...3:random-cluster-coefficient 4:shortest-pathlength 
              %...5:random-shortest-pathlength 6:transitivity 
              %...7:random-transitivity 8:S-Watts-Strogatz 9:S-transitivity
% A_degree_dist.dat : #1.node, 2.threshold, 3.degree hist, 4.degree distr.
% A_degree_node.dat : # 1.node, 2.threshold, 3.degree
% A_connected_compo_node.dat : # 1.node, 2.threshold, 3. connected compon.
% A_cluster_coeffi_node.dat : # node, threshold, clust. coeffi. of node
% A_global_efficiency_node.dat : #1.node, 2,threshold, 3.glo. effi.of node


% network density and egdes
Density = load('A_edges_and_density.dat');
R = Density(:,1);
D = Density(:,3);
L = Density(:,2);

figure(1);
subplot(1,2,1)
set(gca,'FontSize',20)
plot(R,D,'LineWidth',3)
xlabel('Threshold [r]')
ylabel('Network Density')
subplot(1,2,2)
set(gca,'FontSize',20)
plot(R,L,'LineWidth',3)
xlabel('Threshold [r]')
ylabel('Number of Edges [L]')

% Average degree
Degree = load('A_degree_ave.dat');
R = Degree(:,1);
Deg_ave = Degree(:,2);

figure(2);
set(gca,'FontSize',20)
plot(R,D,'LineWidth',3)
xlabel('Threshold [r]')
ylabel('Average Degree')

% Average cluster coefficient
Coef = load('A_cluster_coeffi_ave.dat');
R = Coef(:,1);
cc = Coef(:,2);

figure(3);
set(gca,'FontSize',20)
plot(R,cc,'LineWidth',3)
xlabel('Threshold [r]')
ylabel('Average Cluster Coefficient')

% Connected Components of Network
Con_com = load('A_connected_compo.dat');
R = Con_com(:,1);
Con_comp = Con_com(:,2);

figure(4);
set(gca,'FontSize',20)
plot(R,Con_comp,'LineWidth',3)
xlabel('Threshold [r]')
ylabel('Connected Components')

% Shortest Pathway of Network
S = load('A_shortest_path.dat');
R = S(:,1);
shor = S(:,2);

figure(5);
set(gca,'FontSize',20)
plot(R,shor,'LineWidth',3)
xlabel('Threshold [r]')
ylabel('Shortest Pathway')

% Global Efficiency of Network
Glo = load('A_global_efficiency_ave.dat');
R = Glo(:,1);
Global = Glo(:,2);

figure(6);
set(gca,'FontSize',20)
plot(R,Global,'LineWidth',3)
xlabel('Threshold [r]')
ylabel('Global Efficiency')

% Local efficiency of Network
Loc = load('A_local_efficency_ave.dat');
R = Loc(:,1);
Local = Loc(:,2);

figure(7);
set(gca,'FontSize',20)
plot(R,Local,'LineWidth',3)
xlabel('Threshold [r]')
ylabel('Local Efficiency')

% Tansitivity and Small Worldness of Network
sma = load('A_small_worldness.dat');
R = sma(:,1);
small = sma(:,8);
Trans = sma(:,6);
figure(8);
set(gca,'FontSize',20)
plot(R,Trans,'LineWidth',3)
xlabel('Threshold [r]')
ylabel('Transititvity')
 
figure(9)
set(gca,'FontSize',20)
plot(R,small,'LineWidth',3)
xlabel('Threshold [r]')
ylabel('Small Worldness')

% Degree Distribution
deg_dis = load('A_degree_dist.dat');

figure(10);
set(gca, 'FontSize', 15)
tri = delaunay(deg_dis(:,2),deg_dis(:,1));
trisurf(tri,deg_dis(:,2),deg_dis(:,1),deg_dis(:,4));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Degree Distribution')

% Degrees of single nodes
single_de = load('A_degree_node.dat');
figure(11);
set(gca, 'FontSize', 15)
tri = delaunay(single_de(:,2),single_de(:,1));
trisurf(tri,single_de(:,2),single_de(:,1),single_de(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Degrees of Nodes')

% Connected components of single nodes
comp_dist = load('A_connected_compo_node.dat');

figure(12);
set(gca, 'FontSize', 15)
tri = delaunay(comp_dist(:,2),comp_dist(:,1));
trisurf(tri,comp_dist(:,2),comp_dist(:,1),comp_dist(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Connected Components')

% clustering coefficient of sigle nodes
cc_node = load('A_cluster_coeffi_node.dat');

figure(13);
set(gca, 'FontSize', 15)
tri = delaunay(cc_node(:,2),cc_node(:,1));
trisurf(tri,cc_node(:,2),cc_node(:,1),cc_node(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Cluster Coefficients')

% global efficiency of single nodes
glo_node = load('A_global_efficiency_node.dat');

figure(14);
set(gca, 'FontSize', 15)
tri = delaunay(glo_node(:,2),glo_node(:,1));
trisurf(tri,glo_node(:,2),glo_node(:,1),glo_node(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Global Efficiency')

% local efficiency of single nodes
loc_node = load('A_local_efficency_node.dat');
figure(15);
set(gca, 'FontSize', 15)
tri = delaunay(loc_node(:,2),loc_node(:,1));
trisurf(tri,loc_node(:,2),loc_node(:,1),loc_node(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Local Efficiency')