% plot network measures of test network
% A_network_measures.dat :
%       threshold, L, Density, components, CC_average, check_sum, sum_ave
% A_shortest_pathway.dat :
%       threshold, shortest pahtway
% A_global_efficiency.dat:
%       threshold, gloabal efficiency of full network
% A_local_efficiency.dat :
%       threshold, local efficiency of full network
% A_small_worldness.dat : 
%       1:threshold 2:cluster-coefficient 3:random-cluster-coefficient 
%       4:shortest-pathlength 5:random-shortest-pathlength 6:transitivity 
%       7:random-transitivity 8:S-Watts-Strogatz 9:S-transitivity

% single network measures
Net_mes = load('A_network_measures.dat');
Shor_pat = load('A_shortest_path.dat');
Glo_ef = load('A_global_efficiency.dat');
Loc_ef = load('A_local_efficency.dat');
SWorld = load('A_small_worldness.dat');

% network measures as distributions around nodes
deg_dis = load('A_degree_dist.dat');
node_CC = load('A_node_cc.dat');
comp_dist =load('A_nodes_comp_.dat');
single_de =load('A_single_degrees.dat');

figure(2);
subplot(2,2,1)
set(gca, 'FontSize', 15)
tri = delaunay(deg_dis(:,2),deg_dis(:,1));
trisurf(tri,deg_dis(:,2),deg_dis(:,1),deg_dis(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Degree Distribution')

subplot(2,2,2)
set(gca, 'FontSize', 15)
tri = delaunay(node_CC(:,2),node_CC(:,1));
trisurf(tri,node_CC(:,2),node_CC(:,1),node_CC(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Cluster Coefficients')

subplot(2,2,3)
set(gca, 'FontSize', 15)
tri = delaunay(single_de(:,2),single_de(:,1));
trisurf(tri,single_de(:,2),single_de(:,1),single_de(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Degrees of Nodes')

subplot(2,2,4)
set(gca, 'FontSize', 15)
tri = delaunay(comp_dist(:,2),comp_dist(:,1));
trisurf(tri,comp_dist(:,2),comp_dist(:,1),comp_dist(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Component Dist')

% figure(1)
% 
% subplot(3,3,1)
% set(gca, 'FontSize', 15)
% plot(Net_mes(:,1),Net_mes(:,3))
% xlabel('r')
% ylabel('Network Denstiy (D)')
% 
% subplot(3,3,2)
% set(gca, 'FontSize', 15)
% plot(Net_mes(:,1), Net_mes(:,7))
% xlabel('r')
% ylabel('Average Degree')
% 
% subplot(3,3,3)
% set(gca, 'FontSize', 15)
% plot(Net_mes(:,1), Net_mes(:,5))
% xlabel('r')
% ylabel('Average CC')
% 
% subplot(3,3,4)
% set(gca, 'FontSize', 15)
% plot(Net_mes(:,1), Net_mes(:,4))
% xlabel('r')
% ylabel('Connected Components')
% 
% subplot(3,3,5)
% set(gca, 'FontSize', 15)
% plot(Shor_pat(:,1), Shor_pat(:,2))
% xlabel('r')
% ylabel('Shortest Pathway')
% 
% subplot(3,3,6)
% set(gca, 'FontSize', 15)
% plot(Glo_ef(:,1), Glo_ef(:,2))
% xlabel('r')
% ylabel('Global Efficiency')
% 
% subplot(3,3,7)
% set(gca, 'FontSize', 15)
% plot(Loc_ef(:,1), Loc_ef(:,2))
% xlabel('r')
% ylabel('Local Efficiency')
% 
% subplot(3,3,8)
% set(gca, 'FontSize', 15)
% plot(SWorld(:,1), SWorld(:,8))
% xlabel('r')
% ylabel('Small Worldness')
% 
% subplot(3,3,9)
% set(gca, 'FontSize', 15)
% plot(SWorld(:,1), SWorld(:,7))
% xlabel('r')
% ylabel('Transitivity')