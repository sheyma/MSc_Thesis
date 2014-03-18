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
Density_Ra = load('A_Ra_edges_and_density.dat');
Density_Rb = load('A_Rb_edges_and_density.dat');
Density_Rc = load('A_Rc_edges_and_density.dat');
Density_Rd = load('A_Rd_edges_and_density.dat');


R = Density(:,1);   
D = Density(:,3);   
L = Density(:,2);   

R_Ra = Density_Ra(:,1);
D_Ra = Density_Ra(:,3);
L_Ra = Density_Ra(:,2);

R_Rb = Density_Rb(:,1);
D_Rb = Density_Rb(:,3);
L_Rb = Density_Rb(:,2);

R_Rc = Density_Rc(:,1);
D_Rc = Density_Rc(:,3);
L_Rc = Density_Rc(:,2);

R_Rd = Density_Rd(:,1);
D_Rd = Density_Rd(:,3);
L_Rd = Density_Rd(:,2);

figure(1);
subplot(1,2,1)
set(gca,'FontSize',20)
hold on
plot(R,D,'k','LineWidth',3)
plot(R_Ra,D_Ra,'ob','LineWidth',2)
plot(R_Rb,D_Rb,'og','LineWidth',2)
plot(R_Rc,D_Rc,'or','LineWidth',2)
plot(R_Rd,D_Rd,'oy','LineWidth',2)
legend('Test Network', 'Randomization a','Randomization b','Randomization c','Randomization d')
legend('boxoff')
xlabel('Threshold [r]')
ylabel('Network Density')
hold off
subplot(1,2,2)
set(gca,'FontSize',20)
hold on
plot(R,L,'k','LineWidth',3)
plot(R_Ra,L_Ra,'ob','LineWidth',2)
plot(R_Rb,L_Rb,'og','LineWidth',2)
plot(R_Rc,L_Rc,'or','LineWidth',2)
plot(R_Rd,L_Rd,'oy','LineWidth',2)
legend('Test Network', 'Randomization a','Randomization b','Randomization c','Randomization d')
legend('boxoff')
xlabel('Threshold [r]')
ylabel('Number of Edges [L]')
hold off


% Average degree
Degree = load('A_degree_ave.dat');
R = Degree(:,1);
Deg_ave = Degree(:,2);

Degree_Ra =load('A_Ra_degree_ave.dat');
R_Ra = Degree_Ra(:,1);
Deg_ave_Ra = Degree_Ra(:,2);

Degree_Rb =load('A_Rb_degree_ave.dat');
R_Rb = Degree_Rb(:,1);
Deg_ave_Rb = Degree_Rb(:,2);

Degree_Rc =load('A_Rc_degree_ave.dat');
R_Rc = Degree_Rc(:,1);
Deg_ave_Rc = Degree_Rc(:,2);

Degree_Rd =load('A_Rd_degree_ave.dat');
R_Rd = Degree_Rd(:,1);
Deg_ave_Rd = Degree_Rd(:,2);


figure(2);
set(gca,'FontSize',15)
hold on
plot(R,Deg_ave,'k','LineWidth',3)
plot(R_Ra,Deg_ave_Ra,'ob','LineWidth',2)
plot(R_Rb,Deg_ave_Rb,'og','LineWidth',2)
plot(R_Rc,Deg_ave_Rc,'or','LineWidth',2)
plot(R_Rd,Deg_ave_Rd,'oy','LineWidth',2)
xlabel('Threshold [r]')
ylabel('Average Degree')
legend('Test Network', 'Randomization a','Randomization b','Randomization c','Randomization d')
legend('boxoff')
hold off


% Average cluster coefficient
Coef = load('A_cluster_coeffi_ave.dat');
R = Coef(:,1);
cc = Coef(:,2);

Coef_Ra = load('A_Ra_cluster_coeffi_ave.dat');
R_Ra = Coef_Ra(:,1);
cc_Ra = Coef_Ra(:,2);

Coef_Rb = load('A_Rb_cluster_coeffi_ave.dat');
R_Rb = Coef_Rb(:,1);
cc_Rb = Coef_Rb(:,2);

Coef_Rc = load('A_Rc_cluster_coeffi_ave.dat');
R_Rc = Coef_Rc(:,1);
cc_Rc = Coef_Rc(:,2);

Coef_Rd = load('A_Rd_cluster_coeffi_ave.dat');
R_Rd = Coef_Rd(:,1);
cc_Rd = Coef_Rd(:,2);

figure(3);
set(gca,'FontSize',15)
hold on
plot(R,cc,'k','LineWidth',3)
plot(R_Ra,cc_Ra,'ob','LineWidth',2)
plot(R_Rb,cc_Rb,'og','LineWidth',2)
plot(R_Rc,cc_Rc,'or','LineWidth',2)
plot(R_Rd,cc_Rd,'oy','LineWidth',2)
legend('Test Network', 'Randomization a','Randomization b','Randomization c','Randomization d')
legend('boxoff')
xlabel('Threshold [r]')
ylabel('Average Cluster Coefficient')
hold off

% Connected Components of Network
Con_com = load('A_connected_compo.dat');
R = Con_com(:,1);
Con_comp = Con_com(:,2);

Con_com_Ra = load('A_Ra_connected_compo.dat');
R_Ra = Con_com_Ra(:,1);
Con_comp_Ra = Con_com_Ra(:,2);

Con_com_Rb = load('A_Rb_connected_compo.dat');
R_Rb = Con_com_Rb(:,1);
Con_comp_Rb = Con_com_Rb(:,2);

Con_com_Rc = load('A_Rc_connected_compo.dat');
R_Rc = Con_com_Rc(:,1);
Con_comp_Rc = Con_com_Rc(:,2);

Con_com_Rd = load('A_Rd_connected_compo.dat');
R_Rd = Con_com_Rd(:,1);
Con_comp_Rd = Con_com_Rd(:,2);

figure(4);
set(gca,'FontSize',20)
hold on
plot(R,Con_comp,'k','LineWidth',3)
plot(R_Ra,Con_comp_Ra,'ob','LineWidth',2)
plot(R_Rb,Con_comp_Rb,'og','LineWidth',2)
plot(R_Rc,Con_comp_Rc,'or','LineWidth',2)
plot(R_Rd,Con_comp_Rd,'oy','LineWidth',2)
legend('Test Network', 'Randomization a','Randomization b','Randomization c','Randomization d')
legend('boxoff')
xlabel('Threshold [r]')
ylabel('Connected Components')
hold off

% Shortest Pathway of Network
S = load('A_shortest_path.dat');
R = S(:,1);
shor = S(:,2);

S_Ra = load('A_Ra_shortest_path.dat');
R_Ra = S_Ra(:,1);
shor_Ra = S_Ra(:,2);

S_Rb = load('A_Rb_shortest_path.dat');
R_Rb = S_Rb(:,1);
shor_Rb = S_Rb(:,2);

S_Rc = load('A_Rc_shortest_path.dat');
R_Rc = S_Rc(:,1);
shor_Rc = S_Rc(:,2);

S_Rd = load('A_Rd_shortest_path.dat');
R_Rd = S_Rd(:,1);
shor_Rd = S_Rd(:,2);

figure(5);
set(gca,'FontSize',20)
hold on
plot(R,shor,'k','LineWidth',3)
plot(R_Ra,shor_Ra,'ob','LineWidth',2)
plot(R_Rb,shor_Rb,'og','LineWidth',2)
plot(R_Rc,shor_Rc,'or','LineWidth',2)
plot(R_Rd,shor_Rd,'oy','LineWidth',2)
legend('Test Network', 'Randomization a','Randomization b','Randomization c','Randomization d')
legend('boxoff')
xlabel('Threshold [r]')
ylabel('Shortest Pathway')
hold off

% Global Efficiency of Network
Glo = load('A_global_efficiency_ave.dat');
R = Glo(:,1);
Global = Glo(:,2);

Glo_Ra = load('A_Ra_global_efficiency_ave.dat');
R_Ra = Glo_Ra(:,1);
Global_Ra = Glo_Ra(:,2);

Glo_Rb = load('A_Rb_global_efficiency_ave.dat');
R_Rb = Glo_Rb(:,1);
Global_Rb = Glo_Rb(:,2);

Glo_Rc = load('A_Rc_global_efficiency_ave.dat');
R_Rc = Glo_Rc(:,1);
Global_Rc = Glo_Rc(:,2);

Glo_Rd = load('A_Rd_global_efficiency_ave.dat');
R_Rd = Glo_Rd(:,1);
Global_Rd = Glo_Rd(:,2);

figure(6);
set(gca,'FontSize',15)
hold on
plot(R,Global,'k','LineWidth',3)
plot(R_Ra,Global_Ra,'ob','LineWidth',2)
plot(R_Rb,Global_Rb,'og','LineWidth',2)
plot(R_Rc,Global_Rc,'or','LineWidth',2)
plot(R_Rd,Global_Rd,'oy','LineWidth',2)
legend('Test Network', 'Randomization a','Randomization b','Randomization c','Randomization d')
legend('boxoff')
xlabel('Threshold [r]')
ylabel('Global Efficiency')
hold off


% Local efficiency of Network
Loc = load('A_local_efficency_ave.dat');
R = Loc(:,1);
Local = Loc(:,2);

Loc_Ra = load('A_Ra_local_efficency_ave.dat');
R_Ra = Loc_Ra(:,1);
Local_Ra = Loc_Ra(:,2);

Loc_Rb = load('A_Rb_local_efficency_ave.dat');
R_Rb = Loc_Rb(:,1);
Local_Rb = Loc_Rb(:,2);

Loc_Rc = load('A_Rc_local_efficency_ave.dat');
R_Rc = Loc_Rc(:,1);
Local_Rc = Loc_Rc(:,2);

Loc_Rd = load('A_Rd_local_efficency_ave.dat');
R_Rd = Loc_Rd(:,1);
Local_Rd = Loc_Rd(:,2);


figure(7);
set(gca,'FontSize',15)
hold on
plot(R,Local,'k','LineWidth',3)
plot(R_Ra,Local_Ra,'ob','LineWidth',2)
plot(R_Rb,Local_Rb,'og','LineWidth',2)
plot(R_Rc,Local_Rc,'or','LineWidth',2)
plot(R_Rd,Local_Rd,'oy','LineWidth',2)
legend('Test Network', 'Randomization a','Randomization b','Randomization c','Randomization d')
legend('boxoff')
xlabel('Threshold [r]')
ylabel('Local Efficiency')
hold off

% Tansitivity and Small Worldness of Network
sma = load('A_small_worldness.dat');
R = sma(:,1);
small = sma(:,8);
Trans = sma(:,6);

sma_Ra = load('A_Ra_small_worldness.dat');
R_Ra = sma_Ra(:,1);
small_Ra = sma_Ra(:,8);
Trans_Ra = sma_Ra(:,6);

sma_Rb = load('A_Rb_small_worldness.dat');
R_Rb = sma_Rb(:,1);
small_Rb = sma_Rb(:,8);
Trans_Rb = sma_Rb(:,6);


sma_Rc = load('A_Rc_small_worldness.dat');
R_Rc = sma_Rc(:,1);
small_Rc = sma_Rc(:,8);
Trans_Rc = sma_Rc(:,6);

sma_Rd = load('A_Rd_small_worldness.dat');
R_Rd= sma_Rd(:,1);
small_Rd = sma_Rd(:,8);
Trans_Rd = sma_Rd(:,6);

figure(8);
set(gca,'FontSize',15)
hold on
plot(R,Trans,'k','LineWidth',3)
plot(R_Ra,Trans_Ra,'ob','LineWidth',2)
plot(R_Rb,Trans_Rb,'og','LineWidth',2)
plot(R_Rc,Trans_Rc,'or','LineWidth',2)
plot(R_Rd,Trans_Rd,'oy','LineWidth',2)
legend('Test Network', 'Randomization a','Randomization b','Randomization c','Randomization d')
legend('boxoff')
xlabel('Threshold [r]')
ylabel('Transititvity')
hold off
 
figure(9)
set(gca,'FontSize',15)
hold on
plot(R,small,'k','LineWidth',3)
plot(R_Ra,small_Ra,'ob','LineWidth',2)
plot(R_Rb,small_Rb,'og','LineWidth',2)
plot(R_Rc,small_Rc,'or','LineWidth',2)
plot(R_Rd,small_Rd,'oy','LineWidth',2)
legend('Test Network', 'Randomization a','Randomization b','Randomization c','Randomization d')
legend('boxoff')
xlabel('Threshold [r]')
ylabel('Small Worldness')
hold off

% Degree Distribution
deg_dis = load('A_degree_dist.dat');
deg_dis_Ra = load('A_Ra_degree_dist.dat');
deg_dis_Rb = load('A_Rb_degree_dist.dat');
deg_dis_Rc = load('A_Rc_degree_dist.dat');

figure(10);
subplot(2,3,1)
set(gca, 'FontSize', 15)
tri = delaunay(deg_dis(:,2),deg_dis(:,1));
trisurf(tri,deg_dis(:,2),deg_dis(:,1),deg_dis(:,4));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Degree Distribution')
title('Test Network')
subplot(2,3,2)
set(gca, 'FontSize', 15)
tri = delaunay(deg_dis_Ra(:,2),deg_dis_Ra(:,1));
trisurf(tri,deg_dis_Ra(:,2),deg_dis_Ra(:,1),deg_dis_Ra(:,4));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Degree Distribution')
title('Randomization a')
subplot(2,3,3)
set(gca, 'FontSize', 15)
tri = delaunay(deg_dis_Rb(:,2),deg_dis_Rb(:,1));
trisurf(tri,deg_dis_Rb(:,2),deg_dis_Rb(:,1),deg_dis_Rb(:,4));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Degree Distribution')
title('Randomization b')
subplot(2,3,4)
set(gca, 'FontSize', 15)
tri = delaunay(deg_dis_Rc(:,2),deg_dis_Rc(:,1));
trisurf(tri,deg_dis_Rc(:,2),deg_dis_Rc(:,1),deg_dis_Rc(:,4));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Degree Distribution')
title('Randomization c')

% Degrees of single nodes
single_de = load('A_degree_node.dat');
single_de_Ra = load('A_Ra_degree_node.dat');
single_de_Rb = load('A_Rb_degree_node.dat');
single_de_Rc = load('A_Rc_degree_node.dat');

figure(11);
subplot(2,3,1)
set(gca, 'FontSize', 15)
tri = delaunay(single_de(:,2),single_de(:,1));
trisurf(tri,single_de(:,2),single_de(:,1),single_de(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Degrees of Nodes')
title('Test Network')
subplot(2,3,2)
set(gca, 'FontSize', 15)
tri = delaunay(single_de_Ra(:,2),single_de_Ra(:,1));
trisurf(tri,single_de_Ra(:,2),single_de_Ra(:,1),single_de_Ra(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Degrees of Nodes')
title('Randomization a')
subplot(2,3,3)
set(gca, 'FontSize', 15)
tri = delaunay(single_de_Rb(:,2),single_de_Rb(:,1));
trisurf(tri,single_de_Rb(:,2),single_de_Rb(:,1),single_de_Rb(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Degrees of Nodes')
title('Randomization b')
subplot(2,3,4)
set(gca, 'FontSize', 15)
tri = delaunay(single_de_Rc(:,2),single_de_Rc(:,1));
trisurf(tri,single_de_Rc(:,2),single_de_Rc(:,1),single_de_Rc(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Degrees of Nodes')
title('Randomization c')

% Connected components of single nodes
comp_dist = load('A_connected_compo_node.dat');
comp_dist_Ra = load('A_Ra_connected_compo_node.dat');
comp_dist_Rb = load('A_Rb_connected_compo_node.dat');

figure(12);
subplot(1,3,1)
set(gca, 'FontSize', 15)
tri = delaunay(comp_dist(:,2),comp_dist(:,1));
trisurf(tri,comp_dist(:,2),comp_dist(:,1),comp_dist(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Connected Components')
title('Test Network')
subplot(1,3,2)
set(gca, 'FontSize', 15)
tri = delaunay(comp_dist_Ra(:,2),comp_dist_Ra(:,1));
trisurf(tri,comp_dist_Ra(:,2),comp_dist_Ra(:,1),comp_dist_Ra(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Connected Components')
title('Randomization a')
subplot(1,3,3)
set(gca, 'FontSize', 15)
tri = delaunay(comp_dist_Rb(:,2),comp_dist_Rb(:,1));
trisurf(tri,comp_dist_Rb(:,2),comp_dist_Rb(:,1),comp_dist_Rb(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Connected Components')
title('Randomization b')

% clustering coefficient of sigle nodes
cc_node = load('A_cluster_coeffi_node.dat');
cc_node_Ra = load('A_Ra_cluster_coeffi_node.dat');
cc_node_Rb = load('A_Rb_cluster_coeffi_node.dat');

figure(13);
subplot(1,3,1)
set(gca, 'FontSize', 15)
tri = delaunay(cc_node(:,2),cc_node(:,1));
trisurf(tri,cc_node(:,2),cc_node(:,1),cc_node(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Cluster Coefficients')
title('Test Network')
subplot(1,3,2)
set(gca, 'FontSize', 15)
tri = delaunay(cc_node_Ra(:,2),cc_node_Ra(:,1));
trisurf(tri,cc_node_Ra(:,2),cc_node_Ra(:,1),cc_node_Ra(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Cluster Coefficients')
title('Randomization a')
subplot(1,3,3)
set(gca, 'FontSize', 15)
tri = delaunay(cc_node_Rb(:,2),cc_node_Rb(:,1));
trisurf(tri,cc_node_Rb(:,2),cc_node_Rb(:,1),cc_node_Rb(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Cluster Coefficients')
title('Randomization b')


% global efficiency of single nodes
glo_node = load('A_global_efficiency_node.dat');
glo_node_Ra = load('A_Ra_global_efficiency_node.dat');
glo_node_Rb = load('A_Rb_global_efficiency_node.dat');


figure(14);
subplot(1,3,1)
set(gca, 'FontSize', 15)
tri = delaunay(glo_node(:,2),glo_node(:,1));
trisurf(tri,glo_node(:,2),glo_node(:,1),glo_node(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Global Efficiency')
title('Test Network')
subplot(1,3,2)
set(gca, 'FontSize', 15)
tri = delaunay(glo_node_Ra(:,2),glo_node_Ra(:,1));
trisurf(tri,glo_node_Ra(:,2),glo_node_Ra(:,1),glo_node_Ra(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Global Efficiency')
title('Randomization a')
subplot(1,3,3)
set(gca, 'FontSize', 15)
tri = delaunay(glo_node_Rb(:,2),glo_node_Rb(:,1));
trisurf(tri,glo_node_Rb(:,2),glo_node_Rb(:,1),glo_node_Rb(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Global Efficiency')
title('Randomization b')

% local efficiency of single nodes
loc_node = load('A_local_efficency_node.dat');
loc_node_Ra = load('A_Ra_local_efficency_node.dat');
loc_node_Rb = load('A_Rb_local_efficency_node.dat');

figure(15);
subplot(1,3,1)
set(gca, 'FontSize', 15)
tri = delaunay(loc_node(:,2),loc_node(:,1));
trisurf(tri,loc_node(:,2),loc_node(:,1),loc_node(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Local Efficiency')
title('Test Network')
subplot(1,3,2)
set(gca, 'FontSize', 15)
tri = delaunay(loc_node_Ra(:,2),loc_node_Ra(:,1));
trisurf(tri,loc_node_Ra(:,2),loc_node_Ra(:,1),loc_node_Ra(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Local Efficiency')
title('Randomization a')
subplot(1,3,3)
set(gca, 'FontSize', 15)
tri = delaunay(loc_node_Rb(:,2),loc_node_Rb(:,1));
trisurf(tri,loc_node_Rb(:,2),loc_node_Rb(:,1),loc_node_Rb(:,3));
colorbar;
xlabel('r')
ylabel('Nodes')
zlabel('Local Efficiency')
title('Randomization b')