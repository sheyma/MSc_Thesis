% PLOT network measures of test network and randomized networks


% A_*_single_network_measures.dat: #1. threshold, 2. edges, 3. density 
              %4.clustering coefficient, 5. average degree
              %6. number of connected components, 7. shortest pathway
% A_*_small_worldnes.dat : #1:threshold 2:cluster-coefficient... 
              %...3:random-cluster-coefficient 4:shortest-pathlength 
              %...5:random-shortest-pathlength 6:transitivity 
              %...7:random-transitivity 8:S-Watts-Strogatz 9:S-transitivity
% A_*_global_efficiency_ave.dat : #1.threshold, 2.global efficieny
% A_*_local_efficency_ave.dat : # 1.threshold, 2.local efficiency
% A_*_assortativity.dat : # 1. threshold, 2.assortativity coefficient
% A_*_global_efficiency_node.dat : #1.node, 2,threshold, 3.glo. effi.of node  


% A_*_degree_dist.dat : #1.node, 2.threshold, 3.degree hist, 4.degree distr.
% A_*_connected_compo_node.dat : # 1.node, 2.threshold, 3. connected compon.
% A_*_cc_and_degree_node.dat: # 1.node, 2.threshold, 3. clustering coef.
%                                of each node, 4. degree of node

random_G = ('0a');
color='kr'; type = '-o';

% Network Density
input_name = 'acp_w_single_network_measures.dat';
fig = figure(1);
subplot(1,3,1)
hold on
set(gca,'FontSize',45)
%title('FCM')
for i =1:length(random_G)
    a=strcat(input_name(1:6),'R',random_G(i),input_name(6:end));
    A=load(a);
    plot(A(:,1),A(:,3),strcat(color(i),type(i)),'LineWidth',3)   
end

legend('R_{BG}', 'R_{ER}')
legend('boxoff')
%%set(legend,'FontSize',14)
xlabel('p')
ylabel('\kappa')
hold off
set(fig, 'units', 'inches','position',[16 15 14 10])
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Network_Density_Fnc.eps','eps2c')

% Network Clustering Coefficient
subplot(1,3,2)
hold on
set(gca,'FontSize',45)
%title('FCM')
for i =1:length(random_G)
    a=strcat(input_name(1:6),'R',random_G(i),input_name(6:end));
    A=load(a);
    plot(A(:,1),A(:,4),strcat(color(i),type(i)),'LineWidth',3)    
end
legend('R_{BG}', 'R_{ER}')
legend('boxoff')
%%set(legend,'FontSize',14)
xlabel('p')
ylabel('C')
hold off
set(fig, 'units', 'inches','position',[16 15 14 10])

% Small Worldness
input_name_2 = 'acp_w_small_worldness.dat';
subplot(1,3,3)
set(gca,'FontSize',45)
%title('FCM')
hold on
for i = 1:length(random_G)
    A =load(strcat(input_name_2(1:6),'R',random_G(i),input_name_2(6:end)));
    plot(A(:,1),A(:,8),strcat(color(i),type(i)),'LineWidth',3)  
end
legend('R_{BG}', 'R_{ER}','Location','North' ,'Orientation','horizontal')
%%set(legend,'FontSize',14)
legend('boxoff')
xlabel('p')
ylabel('S')
hold off
set(fig, 'units', 'inches','position',[16 15 14 10])
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Small_Worldness_Fnc.eps','eps2c'


% Transitivity
input_name_2 = 'acp_w_small_worldness.dat';
figure(7);
set(gca,'FontSize',45)
%title('FCM')
hold on
for i = 1:length(random_G)
    a = strcat(input_name_2(1:6),'R',random_G(i),input_name_2(6:end));
    A = load(strcat(input_name_2(1:6),'R',random_G(i),input_name_2(6:end)));
    plot(A(:,1),A(:,6),strcat(color(i),type(i)),'LineWidth',3)
end
legend('R_{BG}', 'R_{ER}','Location','SouthWest' )
%%set(legend,'FontSize',14)
legend('boxoff')
xlabel('p')
ylabel('T')
hold off
set(figure(7), 'units', 'inches','position',[16 15 14 10])
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Transitivity_Fnc.eps','eps2c')


set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Clustering_Coefficient_Fnc.eps','eps2c')

input_name = 'acp_w_single_network_measures.dat';
figure(4);
hold on
set(gca,'FontSize',45)
title('ACM')
for i =1:length(random_G)
    A=load(strcat(input_name(1:6),'R',random_G(i),input_name(6:end)));
    plot(A(:,1),A(:,6),strcat(color(i),type(i)),'LineWidth',3)     
end
legend('R_{BG}', 'R_{ER}','R_{DES}','R_{CM}','R_{PDD}','R_{PR}','Location','NorthWest')
%set(legend,'FontSize',14)
legend('boxoff')
xlabel('p')
ylabel('ACC')
hold off
set(figure(4), 'units', 'inches','position',[16 15 14 10])
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Connected_Components_Average_Stru.eps','eps2c')

