% plot network measures of test network

% A_single_network_measures.dat: #1. threshold, 2. edges, 3. density 
              %4.clustering coefficient, 5. average degree
              %6. number of connected components, 7. shortest pathway
% A_small_worldnes.dat : #1:threshold 2:cluster-coefficient... 
              %...3:random-cluster-coefficient 4:shortest-pathlength 
              %...5:random-shortest-pathlength 6:transitivity 
              %...7:random-transitivity 8:S-Watts-Strogatz 9:S-transitivity
% A_global_efficiency_ave.dat : #1.threshold, 2.global efficieny
% A_local_efficency_ave.dat : # 1.threshold, 2.local efficiency
% A_global_efficiency_node.dat : #1.node, 2,threshold, 3.glo. effi.of node              

% A_degree_dist.dat : #1.node, 2.threshold, 3.degree hist, 4.degree distr.
% A_connected_compo_node.dat : # 1.node, 2.threshold, 3. connected compon.

% Network Density
random_G = ('a');
color='gybr';
%test_network = load('A_aal_single_network_measures.dat');
input_name = 'A_aal_single_network_measures.dat';
figure(1);
hold on
set(gca,'FontSize',20)
%plot(test_network(:,1),test_network(:,3),'k','LineWidth',3)
for i =1:length(random_G)
    a=strcat(input_name(1:6),'R',random_G(i),input_name(6:end));
    A=load(a);
    plot(A(:,1),A(:,3),strcat(color(i),'o'),'LineWidth',3)    
end
%legend('Test Network', 'Rand. a','Rand. b','Rand. c','Rand. d')
%legend('boxoff')
xlabel('Threshold [r]')
ylabel('Network Density')
hold off
% 
% % Network Clustering Coefficient
% figure(2);
% hold on
% set(gca,'FontSize',20)
% plot(test_network(:,1),test_network(:,4),'k','LineWidth',3)
% for i =1:length(random_G)
%     a=strcat(input_name(1:2),'R',random_G(i),input_name(2:end));
%     A=load(a);
%     plot(A(:,1),A(:,4),strcat(color(i),'o'),'LineWidth',3)    
% end
% legend('Test Network', 'Rand. a','Rand. b','Rand. c','Rand. d')
% legend('boxoff')
% xlabel('Threshold [r]')
% ylabel('Clustering Coefficient')
% hold off
% 
% % Average degree of network
% figure(3);
% hold on
% set(gca,'FontSize',20)
% plot(test_network(:,1),test_network(:,5),'k','LineWidth',3)
% for i =1:length(random_G)
%     a=strcat(input_name(1:2),'R',random_G(i),input_name(2:end));
%     A=load(a);
%     plot(A(:,1),A(:,5),strcat(color(i),'o'),'LineWidth',3)    
% end
% legend('Test Network', 'Rand. a','Rand. b','Rand. c','Rand. d')
% legend('boxoff')
% xlabel('Threshold [r]')
% ylabel('Average Degree')
% hold off
% 
% % Number of Connected Components
% figure(4);
% hold on
% set(gca,'FontSize',20)
% plot(test_network(:,1),test_network(:,6),'k','LineWidth',3)
% for i =1:length(random_G)
%     a=strcat(input_name(1:2),'R',random_G(i),input_name(2:end));
%     A=load(a);
%     plot(A(:,1),A(:,6),strcat(color(i),'o'),'LineWidth',3)    
% end
% legend('Test Network', 'Rand. a','Rand. b','Rand. c','Rand. d')
% legend('boxoff')
% xlabel('Threshold [r]')
% ylabel('connected Components')
% hold off
% 
% % Shortest Pathway
% figure(4);
% hold on
% set(gca,'FontSize',20)
% plot(test_network(:,1),test_network(:,7),'k','LineWidth',3)
% for i =1:length(random_G)
%     a=strcat(input_name(1:2),'R',random_G(i),input_name(2:end));
%     A=load(a);
%     plot(A(:,1),A(:,7),strcat(color(i),'o'),'LineWidth',3)    
% end
% legend('Test Network', 'Rand. a','Rand. b','Rand. c','Rand. d')
% legend('boxoff')
% xlabel('Threshold [r]')
% ylabel('Shortest Pathway')
% hold off
% 
% 
% % Small Worldness
% input_name_2 = 'A_small_worldness.dat';
% SM = load('A_small_worldness.dat');
% figure(5);
% set(gca,'FontSize',20)
% plot(SM(:,1),SM(:,8),'k','LineWidth',3)
% hold on
% for i = 1:length(random_G)
%     a = strcat(input_name_2(1:2),'R',random_G(i),input_name_2(2:end));
%     A = load(a);
%     plot(A(:,1),A(:,8),strcat(color(i),'o'),'LineWidth',3)
% end
% legend('Test Network', 'Rand. a','Rand. b','Rand. c','Rand. d')
% legend('boxoff')
% xlabel('Threshold [r]')
% ylabel('Small Worldness')
% hold off
% 
% % Transitivity
% figure(6);
% set(gca,'FontSize',20)
% plot(SM(:,1),SM(:,6),'k','LineWidth',3)
% hold on
% for i = 1:length(random_G)
%     a = strcat(input_name_2(1:2),'R',random_G(i),input_name_2(2:end));
%     A = load(a);
%     plot(A(:,1),A(:,6),strcat(color(i),'o'),'LineWidth',3)
% end
% legend('Test Network', 'Rand. a','Rand. b','Rand. c','Rand. d')
% legend('boxoff')
% xlabel('Threshold [r]')
% ylabel('Transitivity')
% hold off
% 
% 
% % Global Efficiency of Network
% input_name_3 = 'A_global_efficiency_ave.dat';
% GE = load('A_global_efficiency_ave.dat');
% figure(7);
% set(gca,'FontSize',20)
% plot(GE(:,1),GE(:,2),'k','LineWidth',3)
% hold on
% for i = 1:length(random_G)
%     a = strcat(input_name_3(1:2),'R',random_G(i),input_name_3(2:end));
%     A = load(a);
%     plot(A(:,1),A(:,2),strcat(color(i),'o'),'LineWidth',3)
% end
% legend('Test Network', 'Rand. a','Rand. b','Rand. c','Rand. d')
% legend('boxoff')
% xlabel('Threshold [r]')
% ylabel('Global Efficiency')
% hold off
% 
% % Local efficiency of Network
% input_name_4 = 'A_local_efficiency_ave.dat';
% LE = load('A_local_efficency_ave.dat');
% figure(8);
% set(gca,'FontSize',20)
% plot(LE(:,1),LE(:,2),'k','LineWidth',3)
% hold on
% for i = 1:length(random_G)
%     a=strcat(input_name_4(1:2),'R',random_G(i),'_local_efficency_ave.dat');
%     A = load(a);
%     plot(A(:,1),A(:,2),strcat(color(i),'o'),'LineWidth',3)
% end
% legend('Test Network', 'Rand. a','Rand. b','Rand. c','Rand. d')
% legend('boxoff')
% xlabel('Threshold [r]')
% ylabel('Local Efficiency')
% hold off
% 
% 
% % Global efficiency of nodes
% GEF = load('A_global_efficiency_node.dat');
% input_name_5 = 'A_global_efficiency_node.dat';
% 
% z_ = GEF(:,3);
% 
% figure;
% subplot(2,2,1)
% for i = 1:101
%     a = i-1;
%    z(i,:) = z_( ( 64*a+1 :(64*a+64) ),:);
% end
% n=(1:64);
% m=(0:0.01:1);
% surf(n,m,z)
% colorbar
% 
% for i = 1:1
%     a=strcat(input_name_5(1:2),'R',random_G(i),input_name_5(2:end));
%     A = load(a);
%     z_ = A(:,3);
%     for j = 1:101
%         a = j-1;
%         z(j,:) = z_( ( 64*a+1 :(64*a+64) ),:);
%     end
%     subplot(2,2,i+1)
%     n=(1:64);
%     m=(0:0.01:1);
%     surf(n,m,z)
%     colorbar
%     
% end


