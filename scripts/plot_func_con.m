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

% Network Density
%random_G = ('0abfdc');
random_G = ('0adghk');
color='kmbcgr'; type = '-*o+*o';
input_name = 'A_aal_single_network_measures.dat';

fig = figure(1);
hold on
set(gca,'FontSize',45)
%title('FCM')
for i =1:length(random_G)
    a=strcat(input_name(1:6),'R',random_G(i),input_name(6:end));
    A=load(a);
    plot(A(:,1),A(:,3),strcat(color(i),type(i)),'LineWidth',3)   
end

legend('R_{BG}', 'R_{ER}','R_{DES}','R_{CM}','R_{PDD}','R_{PR}')
legend('boxoff')
%%set(legend,'FontSize',14)
xlabel('r')
ylabel('\kappa')
hold off
set(fig, 'units', 'inches','position',[16 15 14 10])
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Network_Density_Fnc.eps','eps2c')

% Network Clustering Coefficient
fig = figure(2);
hold on
set(gca,'FontSize',45)
%title('FCM')
for i =1:length(random_G)
    a=strcat(input_name(1:6),'R',random_G(i),input_name(6:end));
    A=load(a);
    plot(A(:,1),A(:,4),strcat(color(i),type(i)),'LineWidth',3)    
end
legend('R_{BG}', 'R_{ER}','R_{DES}','R_{CM}','R_{PDD}','R_{PR}')
legend('boxoff')
%%set(legend,'FontSize',14)
xlabel('r')
ylabel('C')
hold off
set(fig, 'units', 'inches','position',[16 15 14 10])
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Clustering_Coefficient_Fnc.eps','eps2c')


% Average degree of network
figure(3);
hold on
set(gca,'FontSize',45)
%title('FCM')
for i =1:length(random_G)
    A=load(strcat(input_name(1:6),'R',random_G(i),input_name(6:end)));
    plot(A(:,1),A(:,5),strcat(color(i),type(i)),'LineWidth',3)  
    
          
end
legend('R_{BG}', 'R_{ER}','R_{DES}','R_{CM}','R_{PDD}','R_{PR}')
legend('boxoff')
%%set(legend,'FontSize',14)
xlabel('r')
ylabel('<k>')
hold off
set(figure(3), 'units', 'inches','position',[16 15 15 10]) 
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Degree_Average_Fnc.eps','eps2c')
 
% Number of Connected Components
figure(4);
hold on
set(gca,'FontSize',45)
title('FCM')
for i =1:length(random_G)
    strcat(input_name(1:6),'R',random_G(i),input_name(6:end))
    A=load(strcat(input_name(1:6),'R',random_G(i),input_name(6:end)));
    plot(A(:,1),A(:,6),strcat(color(i),type(i)),'LineWidth',3)     
end
legend('R0', 'Ra','Rd','Rg','Rh','Rk','Location','NorthWest' )
%%set(legend,'FontSize',14)
legend('boxoff')
xlabel('r')
ylabel('Average Connected Components')
hold off
set(figure(4), 'units', 'inches','position',[5 4 10 7]) 
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Connected_Components_Average_Fnc.eps','eps2c')

% Shortest Pathway
figure(5);
hold on
set(gca,'FontSize',45)
%title('FCM')
for i =1:length(random_G)
    strcat(input_name(1:6),'R',random_G(i),input_name(6:end))
    A=load(strcat(input_name(1:6),'R',random_G(i),input_name(6:end)));
    plot(A(:,1),A(:,7),strcat(color(i),type(i)),'LineWidth',3) 
end
legend('R_{BG}', 'R_{ER}','R_{DES}','R_{PDD}','R_{CM}','R_{PR}','Location','NorthWest')
%%set(legend,'FontSize',14)
legend('boxoff')
xlabel('r')
ylabel('d_{ij}')
hold off
set(figure(5), 'units', 'inches','position',[16 15 15 11]) 
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Shortest_Pathway_Fnc.eps','eps2c')


% Small Worldness
input_name_2 = 'A_aal_small_worldness.dat';
figure(6);
set(gca,'FontSize',45)
%title('FCM')
hold on
for i = 1:length(random_G)
    A =load(strcat(input_name_2(1:6),'R',random_G(i),input_name_2(6:end)));
    plot(A(:,1),A(:,8),strcat(color(i),type(i)),'LineWidth',3)  
end
legend('R_{BG}', 'R_{ER}','R_{DES}','R_{PDD}','R_{CM}','R_{PR}','Location','SouthWest' )
%%set(legend,'FontSize',14)
legend('boxoff')
xlabel('r')
ylabel('S')
hold off
set(figure(6), 'units', 'inches','position',[16 15 15 10]) 
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Small_Worldness_Fnc.eps','eps2c')


% Transitivity
figure(7);
set(gca,'FontSize',45)
%title('FCM')
hold on
for i = 1:length(random_G)
    a = strcat(input_name_2(1:6),'R',random_G(i),input_name_2(6:end));
    A = load(strcat(input_name_2(1:6),'R',random_G(i),input_name_2(6:end)));
    plot(A(:,1),A(:,6),strcat(color(i),type(i)),'LineWidth',3)
end
legend('R_{BG}', 'R_{ER}','R_{DES}','R_{PDD}','R_{CM}','R_{PR}','Location','SouthWest' )
%%set(legend,'FontSize',14)
legend('boxoff')
xlabel('r')
ylabel('T')
hold off
set(figure(7), 'units', 'inches','position',[16 15 15 10]) 
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Transitivity_Fnc.eps','eps2c')

 
% Global Efficiency of Network
input_name_3 = 'A_aal_global_efficiency_ave.dat';
figure(8);
set(gca,'FontSize',45)
%title('FCM')
hold on
for i = 1:length(random_G)
    a = strcat(input_name_3(1:6),'R',random_G(i),input_name_3(6:end));
    A = load(a);
    plot(A(:,1),A(:,2),strcat(color(i),type(i)),'LineWidth',3)  
end
legend('R_{BG}', 'R_{ER}','R_{DES}','R_{PDD}','R_{CM}','R_{PR}','Location','SouthWest' )
%%set(legend,'FontSize',14)
legend('boxoff')
xlabel('r')
ylabel('E')
hold off
set(figure(8), 'units', 'inches','position',[16 15 16 12])
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Global_Efficiency_Average_Fnc.eps','eps2c')

% Local efficiency of Network
input_name_4 = 'A_aal_local_efficiency_ave.dat';
figure(9);
set(gca,'FontSize',45)
%title('FCM')
hold on
for i = 1:length(random_G)
    A =load(strcat(input_name_4(1:6),'R',random_G(i),'_local_efficency_ave.dat'));
    plot(A(:,1),A(:,2),strcat(color(i),type(i)),'LineWidth',3)
end
legend('R_{BG}', 'R_{ER}','R_{DES}','R_{PDD}','R_{CM}','R_{PR}','Location','SouthWest' )
%%set(legend,'FontSize',14)
legend('boxoff')
xlabel('r')
ylabel('E_{loc}')
hold off
set(figure(9), 'units', 'inches','position',[16 15 15 10]) 
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Local_Efficiency_Average_Fnc.eps','eps2c')

% Assortativity coefficient of network
in_name_5 = 'A_aal_assortativity.dat';
figure(10);
set(gca,'FontSize',45)
%title('FCM')
hold on
for i = 1:length(random_G)
    A = load(strcat(in_name_5(1:6),'R',random_G(i),in_name_5(6:end)));
    plot(A(:,1),A(:,2),strcat(color(i),type(i)), 'Linewidth', 3)
end
legend('R_{BG}', 'R_{ER}','R_{DES}','R_{PDD}','R_{CM}','R_{PR}','Location','south','Orientation','horizontal')
%%set(legend,'FontSize',14)
legend('boxoff')
xlabel('r')
ylabel('A')
hold off
axis([0 1 -1 1])
set(figure(10), 'units', 'inches','position',[16 15 15 10]) 
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Assortativity_Fnc.eps','eps2c')

% DISTRIBUTIONS
% Global efficiency of nodes
in_name_5 = 'A_aal_global_efficiency_node.dat';
figure(11);
title('FCM')
for j = 1:length(random_G)
   
    a =strcat(in_name_5(1:6),'R',random_G(j),in_name_5(6:end));
    GEF = load(a);
    z_ = GEF(:,3);
    z = zeros(101,90);
    
    subplot(3,2,j)
    set(gca,'FontSize',20)
    if random_G(j) ~= 'd' && random_G(j) ~='k' 
        for i = 1:101
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0:0.01:1);
        imagesc(n,m,z)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0 0.25 0.50 0.75 1])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
    elseif random_G(j)=='d'
        c = zeros(76,90);
        for k = 1:76
            c(k,:) = z_( 90*(k-1)+1 : 90*k )';
        end
        imagesc((1:1:90),(0.25:0.01:1),c)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0.25 0.45 0.65 0.85 1.00])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))    
    elseif random_G(j) == 'k'
        c = zeros(70,90);
        for k = 1:70
            c(k,:) = z_( 90*(k-1)+1 : 90*(k) )';
        end
        imagesc((1:1:90),(0.25:0.01:0.94),c)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0.25 0.38 0.52 0.78 0.94])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
    end
    set(gca,'XTick',15:15:90);
    xlabel('Nodes')
    ylabel('r')
    title(strcat('Global Efficiency of Nodes, R', random_G(j)))
    
end
set(figure(11), 'units', 'inches','position',[10 10 13 20]) 
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Global_Efficiency_Nodes_Fnc.eps','eps2c')

%Local efficiency of nodes
in_name_6 = 'A_aal_local_efficency_node.dat';
figure(12);
title('FCM')
for j = 1:length(random_G)
   
    a =strcat(in_name_6(1:6),'R',random_G(j),in_name_6(6:end));
    LEF = load(a);
    z_ = LEF(:,3);
    z = zeros(101,90);
    
    subplot(3,2,j)
    set(gca,'FontSize',20)
    if random_G(j) ~= 'd' && random_G(j) ~='k' 
        for i = 1:101
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0:0.01:1);
        imagesc(n,m,z)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0 0.25 0.50 0.75 1])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
    elseif random_G(j)=='d'
        c = zeros(76,90);
        for k = 1:76
            c(k,:) = z_( 90*(k-1)+1 : 90*k )';
        end
        imagesc((1:1:90),(0.25:0.01:1),c)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0.25 0.45 0.65 0.85 1.00])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))    
     
    elseif random_G(j) == 'k'
        c = zeros(70,90);
        for k = 1:70
            c(k,:) = z_( 90*(k-1)+1 : 90*(k) )';
        end
        imagesc((1:1:90),(0.25:0.01:0.94),c)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0.25 0.38 0.52 0.78 0.94])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
    end
    set(gca,'XTick',15:15:90);
    xlabel('Nodes')
    ylabel('r')
    title(strcat('Local Efficiency of Nodes, R', random_G(j)))
    
end
set(figure(12), 'units', 'inches','position',[10 10 13 20]) 
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Local_Efficiency_Nodes_Fnc.eps','eps2c')


% Connected Components of Nodes
in_name_7 = 'A_aal_connected_compo_node.dat';
figure(13);
title('FCM')
for j = 1:length(random_G)
   
    a =strcat(in_name_7(1:6),'R',random_G(j),in_name_7(6:end));
    COM = load(a);
    z_ = COM(:,3);
    z = zeros(101,90);
    
    subplot(3,2,j)
    set(gca,'FontSize',15)
    if random_G(j) ~= 'd' && random_G(j) ~='k' 
        for i = 1:101
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0:0.01:1);
        % Apply a logarithmic colorbar
        log_plot = imagesc(n,m,log10(z));
        colorbar_log([10^(0) 10^2])
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0 0.25 0.50 0.75 1])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
    elseif random_G(j)=='d'
        c = zeros(76,90);
        for k = 1:76
            c(k,:) = z_( 90*(k-1)+1 : 90*k )';
        end
        % Apply a logarithmic colorbar
        log_plot = imagesc((1:1:90),(0.25:0.01:1),log10(c));
        colorbar_log([10^(0) 10^2])
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0.25 0.45 0.65 0.85 1.00])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))  
    
    elseif random_G(j) == 'k'
        c = zeros(70,90);
        for k = 1:70
            c(k,:) = z_( 90*(k-1)+1 : 90*(k) )';
        end
        % Apply a logarithmic colorbar
        log_plot = imagesc((1:1:90),(0.25:0.01:0.94),log10(c));
        colorbar_log([10^(0) 10^2])
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0.25 0.38 0.52 0.78 0.94])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))                      
    end
    set(gca,'XTick',15:15:90);
    xlabel('Nodes')
    ylabel('r')
    title(strcat('Connected Components of Nodes, R', random_G(j)), 'FontSize',20)
    
end
set(figure(13), 'units', 'inches','position',[10 10 13 20]) 
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Connected_Components_Nodes_Fnc.eps','eps2c')

% Degree Distribution
in_name_8 = 'A_aal_degree_dist.dat';
figure(14);
title('FCM')

for j = 1:length(random_G)
   
    a =strcat(in_name_8(1:6),'R',random_G(j),in_name_8(6:end));
    DD = load(a);
    z_ = DD(:,3);
    z = zeros(101,90);
    
    subplot(3,2,j)
    set(gca,'FontSize',40)
    if random_G(j) ~= 'd' && random_G(j) ~='k' 
        for i = 1:101
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0:0.01:1);
        
        % Apply a logarithmic colorbar
        log_plot = imagesc(n,m,log10(z));
        colorbar_log([10^(0) 10^1])
        h=colorbar; set(h,'fontsize',15);
        set(gca, 'FontSize', 40)
        set(gca,'YTick',[0 0.25 0.50 0.75 1])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
        
        if random_G(j) == 'h'
            xlabel('Nodes')
        end

    elseif random_G(j)=='d'
        c = zeros(76,90);
        for k = 1:76
            c(k,:) = z_( 90*(k-1)+1 : 90*k )';
        end
        % Apply a logarithmic colorbar
        log_plot = imagesc((1:1:90),(0.25:0.01:1),log10(c));
        colorbar_log([10^(0) 10^1])
        h=colorbar; set(h,'fontsize',15);
        set(gca, 'FontSize', 40)
        set(gca,'YTick',[0.25 0.45 0.65 0.85 1.00])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))     
        
    elseif random_G(j) == 'k'
        c = zeros(70,90);
        for k = 1:70
            c(k,:) = z_( 90*(k-1)+1 : 90*(k) )';
        end
         % Apply a logarithmic colorbar
        log_plot = imagesc((1:1:90),(0.25:0.01:0.94),log10(c));
        colorbar_log([10^(0) 10^1])
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0.25 0.38 0.54 0.76 0.94])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
        set(gca, 'FontSize', 40)
        set(gca,'XTick',20:20:80);
        xlabel('Nodes')
    end
    
    ylabel('r')
    set(gca, 'FontSize', 40)
    
    
end
set(figure(14), 'units', 'inches','position',[20 10 13 20]) 
set(gcf, 'PaperPositionMode','auto')

%saveas(gcf,'Degree_Distribution_Fnc.eps','eps2c')

% Degree Distribution
R_G = '0h'
in_name_8 = 'A_aal_degree_dist.dat';

figure(20);
title('FCM')
for j = 1:length(R_G)
   
    a =strcat(in_name_8(1:6),'R',R_G(j),in_name_8(6:end));
    DD = load(a);
    z_ = DD(:,3);
    z = zeros(101,90);
    

    set(gca,'FontSize',30)
    
    for i = 1:101
        a = i-1;
       z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
    end
    n=(1:90);
    m=(0:0.01:1);

    linear_axes = subplot(1,2,j);
    my_clim = [1e-2 1e1];
    linear_plot = imagesc( n,m,z);
    colormap(jet(1024)), caxis(my_clim)
    cbar = colorbar('peer', linear_axes, 'Yscale', 'log' ,'fontsize',20);

    
    %     % Apply a logarithmic colorbar
%     log_plot = imagesc(n,m,log10(z));
%     colorbar_log([10^(0) 10^1])
%     h=colorbar; set(h,'fontsize',15);

    set(gca,'YTick',[0 0.25 0.50 0.75 1])
    set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))

  
   
    set(gca,'XTick',15:15:90);
    xlabel('Nodes')
    ylabel('r')
    title('p(k)', 'FontSize',30)
    
end
set(figure(20), 'units', 'inches','position',[20 20 20 6]) 
set(gcf, 'PaperPositionMode','auto')

% Clustering Coeficient of Nodes
in_name_8 = 'A_aal_cc_and_degree_node.dat';
figure(15);
title('FCM')
for j = 1:length(random_G)
   
    a =strcat(in_name_8(1:6),'R',random_G(j),in_name_8(6:end));
    DD = load(a);
    z_ = DD(:,3);
    z = zeros(101,90);
    
    subplot(3,2,j)
    set(gca,'FontSize',15)
    if random_G(j) ~= 'd' && random_G(j) ~='k' 
        for i = 1:101
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0:0.01:1);
        imagesc(n,m,z)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0 0.25 0.50 0.75 1])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
    elseif random_G(j)=='d'
        c = zeros(76,90);
        for k = 1:76
            c(k,:) = z_( 90*(k-1)+1 : 90*k )';
        end
        imagesc((1:1:90),(0.25:0.01:1),c)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0.25 0.45 0.65 0.85 1.00])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f')) 
    elseif random_G(j) == 'k'
        c = zeros(70,90);
        for k = 1:70
            c(k,:) = z_( 90*(k-1)+1 : 90*(k) )';
        end
        imagesc((1:1:90),(0.25:0.01:0.94),c)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0.25 0.38 0.54 0.76 0.94])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
    end
    set(gca,'XTick',15:15:90);
    xlabel('Nodes')
    ylabel('r')
    title(strcat('C_{i}, R', random_G(j)), 'FontSize',20)
  
end
set(figure(15), 'units', 'inches','position',[10 10 13 20]) 
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Clustering_Coefficient_Node_Fnc.eps','eps2c')

