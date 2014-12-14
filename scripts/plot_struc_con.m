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

random_G = ('0adghk');
color='kmbcgr'; type = '-*o+*o';

% Network Density
input_name = 'acp_w_single_network_measures.dat';
fig = figure(1);
hold on
set(gca,'FontSize',45)
%title('ACM')
for i =1:length(random_G)
    a=strcat(input_name(1:6),'R',random_G(i),input_name(6:end));
    A=load(a);
    plot(A(:,1),A(:,3),strcat(color(i),type(i)),'LineWidth',3)   
end
legend('R_{BG}', 'R_{ER}','R_{DES}','R_{CM}','R_{PDD}','R_{PR}')
legend('boxoff')
%set(legend,'FontSize',14)
xlabel('p')
ylabel('\kappa')
hold off
set(fig, 'units', 'inches','position',[16 15 14 10]) 
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Network_Density_Stru.eps','eps2c')

% Network Clustering Coefficient
fig = figure(2);
input_name = 'acp_w_single_network_measures.dat';
hold on
set(gca,'FontSize',45)
%title('ACM')
for i =1:length(random_G)
    a=strcat(input_name(1:6),'R',random_G(i),input_name(6:end));
    A=load(a);
    plot(A(:,1),A(:,4),strcat(color(i),type(i)),'LineWidth',3)    
end
%legend('R_{BG}', 'R_{ER}','R_{DES}','R_{CM}','R_{PDD}','R_{PR}')
legend('R_{BG}', 'R_{ER}','R_{DES}','R_{CM}','R_{PDD}','R_{PR}','Location','north','Orientation','horizontal')

legend('boxoff')
%set(legend,'FontSize',14)
xlabel('p')
ylabel('C')
hold off
set(fig, 'units', 'inches','position',[16 15 14 10])
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Clustering_Coefficient_Stru.eps','eps2c')

% Average degree of network
input_name = 'acp_w_single_network_measures.dat';
figure(3);
hold on
set(gca,'FontSize',45)
%title('ACM')
for i =1:length(random_G)
    A=load(strcat(input_name(1:6),'R',random_G(i),input_name(6:end)));
    plot(A(:,1),A(:,5),strcat(color(i),type(i)),'LineWidth',3)  
    
          
end
legend('R_{BG}', 'R_{ER}','R_{DES}','R_{CM}','R_{PDD}','R_{PR}')
legend('boxoff')
%set(legend,'FontSize',14)
xlabel('p')
ylabel('<k>')
hold off
set(figure(3), 'units', 'inches','position',[16 15 14 10]) 
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Degree_Average_Stru.eps','eps2c')

% Number of Connected Components
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

% Shortest Pathway
input_name = 'acp_w_single_network_measures.dat';
figure(5);
hold on
set(gca,'FontSize',45)
%title('ACM')
for i =1:length(random_G)
    A=load(strcat(input_name(1:6),'R',random_G(i),input_name(6:end)));
    plot(A(:,1),A(:,7),strcat(color(i),type(i)),'LineWidth',3) 
end
legend('R_{BG}', 'R_{ER}','R_{DES}','R_{CM}','R_{PDD}','R_{PR}','Location','north','Orientation','horizontal')
%set(legend,'FontSize',14)
legend('boxoff')
xlabel('p')
ylabel('d_{ij}')
hold off
set(figure(5), 'units', 'inches','position',[16 15 14 10]) 
axis([0 1 0 5])
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Shortest_Pathway_Stru.eps','eps2c')
 
% Small Worldness
input_name_2 = 'acp_w_small_worldness.dat';
figure(6);
set(gca,'FontSize',45)
%title('ACM')
hold on
for i = 1:length(random_G)
    A =load(strcat(input_name_2(1:6),'R',random_G(i),input_name_2(6:end)));
    plot(A(:,1),A(:,8),strcat(color(i),type(i)),'LineWidth',3)  
end
legend('R_{BG}', 'R_{ER}','R_{DES}','R_{CM}','R_{PDD}','R_{PR}','Location','north','Orientation','horizontal')
%set(legend,'FontSize',14)
legend('boxoff')
xlabel('p')
ylabel('S')
hold off
axis([0 1 0 15])
set(figure(6), 'units', 'inches','position',[16 15 14 10])
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Small_Worldness_Stru.eps','eps2c')

% Transitivity
figure(7);
set(gca,'FontSize',45)
%title('ACM')
hold on
for i = 1:length(random_G)
    a = strcat(input_name_2(1:2),'R',random_G(i),input_name_2(2:end));
    A = load(strcat(input_name_2(1:6),'R',random_G(i),input_name_2(6:end)));
    plot(A(:,1),A(:,6),strcat(color(i),type(i)),'LineWidth',3)
end
legend('R_{BG}', 'R_{ER}','R_{DES}','R_{CM}','R_{PDD}','R_{PR}','Location','north','Orientation','horizontal')

legend('boxoff')
xlabel('p')
ylabel('T')
hold off
set(figure(7), 'units', 'inches','position',[16 15 14 10]) 

%set(figure(7), 'units', 'inches','position',[5 4 10 7]) 
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Transitivity_Stru.eps','eps2c')

% Global Efficiency of Network
input_name_3 = 'acp_w_global_efficiency_ave.dat';
figure(8);
set(gca,'FontSize',45)
%title('ACM')
hold on
for i = 1:length(random_G)
    a = strcat(input_name_3(1:6),'R',random_G(i),input_name_3(6:end));
    A = load(a);
    plot(A(:,1),A(:,2),strcat(color(i),type(i)),'LineWidth',3)  
end
legend('R_{BG}', 'R_{ER}','R_{DES}','R_{CM}','R_{PDD}','R_{PR}','Location','north','Orientation','horizontal')
%set(legend,'FontSize',14)
legend('boxoff')
xlabel('p')
ylabel('E')
hold off
set(figure(8), 'units', 'inches','position',[16 15 14 10])
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Global_Efficiency_Average_Stru.eps','eps2c')

% Local efficiency of Network
input_name_4 = 'acp_w_local_efficiency_ave.dat';
figure(9);
set(gca,'FontSize',45)
%title('ACM')
hold on
for i = 1:length(random_G)
    A =load(strcat(input_name_4(1:6),'R',random_G(i),'_local_efficency_ave.dat'));
    plot(A(:,1),A(:,2),strcat(color(i),type(i)),'LineWidth',3)
end
legend('R_{BG}', 'R_{ER}','R_{DES}','R_{CM}','R_{PDD}','R_{PR}','Location','north','Orientation','horizontal')
%set(legend,'FontSize',14)
legend('boxoff')
xlabel('p')
ylabel('E_{loc}')
hold off
set(figure(9), 'units', 'inches','position',[16 15 14 10])
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Local_Efficiency_Average_Stru.eps','eps2c')

% Assortativity coefficient of network
in_name_5 = 'acp_w_assortativity.dat';
figure(10);
set(gca,'FontSize',45)
%title('ACM')
hold on
for i = 1:length(random_G)
    A = load(strcat(in_name_5(1:6),'R',random_G(i),in_name_5(6:end)));
    plot(A(:,1),A(:,2),strcat(color(i),type(i)), 'Linewidth', 3)
end
legend('R_{BG}', 'R_{ER}','R_{DES}','R_{CM}','R_{PDD}','R_{PR}','Location','north','Orientation','horizontal')
%set(legend,'FontSize',14)
legend('boxoff')
xlabel('p')
ylabel('A')
hold off
axis([0 1 -1 1])
set(figure(10), 'units', 'inches','position',[16 15 14 10])
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Assortativity_Stru.eps','eps2c')

% DISTRIBUTIONS
% Global efficiency of nodes
random_G = ('0adghk');

in_name_5 = 'acp_w_global_efficiency_node.dat';
figure(11);
title('ACM')
for j = 1:length(random_G)
   
    a =strcat(in_name_5(1:6),'R',random_G(j),in_name_5(6:end));
    GEF = load(a);
    z_ = GEF(:,3);
    z = zeros(size(GEF,1)/90,90);
    
    subplot(3,2,j)
    set(gca,'FontSize',20)
    if random_G(j) == '0' 
        for i = 1:(size(GEF,1)/90)
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0:0.01:0.99);
        imagesc(n,m,z)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0 0.25 0.50 0.80 0.99])
        
    elseif random_G(j) == 'a' %&& random_G(j) =='a' && random_G(j) == 'g' 
        for i = 1:(size(GEF,1)/90)
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0:0.01:0.99);
        imagesc(n,m,z)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0 0.25 0.50 0.80 0.99])

    elseif random_G(j) == 'd' %&& random_G(j) =='a' && random_G(j) == 'g' 
        for i = 1:(size(GEF,1)/90)
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0.01:0.01:0.99);
        imagesc(n,m,z)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0.01 0.25 0.50 0.80 0.99])
 
    elseif random_G(j) == 'g' %&& random_G(j) =='a' && random_G(j) == 'g' 
        for i = 1:(size(GEF,1)/90)
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0:0.01:0.99);
        imagesc(n,m,z)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0 0.25 0.50 0.80 0.99])
        
    elseif random_G(j) == 'h' %&& random_G(j) =='a' && random_G(j) == 'g' 
        for i = 1:(size(GEF,1)/90)
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0.05:0.01:0.99);
        imagesc(n,m,z)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0.05 0.25 0.50 0.80 0.99])
    
    elseif random_G(j) == 'k' %&& random_G(j) =='a' && random_G(j) == 'g' 
        for i = 1:(size(GEF,1)/90)
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0.01:0.01:0.98);
        imagesc(n,m,z)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0.01 0.25 0.50 0.75 0.98])
    end
    
    set(gca,'XTick',15:15:90);
    xlabel('Nodes')
    ylabel('p')
    random_G(j)
    strcat('Global Efficiency of Nodes, R', random_G(j))
    title(strcat('E_{nodes}, R', random_G(j)))
    set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
    

end
set(figure(11), 'units', 'inches','position',[10 10 13 20]) 
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Global_Efficiency_Nodes_Stru.eps','eps2c')

%Local efficiency of nodes
in_name_6 = 'acp_w_local_efficency_node.dat';
figure(12);
title('ACM')
for j = 1:length(random_G)
   
    a =strcat(in_name_6(1:6),'R',random_G(j),in_name_6(6:end));
    GEF = load(a);
    z_ = GEF(:,3);
    z = zeros(size(GEF,1)/90,90);
    
    subplot(3,2,j)
    set(gca,'FontSize',20)
    if random_G(j) == '0' 
        for i = 1:(size(GEF,1)/90)
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0:0.01:0.99);
        imagesc(n,m,z)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0 0.25 0.50 0.80 0.99])
        
    elseif random_G(j) == 'a' %&& random_G(j) =='a' && random_G(j) == 'g' 
        for i = 1:(size(GEF,1)/90)
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0:0.01:0.99);
        imagesc(n,m,z)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0 0.25 0.50 0.80 0.99])

    elseif random_G(j) == 'd' %&& random_G(j) =='a' && random_G(j) == 'g' 
        for i = 1:(size(GEF,1)/90)
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0.01:0.01:0.99);
        imagesc(n,m,z)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0.01 0.25 0.50 0.80 0.99])
 
    elseif random_G(j) == 'g' %&& random_G(j) =='a' && random_G(j) == 'g' 
        for i = 1:(size(GEF,1)/90)
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0:0.01:0.99);
        imagesc(n,m,z)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0 0.25 0.50 0.80 0.99])
        
    elseif random_G(j) == 'h' %&& random_G(j) =='a' && random_G(j) == 'g' 
        for i = 1:(size(GEF,1)/90)
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0.05:0.01:0.99);
        imagesc(n,m,z)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0.05 0.25 0.50 0.80 0.99])
    
    elseif random_G(j) == 'k' %&& random_G(j) =='a' && random_G(j) == 'g' 
        for i = 1:(size(GEF,1)/90)
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0.01:0.01:0.98);
        imagesc(n,m,z)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0.01 0.25 0.50 0.75 0.98])
    end
    
    set(gca,'XTick',15:15:90);
    xlabel('Nodes')
    ylabel('p')
    random_G(j)
    
    title(strcat(strcat('E_{loc, nodes}, R', random_G(j))))
    set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
    
end
set(figure(12), 'units', 'inches','position',[10 10 13 20]) 
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Local_Efficiency_Nodes_Stru.eps','eps2c')


%????
% Connected Components of Nodes
in_name_7 = 'acp_w_connected_compo_node.dat';
figure(13);
title('ACM')
for j = 1:length(random_G)
   
    a =strcat(in_name_7(1:6),'R',random_G(j),in_name_7(6:end));
    COM = load(a);
    z_ = COM(:,3);
    z = zeros(size(COM,1)/90,90);
    
    subplot(3,2,j)
    set(gca,'FontSize',15)
    
    if random_G(j) =='0'
        for i = 1:size(COM,1)/90
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0:0.01:0.99);
        %Apply a logarithmic colorbar
        log_plot = imagesc(n,m,log10(z));
        colorbar_log([10^(0) 10^0.05])
        %imagesc(n,m,z)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0 0.25 0.50 0.80 0.99])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
   
    elseif random_G(j) =='a'
        for i = 1:size(COM,1)/90
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0:0.01:0.99);
        %Apply a logarithmic colorbar
        log_plot = imagesc(n,m,log10(z));
        colorbar_log([10^(0) 10^0.05])
        %imagesc(n,m,z)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0 0.25 0.50 0.80 0.99])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))

   
    elseif random_G(j) =='d'
        for i = 1:size(COM,1)/90
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0.01:0.01:0.99);
        %Apply a logarithmic colorbar
        log_plot = imagesc(n,m,log10(z));
        colorbar_log([10^(0) 10^0.05])
        %imagesc(n,m,z)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0.01 0.25 0.50 0.75 0.99])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
            
    elseif random_G(j) =='g'
        for i = 1:size(COM,1)/90
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0:0.01:0.99);
        %Apply a logarithmic colorbar
        log_plot = imagesc(n,m,log10(z));
        colorbar_log([10^(0) 10^0.05])
        %imagesc(n,m,z)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0 0.25 0.50 0.75 0.99])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))   
        
        
    elseif random_G(j) =='h'
        for i = 1:size(COM,1)/90
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0.05:0.01:0.99);
        %Apply a logarithmic colorbar
        log_plot = imagesc(n,m,log10(z));
        colorbar_log([10^(0) 10^0.05])
        %imagesc(n,m,z)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0.05 0.25 0.50 0.75 0.99])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))     
   
  elseif random_G(j) =='k'
        for i = 1:size(COM,1)/90
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0.01:0.01:0.98);
        %Apply a logarithmic colorbar
        log_plot = imagesc(n,m,log10(z));
        colorbar_log([10^(0) 10^0.05])
        %imagesc(n,m,z)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0.01 0.25 0.50 0.75 0.98])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))     
    end
    set(gca,'XTick',15:15:90);
    xlabel('Nodes')
    ylabel('p')
    title(strcat('Connected Components of Nodes, R', random_G(j)))
end
set(figure(13), 'units', 'inches','position',[10 10 13 20]) 
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Connected_Components_Nodes_Stru.eps','eps2c')

% Degree Distribution
in_name_8 = 'acp_w_degree_dist.dat';
figure(14);
%title('ACM')
for j = 1:length(random_G)
   
    a =strcat(in_name_8(1:6),'R',random_G(j),in_name_8(6:end));
    DD = load(a);
    z_ = DD(:,3);
    z = zeros(size(DD,1)/100,90);
    
    linear_axes = subplot(3,2,j);
    my_clim = [1 1e1];
    
    if random_G(j) =='0'
        title('R_{BG}', 'fontsize', 25)
        for i = 1:size(DD,1)/90
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0:0.01:0.99);
    
        linear_plot = imagesc( n,m,z);
        colormap(jet(1024)), caxis(my_clim)
        cbar = colorbar('peer', linear_axes, 'Yscale', 'log' ,'fontsize',10);

        set(gca,'YTick',[0 0.25 0.50 0.75 0.99])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
   
    elseif random_G(j) =='a'
        title('R_{ER}', 'fontsize', 25)
        for i = 1:size(DD,1)/90
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0:0.01:0.99);
    
        linear_plot = imagesc( n,m,z);
        colormap(jet(1024)), caxis(my_clim)
        cbar = colorbar('peer', linear_axes, 'Yscale', 'log' ,'fontsize',10);

        set(gca,'YTick',[0 0.25 0.50 0.75 0.99])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
    
    elseif random_G(j) =='d'
        title('R_{DES}', 'fontsize', 25)
        for i = 1:size(DD,1)/90
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0.01:0.01:0.99);
    
        linear_plot = imagesc( n,m,z);
        colormap(jet(1024)), caxis(my_clim)
        cbar = colorbar('peer', linear_axes, 'Yscale', 'log' ,'fontsize',10);

        set(gca,'YTick',[0.01 0.25 0.50 0.75 0.99])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f')) 
        
   elseif random_G(j) == 'g'
        title('R_{CM}', 'fontsize', 25)
        for i = 1:size(DD,1)/90
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0:0.01:0.99);
    
        linear_plot = imagesc( n,m,z);
        colormap(jet(1024)), caxis(my_clim)
        cbar = colorbar('peer', linear_axes, 'Yscale', 'log' ,'fontsize',10);

        set(gca,'YTick',[0 0.25 0.50 0.75 0.99])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f')) 
    
   elseif random_G(j) == 'h'
        title('R_{PDD}', 'fontsize', 25)
        for i = 1:size(DD,1)/90
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0.05:0.01:0.99);
    
        linear_plot = imagesc( n,m,z);
        colormap(jet(1024)), caxis(my_clim)
        cbar = colorbar('peer', linear_axes, 'Yscale', 'log' ,'fontsize',10);

        set(gca,'YTick',[0.05 0.25 0.50 0.75 0.99])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f')) 
        
        xlabel('k', 'fontsize', 25)

        
   elseif random_G(j) == 'k'
        title('R_{PR}', 'fontsize', 25)
        for i = 1:size(DD,1)/90
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0.01:0.01:0.98);
    
        linear_plot = imagesc( n,m,z);
        colormap(jet(1024)), caxis(my_clim)
        cbar = colorbar('peer', linear_axes, 'Yscale', 'log' ,'fontsize',10);

        set(gca,'YTick',[0.01 0.25 0.50 0.75 0.98])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))   
        xlabel('k', 'fontsize', 25)
        
    end   
   
    set(gca,'XTick',20:20:80);
    ylabel('p', 'fontsize', 25)

 
end
set(figure(14), 'units', 'inches','position',[20 10 18 20]) 
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Degree_Distribution_Stru.eps','eps2c')



% Clustering Coeficient of Nodes
in_name_8 = 'acp_w_cc_and_degree_node.dat';
figure(15);
%title('ACM')
for j = 1:length(random_G)
   
    a =strcat(in_name_8(1:6),'R',random_G(j),in_name_8(6:end));
    DD = load(a);
    z_ = DD(:,3);
    z = zeros(size(DD,1)/90,90);
    
    subplot(3,2,j)
    set(gca,'FontSize',15)
    if random_G(j) =='0' 
        for i = 1:size(DD,1)/90
            a = i-1;
           z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
        end
        n=(1:90);
        m=(0:0.01:0.99);
        imagesc(n,m,z)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0 0.25 0.50 0.75 0.99])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))

    elseif random_G(j)=='a'
        c = zeros(size(DD,1)/90 ,90 );
        for k = 1:size(DD,1)/90
            c(k,:) = z_( 90*(k-1)+1 : 90*k )';
        end
        imagesc((1:1:90),(0:0.01:0.99),c)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0.01 0.25 0.50 0.75 0.99])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))    
   
    elseif random_G(j) == 'd'
        c = zeros(size(DD,1)/90 ,90 );
        for k = 1:size(DD,1)/90
            c(k,:) = z_( 90*(k-1)+1 : 90*(k) )';
        end
        imagesc((1:1:90),(0.01:0.01:0.99),c)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0.01 0.25 0.50 0.75 0.99])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
        
    elseif random_G(j) == 'g'
        c = zeros(size(DD,1)/90 ,90 );
        for k = 1:size(DD,1)/90
            c(k,:) = z_( 90*(k-1)+1 : 90*(k) )';
        end
        imagesc((1:1:90),(0:0.01:0.99),c)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0 0.25 0.50 0.75 0.99])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))    
        
   elseif random_G(j) == 'h'
        c = zeros(size(DD,1)/90 ,90 );
        for k = 1:size(DD,1)/90
            c(k,:) = z_( 90*(k-1)+1 : 90*(k) )';
        end
        imagesc((1:1:90),(0.05:0.01:0.99),c)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0.05 0.25 0.50 0.75 0.99])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))      
        
   elseif random_G(j) == 'k'
        c = zeros(size(DD,1)/90 ,90 );
        for k = 1:size(DD,1)/90
            c(k,:) = z_( 90*(k-1)+1 : 90*(k) )';
        end
        imagesc((1:1:90),(0.01:0.01:0.98),c)
        h=colorbar; set(h,'fontsize',15);
        set(gca,'YTick',[0.01 0.25 0.50 0.75 0.98])
        set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))    
        
    end    
        
    
   set(gca,'XTick',20:20:80);
   ylabel('p', 'fontsize', 25)
  
   if random_G(j) == '0'
        title('R_{BG}', 'fontsize', 25)
   
   elseif random_G(j) == 'a'
       title('R_{ER}', 'fontsize', 25)
   
   elseif random_G(j) == 'd'
       title('R_{DES}', 'fontsize', 25)

   elseif random_G(j) == 'g'
       title('R_{CM}', 'fontsize', 25)

   elseif random_G(j) == 'h'
       title('R_{PDD}', 'fontsize', 25)

   elseif random_G(j) == 'k'
       title('R_{PR}', 'fontsize', 25)    
   end

   set(gca, 'fontsize', 25)   
       
    
    
end
set(figure(15), 'units', 'inches','position',[20 10 18 20])
set(gcf, 'PaperPositionMode','auto')
%saveas(gcf,'Clustering_Coefficient_Node_Stru.eps','eps2c')

