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
% A_*_global_efficiency_node.dat : #1.node, 2,threshold, 3.glo. effi.of node              

% A_*_degree_dist.dat : #1.node, 2.threshold, 3.degree hist, 4.degree distr.
% A_*_connected_compo_node.dat : # 1.node, 2.threshold, 3. connected compon.
% A_*_cc_and_degree_node.dat: # 1.node, 2.threshold, 3. clustering coef.
%                                of each node, 4. degree of node

% Network Density
random_G = ('abcd');
color='gbyr';
test_network = load('A_aal_single_network_measures.dat');
input_name = 'A_aal_single_network_measures.dat';
fig = figure(1);
hold on
set(gca,'FontSize',30)
plot(test_network(:,1),test_network(:,3),'k','LineWidth',5)
for i =1:length(random_G)
    a=strcat(input_name(1:6),'R',random_G(i),input_name(6:end));
    A=load(a);
    if color(i)=='g'
        plot(A(:,1),A(:,3),strcat(color(i),'^'),'LineWidth',6) 
    else
        plot(A(:,1),A(:,3),strcat(color(i),'o'),'LineWidth',5)
    end
end
legend('Test Network', 'Ra','Rb','Rc','Rd')
legend('boxoff')
xlabel('Threshold [r]','FontSize',35)
ylabel('Network Density','FontSize',35)
hold off
set(fig, 'units', 'inches','position',[10 10 30 20]) 
set(gcf, 'PaperPositionMode','auto')
saveas(gcf,'Network_Density.eps','eps2c')



% Network Clustering Coefficient
figure(2);
hold on
set(gca,'FontSize',30)
plot(test_network(:,1),test_network(:,4),'k','LineWidth',5)
for i =1:length(random_G)
    a=strcat(input_name(1:6),'R',random_G(i),input_name(6:end));
    A=load(a);
    plot(A(:,1),A(:,4),strcat(color(i),'o'),'LineWidth',5)    
end
legend('Test Network', 'Ra','Rb','Rc','Rd')
legend('boxoff')
xlabel('Threshold [r]','FontSize',35)
ylabel('Clustering Coefficient','FontSize',35)
hold off
set(figure(2), 'units', 'inches','position',[10 10 30 20]) 
set(gcf, 'PaperPositionMode','auto')
saveas(gcf,'Clustering_Coefficient.eps','eps2c')


% Average degree of network
figure(3);
hold on
set(gca,'FontSize',30)
plot(test_network(:,1),test_network(:,5),'k','LineWidth',5)
for i =1:length(random_G)
    A=load(strcat(input_name(1:6),'R',random_G(i),input_name(6:end)));
    if color(i) == 'g'
        plot(A(:,1),A(:,5),strcat(color(i),'^'),'LineWidth',6)  
    else
        plot(A(:,1),A(:,5),strcat(color(i),'o'),'LineWidth',5)
    end        
end
legend('Test Network', 'Ra','Rb','Rc','Rd')
legend('boxoff')
xlabel('Threshold [r]','FontSize',35)
ylabel('Average Degree, <k>','FontSize',35)
hold off
set(figure(3), 'units', 'inches','position',[10 10 30 20]) 
set(gcf, 'PaperPositionMode','auto')
saveas(gcf,'Degree_Average.eps','eps2c')
 
% Number of Connected Components
figure(4);
hold on
set(gca,'FontSize',30)
plot(test_network(:,1),test_network(:,6),'k','LineWidth',5)
for i =1:length(random_G)
    A=load(strcat(input_name(1:6),'R',random_G(i),input_name(6:end)));
    if color(i) == 'g'
        plot(A(:,1),A(:,6),strcat(color(i),'^'),'LineWidth',6)  
    else
        plot(A(:,1),A(:,6),strcat(color(i),'o'),'LineWidth',5)
    end    
end
legend('Test Network', 'Ra','Rb','Rc','Rd','Location','NorthWest')
legend('boxoff')
xlabel('Threshold [r]','FontSize',35)
ylabel('Average Connected Components','FontSize',35)
hold off
set(figure(4), 'units', 'inches','position',[10 10 30 20]) 
set(gcf, 'PaperPositionMode','auto')
saveas(gcf,'Connected_Components_Average.eps','eps2c')

% Shortest Pathway
figure(5);
hold on
set(gca,'FontSize',30)
plot(test_network(:,1),test_network(:,7),'k','LineWidth',5)
for i =1:length(random_G)
    A=load(strcat(input_name(1:6),'R',random_G(i),input_name(6:end)));
    if color(i) == 'g'
        plot(A(:,1),A(:,7),strcat(color(i),'^'),'LineWidth',6)  
    else
        plot(A(:,1),A(:,7),strcat(color(i),'o'),'LineWidth',5)
    end   
end
legend('Test Network', 'Ra','Rb','Rc','Rd')
legend('boxoff')
xlabel('Threshold [r]','FontSize',35)
ylabel('Shortest Pathway','FontSize',35)
hold off
set(figure(5), 'units', 'inches','position',[10 10 30 20]) 
set(gcf, 'PaperPositionMode','auto')
saveas(gcf,'Shortest_Pathway.eps','eps2c')
 
% Small Worldness
input_name_2 = 'A_aal_small_worldness.dat';
SM = load('A_aal_small_worldness.dat');
figure(6);
set(gca,'FontSize',30)
plot(SM(:,1),SM(:,8),'k','LineWidth',5)
hold on
for i = 1:length(random_G)
    A =load(strcat(input_name_2(1:6),'R',random_G(i),input_name_2(6:end)));
    if color(i) == 'g'
        plot(A(:,1),A(:,8),strcat(color(i),'^'),'LineWidth',6)  
    else
        plot(A(:,1),A(:,8),strcat(color(i),'o'),'LineWidth',5)
    end 
end
legend('Test Network', 'Ra','Rb','Rc','Rd')
legend('boxoff')
xlabel('Threshold [r]','FontSize',35)
ylabel('Small Worldness','FontSize',35)
hold off
set(figure(6), 'units', 'inches','position',[10 10 30 20]) 
set(gcf, 'PaperPositionMode','auto')
saveas(gcf,'Small_Worldness.eps','eps2c')

% Transitivity
figure(7);
set(gca,'FontSize',30)
plot(SM(:,1),SM(:,6),'k','LineWidth',5)
hold on
for i = 1:length(random_G)
    a = strcat(input_name_2(1:2),'R',random_G(i),input_name_2(2:end));
    A = load(strcat(input_name_2(1:6),'R',random_G(i),input_name_2(6:end)));
    plot(A(:,1),A(:,6),strcat(color(i),'o'),'LineWidth',5)
end
legend('Test Network', 'Ra','Rb','Rc','Rd')
legend('boxoff')
xlabel('Threshold [r]','FontSize',35)
ylabel('Transitivity','FontSize',35)
hold off
set(figure(7), 'units', 'inches','position',[10 10 30 20]) 
set(gcf, 'PaperPositionMode','auto')
saveas(gcf,'Transitivity.eps','eps2c')

% Global Efficiency of Network
input_name_3 = 'A_aal_global_efficiency_ave.dat';
GE = load('A_aal_global_efficiency_ave.dat');
figure(8);
set(gca,'FontSize',30)
plot(GE(:,1),GE(:,2),'k','LineWidth',5)
hold on
for i = 1:length(random_G)
    a = strcat(input_name_3(1:6),'R',random_G(i),input_name_3(6:end));
    A = load(a);
    if color(i) == 'g'
        plot(A(:,1),A(:,2),strcat(color(i),'^'),'LineWidth',6)  
    else
        plot(A(:,1),A(:,2),strcat(color(i),'o'),'LineWidth',5)
    end 
end
legend('Test Network', 'Ra','Rb','Rc','Rd')
legend('boxoff')
xlabel('Threshold [r]','FontSize',35)
ylabel('Average Global Efficiency','FontSize',35)
hold off
set(figure(8), 'units', 'inches','position',[10 10 30 20]) 
set(gcf, 'PaperPositionMode','auto')
saveas(gcf,'Global_Efficiency_Average.eps','eps2c')

% Local efficiency of Network
input_name_4 = 'A_aal_local_efficiency_ave.dat';
LE = load('A_aal_local_efficency_ave.dat');
figure(9);
set(gca,'FontSize',30)
plot(LE(:,1),LE(:,2),'k','LineWidth',5)
hold on
for i = 1:length(random_G)
   
    A =load(strcat(input_name_4(1:6),'R',random_G(i),'_local_efficency_ave.dat'));
    plot(A(:,1),A(:,2),strcat(color(i),'o'),'LineWidth',5)
end
legend('Test Network', 'Ra', 'Rb','Rc','Rd')
legend('boxoff')
xlabel('Threshold [r]','FontSize',35)
ylabel('Local Efficiency','FontSize',35)
hold off
set(figure(9), 'units', 'inches','position',[10 10 30 20]) 
set(gcf, 'PaperPositionMode','auto')
saveas(gcf,'Local_Efficiency_Average.eps','eps2c')

% DISTRIBUTIONS
% Global efficiency of nodes
GEF = load('A_aal_global_efficiency_node.dat');
in_name_5 = 'A_aal_global_efficiency_node.dat';
z_ = GEF(:,3);
z = zeros(101,90);
figure(10);
subplot(3,2,1)
for i = 1:101
    a = i-1;
   z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
end
n=(1:90);
m=(0:0.01:1);
set(gca,'FontSize',20)
imagesc(n,m,z)
h=colorbar; set(h,'fontsize',15);
set(gca,'XTick',15:15:90);
set(gca,'YTick',[0 0.25 0.50 0.75 1])
set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
xlabel('Nodes')
ylabel('Threshold [r]')
title('Global Efficiency of Nodes (Test Network)','FontSize',20)

for i =1:4
    A = load(strcat(in_name_5(1:6),'R',random_G(i),in_name_5(6:end)));
    b_ = A(:,3);

    subplot(3,2,i+1)
    set(gca,'FontSize',20)
    if random_G(i)~='d'
        b = zeros(101,90);
        for j = 1:101
            a = j-1;
            b(j,:) = b_( ( 90*a+1 :(90*a+90) ),:);
        end
        imagesc(n,m,b)
        set(gca,'YTick',[0 0.25 0.50 0.75 1])
    else
        c = zeros(76,90);
        for j = 1:76
            c(j,:) = b_( 90*(j-1)+1 : 90*j )';
        end
        imagesc((1:1:90),(0.25:0.01:1),c)
        set(gca,'YTick',[0.25 0.45 0.65 0.85])
    end
    h=colorbar; set(h,'fontsize',15); set(h,'fontsize',15);
    set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
    set(gca,'XTick',15:15:90);
    xlabel('Nodes')
    ylabel('Threshold [r]')
    title(strcat('Global Efficiency of Nodes, R',random_G(i)),'FontSize',20)
end
        
set(figure(10), 'units', 'inches','position',[10 10 15 20]) 
set(gcf, 'PaperPositionMode','auto')
saveas(gcf,'Global_Efficiency_Nodes.eps','eps2c')

% Local efficiency of nodes
LEF = load('A_aal_local_efficency_node.dat');
in_name_6 = 'A_aal_local_efficency_node.dat';
z_ = LEF(:,3);
z = zeros(101,90);
figure(11);
subplot(3,2,1)
for i = 1:101
    a = i-1;
   z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
end
n=(1:90);
m=(0:0.01:1);
set(gca,'FontSize',20)
imagesc(n,m,z)
h=colorbar; set(h,'fontsize',15);
set(gca,'XTick',15:15:90);
set(gca,'YTick',[0 0.25 0.50 0.75 1])
set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
xlabel('Nodes')
ylabel('Threshold [r]')
title('Local Efficiency of Nodes (Test Network)','FontSize',20)

for i =1:4
    A = load(strcat(in_name_6(1:6),'R',random_G(i),in_name_6(6:end)));
    b_ = A(:,3);
    subplot(3,2,i+1)
    set(gca,'FontSize',20)
    
    if random_G(i)~='d'
        b = zeros(101,90);
        for j = 1:101
            a = j-1;
            b(j,:) = b_( ( 90*a+1 :(90*a+90) ),:);
        end
        imagesc(n,m,b)
        set(gca,'YTick',[0 0.25 0.50 0.75 1])        
    else
        c = zeros(76,90);
        for j = 1:76
            c(j,:) = b_( 90*(j-1)+1 : 90*j )';
        end
            imagesc((1:1:90),(0.25:0.01:1),c)
            set(gca,'YTick',[0.25 0.45 0.65 0.85])

    end    
    set(gca,'XTick',15:15:90);
    set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
    h=colorbar; set(h,'fontsize',15);
    xlabel('Nodes')
    ylabel('Threshold [r]')
    title(strcat('Local Efficiency of Nodes, R',random_G(i)),'FontSize',20)
end

set(figure(11), 'units', 'inches','position',[10 10 15 20]) 
set(gcf, 'PaperPositionMode','auto')
saveas(gcf,'Local_Efficiency_Nodes.eps','eps2c')

% Connected Components of Nodes

% Local efficiency of nodes
COM = load('A_aal_connected_compo_node.dat');
in_name_7 = 'A_aal_connected_compo_node.dat';
z_ = COM(:,3);
z = zeros(101,90);
figure(12);
subplot(3,2,1)
for i = 1:101
    a = i-1;
   z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
end
n=(1:90);
m=(0:0.01:1);
set(gca,'FontSize',20)
imagesc(n,m,z)
h=colorbar; set(h,'fontsize',15);
set(gca,'XTick',15:15:90);
set(gca,'YTick',[0 0.25 0.50 0.75 1])
set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
xlabel('Nodes')
ylabel('Threshold [r]')
title('Connected Component of Nodes (Test Network)','FontSize',20)

for i =1:4
    A = load(strcat(in_name_7(1:6),'R',random_G(i),in_name_7(6:end)));
    b_ = A(:,3);
    subplot(3,2,i+1)
    set(gca,'FontSize',20)
    
    if random_G(i)~='d'
        b = zeros(101,90);
        for j = 1:101
            a = j-1;
            b(j,:) = b_( ( 90*a+1 :(90*a+90) ),:);
        end
        imagesc(n,m,b)
        set(gca,'YTick',[0 0.25 0.50 0.75 1])        
    else
        c = zeros(76,90);
        for j = 1:76
            c(j,:) = b_( 90*(j-1)+1 : 90*j )';
        end
            imagesc((1:1:90),(0.25:0.01:1),c)
            set(gca,'YTick',[0.25 0.45 0.65 0.85])

    end    
    set(gca,'XTick',15:15:90);
    set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
    h=colorbar; set(h,'fontsize',15);
    xlabel('Nodes')
    ylabel('Threshold [r]')
    title(strcat('Connected Component of Nodes, R',random_G(i)),'FontSize',20)
end

set(figure(12), 'units', 'inches','position',[10 10 15 20]) 
set(gcf, 'PaperPositionMode','auto')
saveas(gcf,'Connected_Components_Nodes.eps','eps2c')


% Degree Distribution
DD = load('A_aal_degree_dist.dat');
in_name_8 = 'A_aal_degree_dist.dat';
z_ = DD(:,3);
z = zeros(101,90);
figure(13);
subplot(3,2,1)
for i = 1:101
    a = i-1;
   z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
end
n=(1:90);
m=(0:0.01:1);
set(gca,'FontSize',20)
imagesc(n,m,z)
h=colorbar; set(h,'fontsize',15);
set(gca,'XTick',15:15:90);
set(gca,'YTick',[0 0.25 0.50 0.75 1])
set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
xlabel('Nodes')
ylabel('Threshold [r]')
title('Degree Distribution (Test Network)','FontSize',20)

for i =1:4
    A = load(strcat(in_name_8(1:6),'R',random_G(i),in_name_8(6:end)));
    b_ = A(:,3);
    subplot(3,2,i+1)
    set(gca,'FontSize',20)
    
    if random_G(i)~='d'
        b = zeros(101,90);
        for j = 1:101
            a = j-1;
            b(j,:) = b_( ( 90*a+1 :(90*a+90) ),:);
        end
        imagesc(n,m,b)
        set(gca,'YTick',[0 0.25 0.50 0.75 1])        
    else
        c = zeros(76,90);
        for j = 1:76
            c(j,:) = b_( 90*(j-1)+1 : 90*j )';
        end
            imagesc((1:1:90),(0.25:0.01:1),c)
            set(gca,'YTick',[0.25 0.45 0.65 0.85])

    end    
    set(gca,'XTick',15:15:90);
    set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
    h=colorbar; set(h,'fontsize',15);
    xlabel('Nodes')
    ylabel('Threshold [r]')
    title(strcat('Degree Distribution, R',random_G(i)),'FontSize',20)
end

set(figure(13), 'units', 'inches','position',[10 10 15 20]) 
set(gcf, 'PaperPositionMode','auto')
saveas(gcf,'Degree_Distribution.eps','eps2c')

% Clustering Coeficient of Nodes
cc_node = load('A_aal_cc_and_degree_node.dat');
in_name_9 = 'A_aal_cc_and_degree_node.dat';
z_ = cc_node(:,3);
z = zeros(101,90);
figure(14);
subplot(3,2,1)
for i = 1:101
    a = i-1;
   z(i,:) = z_( ( 90*a+1 :(90*a+90) ),:);
end
n=(1:90);
m=(0:0.01:1);
set(gca,'FontSize',20)
imagesc(n,m,z)
h=colorbar; set(h,'fontsize',15);
set(gca,'XTick',15:15:90);
set(gca,'YTick',[0 0.25 0.50 0.75 1])
set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
xlabel('Nodes')
ylabel('Threshold [r]')
title('Clustering Coefficient of Node (TN)','FontSize',20)

for i =1:4
    A = load(strcat(in_name_9(1:6),'R',random_G(i),in_name_9(6:end)));
    b_ = A(:,3);
    subplot(3,2,i+1)
    set(gca,'FontSize',20)
    
    if random_G(i)~='d'
        b = zeros(101,90);
        for j = 1:101
            a = j-1;
            b(j,:) = b_( ( 90*a+1 :(90*a+90) ),:);
        end
        imagesc(n,m,b)
        set(gca,'YTick',[0 0.25 0.50 0.75 1]); set(h,'YTick',0:0.2:1);
        h=colorbar; set(h,'fontsize',15); 
    else
        c = zeros(76,90);
        for j = 1:76
            c(j,:) = b_( 90*(j-1)+1 : 90*j )';
        end
            imagesc((1:1:90),(0.25:0.01:1),c)
            set(gca,'YTick',[0.25 0.45 0.65 0.85])
            h=colorbar; set(h,'fontsize',15); set(h,'YTick',0:0.2:1);
    end    
    set(gca,'XTick',15:15:90);
    set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%.2f'))
    xlabel('Nodes')
    ylabel('Threshold [r]')
    title(strcat('Clustering Coefficient of Node, R',random_G(i)),'FontSize',20)
end

set(figure(14), 'units', 'inches','position',[10 10 15 20]) 
set(gcf, 'PaperPositionMode','auto')
saveas(gcf,'Clustering_Coefficient_Node.eps','eps2c')