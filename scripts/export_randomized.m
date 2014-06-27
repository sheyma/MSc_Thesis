
% iteration required for randmio_und.m
iter = 1000;
% partly define the name of adjacency matrices
name = 'A_aal_0_ADJ_thr_0.';
name_struc = 'acp_w_0_ADJ_thr_0.';
for i = 48:66  % has to be chosen manually
    
    input = strcat(name, num2str(i), '.dat' );
    % load the adjacency matrix
    adj_mtx = load(input);
    
    % randomize the adjacency matrix
    %ir_adj_mtx = randmio_und(adj_mtx , iter);    
    %filename_ir =strcat('A_aal_ir_ADJ_thr_0.', num2str(i), '.dat');
    %dlmwrite(filename_ir, ir_adj_mtx, 'delimiter','\t', 'precision', 1);
    
%     jr_adj_mtx = randmio_und_connected(adj_mtx , round(iter/10));
%     filename_jr =strcat('A_aal_jr_ADJ_thr_0.', num2str(i), '.dat');
%     dlmwrite(filename_jr, jr_adj_mtx, 'delimiter','\t', 'precision', 1);

%     for alpha = 0.1: 0.1 : 1;
%         a = num2str(alpha);
%         kr_adj_mtx = randomizer_bin_und(adj_mtx, alpha );
%         filename_kr = strcat('A_aal_kr_ADJ_thr_0.', num2str(i), '_alpha_', a, '.dat');
%         dlmwrite(filename_kr, kr_adj_mtx, 'delimiter','\t', 'precision', 1);
%     end
%     
    %lr_adj_mtx = randomizer_bin_und(......)
    
    swap = 500;
    input_stru = strcat(name, num2str(i), '.dat' )
    str_mtx = load(input_stru);
    lr_adj_mtx = randomize_graph_partial_und(adj_mtx , str_mtx , swap);
    filename_lr = strcat('A_aal_lr_ADJ_thr_0.', num2str(i), '.dat')
    dlmwrite(filename_lr, lr_adj_mtx, 'delimiter','\t', 'precision', 1);
        
end
