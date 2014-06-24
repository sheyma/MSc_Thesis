
% iteration required for randmio_und.m
iter = 100;
% partly define the name of adjacency matrices
name = 'A_aal_0_ADJ_thr_0.';
for i = 48:66  % has to be chosen manually
    
    input = strcat(name, num2str(i), '.dat' );
    % load the adjacency matrix
    adj_mtx = load(input);
    % randomize the adjacency matrix
    new_adj_mtx = randmio_und(adj_mtx , iter);    
    filename =strcat('A_aal_ir_ADJ_thr_0.', num2str(i), '.dat');
    % save randomized adjacency matrix as -ascii
    dlmwrite(filename, new_adj_mtx, 'delimiter','\t', 'precision', 1);
end
