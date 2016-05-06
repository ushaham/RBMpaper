% the graph adjacency matrices, edge weigts are P(h=1 | y=1) for psi 
% and P(h=0|y=0) for eta.
% V(from,to) is the direction of edges (compatibility with biograph)
function [V_psi,V_eta,V_weights] = ...
    generateNetworkStructure (m,  min_psi, min_eta, num_nodes_per_layer)
    %% Init
    total_num_nodes = 1+sum(num_nodes_per_layer);
    if m ~= num_nodes_per_layer(end)
        % m = number of predictors / number of leaves in graph
        fprintf('\n\nError: total number of leafs needs to be m\n');
        return;
    end
    
    %% Generate dependency graph
    V_psi = zeros(total_num_nodes); % add 1 node for the true y
    V_eta = zeros(total_num_nodes);
    V_weights = zeros(total_num_nodes);

    layers = [1 num_nodes_per_layer];
    start_parent = 1;
    start_cur_layer = 2;

    for cur_layer=2:length(layers)
        % size of the sub-matrix that represents the adjancencies between
        % this layer and the next.
        block_size = [layers(cur_layer-1),layers(cur_layer)];

        % the rows and columns of the full matrix that belong to this
        % sub-matrix.
        r = start_parent:(start_parent + block_size(1) - 1);
        c = start_cur_layer:(start_cur_layer + block_size(2) - 1);

        % generate the eta's and psi's for this sub-matrix
        cur_min_psi = min_psi(min(cur_layer-1, length(min_psi)));
        cur_min_eta = min_eta(min(cur_layer-1, length(min_eta)));
        V_psi(r,c) = rand(block_size) * (1-cur_min_psi) + cur_min_psi;
        V_eta(r,c) = rand(block_size) * (1-cur_min_eta) + cur_min_eta;
        weights_rc = rand(length(r),length(c));
        
        % normalize the incoming edges weights to 1 (sum of every column
        % should be 1).
        col_sums = sum(weights_rc,1);
        for i=1:length(col_sums)
            weights_rc(:,i) = weights_rc(:,i) ./ col_sums(i);
        end
        V_weights(r,c) = weights_rc;

        % advance to the next sub-matrix
        start_parent = start_parent + block_size(1);
        start_cur_layer = start_cur_layer + block_size(2);
    end

%     G_psi = biograph(V_psi,[],'ShowWeights','on');
%     G_eta = biograph(V_eta,[],'ShowWeights','on');
%     G_weights = biograph(V_weights,[],'ShowWeights','on');

%     close all hidden;    
%     view(G_psi); view(G_eta); view(G_weights);
end