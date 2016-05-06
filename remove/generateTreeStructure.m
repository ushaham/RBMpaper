function [tree_struct] = generateTreeStructure(m, n, min_psi, min_eta, num_nodes_per_layer)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Generates tree dependence
%   num_nodes_per_layer is a vector with the number of children at every
%   layer. (for now it generates a balanced tree).
%
%   Example: generateTreeDependentData(12,100,0.5,0.6,0.5,[2 2 1 3])
%   m=12;n=100;prob_y_is_1=0.5;min_psi=[0.6 0.9];min_eta=[0.5 0.9];
%   num_nodes_per_layer=[2 2 1 3];
%
%   Example 2: generateTreeDependentData(12,100,0.5,[0.6 0.9],[0.5 0.9],[2 2 1 3])
%   m=12;n=100;prob_y_is_1=0.5;
%   min_psi=[0.6 0.9];min_eta=[0.5 0.9];
%   num_nodes_per_layer=[2 2 1 3];
%
%   Example 2 will create weak dependence for the first layer of hidden
%   variables (eta/psi > 0.5), and a strong dependence (>0.9) for the move
%   in every other layer.
%
%   For help with the tree data structure use: 
%       https://tinevez.github.io/matlab-tree/
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (prod(num_nodes_per_layer) ~= m)
        fprintf('\n\nError: total number of leafs needs to be m\n');
        return;
    end
    
    %% Build network data structure
    tree_struct = tree('y');
    % Add children
    next_layer = cell(num_nodes_per_layer(1),1);
    for idx = 1:num_nodes_per_layer(1)
        psi = rand*(1-min_psi(1)) + min_psi(1);
        eta = rand*(1-min_eta(1)) + min_eta(1);
        [tree_struct next_layer{idx}] = tree_struct.addnode(1, [psi, eta]);
    end

    layer = 2;
    while layer <= length(num_nodes_per_layer)
        layer_psi = min(layer, length(min_psi));
        layer_eta = min(layer, length(min_eta));
        this_layer = next_layer; % keep track of the nodes in the next layer
        next_layer = cell(numel(this_layer) * num_nodes_per_layer(layer),1);
        start = 0;
        for node_idx=1:length(this_layer) % iterate over the current layer's nodes
            node = this_layer{node_idx};
            % Add children for every node
            for idx = 1:num_nodes_per_layer(layer)
                psi = rand*(1-min_psi(layer_psi)) + min_psi(layer_psi);
                eta = rand*(1-min_eta(layer_eta)) + min_eta(layer_eta);
                [tree_struct next_layer{start+idx}] = tree_struct.addnode(node, [psi, eta]);
            end
            start = start+idx;
        end
        layer = layer + 1;
    end
    disp(tree_struct.tostring)
end