function [f,y] = generateTreeDependentData(tree_struct, m, n, prob_y_is_1)
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
    %% Init
    f = zeros(m,n); % data matrix
    y = zeros(n,1); % true labels
    
    %% Generate f,y
    y = rand(n,1) > 1-prob_y_is_1;
    
    instance = cell(n,1);
    dfs_idxs = tree_struct.depthfirstiterator;

    for i=1:n
        instance{i} = tree(tree_struct,'clear'); % generate an instance of the data
        instance{i} = instance{i}.set(1, y(i));
        sample_vec = zeros(m,1);
        sample_vec_itr = 1;
        
        % generate values for every layer
        for node=dfs_idxs(2:end)
            acc = tree_struct.get(node);
            psi = acc(1); eta = acc(2);
            if instance{i}.get(instance{i}.getparent(node)) == 1
                prob = 1-psi;
            else
                prob = eta;
            end
            val = rand > prob;
            instance{i} = instance{i}.set(node,val); 
            
            % Update the current sample
            if tree_struct.isleaf(node)
                sample_vec(sample_vec_itr) = val;
                sample_vec_itr = sample_vec_itr + 1;
            end
        end
        f(:,i) = sample_vec';
    end
end