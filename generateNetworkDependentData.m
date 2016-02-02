function [f,y] = generateNetworkDependentData(m, n, prob_y_is_1, num_nodes_per_layer, V_psi, V_eta, V_weights)
    if nargin==0
        m=15; n=5000; prob_y_is_1 = 0.5; num_nodes_per_layer = [1,3,15];
    end
    %% Init
    f = zeros(m,n); % data matrix
    y = zeros(n,1); % true labels

    total_num_nodes = 1+sum(num_nodes_per_layer);
    if m ~= num_nodes_per_layer(end)
        % m = number of predictors / number of leaves in graph
        fprintf('\n\nError: total number of leafs needs to be m\n');
        return;
    end
    
    %% Randomly Generate f,y
    y = rand(n,1) > 1-prob_y_is_1;
    nodes = -1*ones(total_num_nodes,1);
    
    for sample=1:n
        nodes(1) = y(sample);
        
        % we need to select a label for every node in the graph
        for cur_node=2:total_num_nodes 
            
            p = 0; % p = Pr(cur_node = 1)
            % loop over all parent nodes (of cur_node)
            parent_idxs = (find(V_psi(:,cur_node) ~= 0))';
            for parent = parent_idxs
                if nodes(parent) == 1
                    p = p + V_psi(parent, cur_node) * V_weights(parent,cur_node);
                    
                elseif nodes(parent) == 0
                    p = p + (1 - V_eta(parent, cur_node)) * V_weights(parent,cur_node);
                    
                else
                    % FAIL: We got to a child node before its parent
                    throw(MException('genNetDep:ParentUninitialized', ...
                        'Parent not initialized. Graph structure error.'));
                end
            end
            
            nodes(cur_node) = rand < p;
        end % cur_node=2:total_num_nodes 
        f(:,sample) = nodes(end-m+1:end);
    end % for sample=1:n
end