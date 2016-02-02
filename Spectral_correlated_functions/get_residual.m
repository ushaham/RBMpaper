function res = get_residual(R,clusters)
    % The function receives a covariance matrix and computes the residual
    % Input: R - covariance matrix
    %        clusters - cluster vector c(i)
    % Output: residual
    
    %get outer block error
    m = size(R,1);
    R_out = rank_1_matrix_completion(R,clusters);
    not_same_cluster = bsxfun(@minus,clusters',clusters)~=0;
    same_cluster = 1-not_same_cluster;
    same_cluster(logical(eye(m)))=0;
    [V,~] = eigs(R_out,1);
    %err_out = norm( trace(R_out)*V*V'-R_out,'fro');
    err_out = norm( (trace(R_out)*V*V'-R_out).*not_same_cluster,'fro')^2;
    
    %get inner block error
    R_in = zeros(m);
    for k = 1:max(clusters)
        c_idx = find(clusters==k);
        if length(c_idx)>2
            R_block_k = rank_1_matrix_completion(R(c_idx,c_idx),1:length(c_idx));
            [V,~] = eigs(R_block_k,1);
            R_in(c_idx,c_idx) = trace(R_block_k)*V*V';
        else
            R_in(c_idx,c_idx) = R(c_idx,c_idx);
        end
    end
    err_in = norm((R_in-R).*same_cluster,'fro')^2;
    res = err_in+err_out;
end
