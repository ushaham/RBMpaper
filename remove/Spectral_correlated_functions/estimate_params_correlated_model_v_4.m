function [b_hat, alpha_hat,psi_alpha_hat,eta_alpha_hat,psi_alpha_i,eta_alpha_i,ll_hat] = ...
        estimate_params_correlated_model_v_4(Z,clusters)
    
    delta = 0.01;
    
    %% 1. Step 1: get vectors v_out/vin
    [m,n]= size(Z);
    R = cov(Z');
    K = max(clusters);
    
    % estimate v_in, v_out by matrix completion
    R_c = rank_1_matrix_completion(R,clusters);
    [v_out,~] = eigs(R_c,1);
    not_same_cluster = bsxfun(@minus,clusters',clusters)~=0;
    
    %get constant C for first eigenvector min(C*V*V'-R)
    R_v = v_out*v_out';
    Y = R( logical(not_same_cluster) );
    X = R_v( logical(not_same_cluster) );
    [~,C] = evalc('lsqr(X,Y)');
    v_out = v_out*sqrt(C);
    v_out = sign(sum(sign(v_out)))*v_out;
    
    %% Step 2: estimate b by the tensor method
    A = [];
    B = [];
    Z_c = Z - repmat(mean(Z,2),1,n);
    for i = 1:m
        for j = 1:m
            for k = 1:m
                if (length(unique(clusters([ i j k])))==3)
                    A = [A ; v_out(i)*v_out(j)*v_out(k)];
                    T_ijk = mean(prod(Z_c([i j k],:),1));
                    B = [B ; T_ijk];
                end
            end
        end
    end
    alpha = lsqr(A,B);
    b_hat = -alpha / sqrt(4+alpha^2);
    
    
    %% Step 3: estimate psi,eta for all classifiers
    
    A = zeros(2*m+K,2*K);
    B = zeros(2*m+K,1);
    p_hat = (1+b_hat)/2;
    
    kappa_plus = sqrt( (1+b_hat)/(1-b_hat) );
    kappa_min = sqrt( (1-b_hat)/(1+b_hat) );
    psi_hat = 0.5*(1+mean(Z,2)+v_out*kappa_min);
    eta_hat = 0.5*(1-mean(Z,2)+v_out*kappa_plus);
    psi_hat = max(psi_hat,delta);psi_hat = min(psi_hat,1-delta);
    eta_hat = max(eta_hat,delta);eta_hat = min(eta_hat,1-delta);
    
    
    %% Step 4: estimate the parameters of the internal groups - psi_alpha_i,eta_alpha_i
    psi_alpha_i = zeros(m,1);
    eta_alpha_i = zeros(m,1);
    delta = 0.01;
    alpha_hat = zeros(1,K);
    for k = 1:K
        g_idx = find(clusters==k);
        if length(g_idx)>2
            %estimate parameters psi_alpha_i,eta_alpha_i and mean(alpha)
            alpha_hat(k) = estimate_class_imbalance_restricted_likelihood(Z(g_idx,:),delta);
            [~,psi_alpha_i(g_idx),eta_alpha_i(g_idx)] = estimate_ensemble_parameters(Z(g_idx,:),alpha_hat(k),delta);
        else
            alpha_hat(k) = b_hat;
            psi_alpha_i(g_idx) = psi_hat(g_idx);
            eta_alpha_i(g_idx) = eta_hat(g_idx);
        end
        
    end
    
    %% Step 5: estimate psi_alpha for all groups
    psi_alpha_hat = zeros(K,1);
    eta_alpha_hat = zeros(K,1);
    for k = 1:K
        
        g_idx = find(clusters==k);
        if length(g_idx)>2
            A = psi_alpha_i(g_idx)+eta_alpha_i(g_idx)-1;
            B = psi_hat(g_idx)+eta_alpha_i(g_idx)-1;
            psi_alpha_hat(k) = lsqr(A,B);
            
            A = psi_alpha_i(g_idx)+eta_alpha_i(g_idx)-1;
            B = eta_hat(g_idx)+psi_alpha_i(g_idx)-1;
            eta_alpha_hat(k) = lsqr(A,B);
        else
            psi_alpha_hat(k) = 1;
            eta_alpha_hat(k) = 1;
            
        end
        
    end
    psi_alpha_hat = max(psi_alpha_hat,delta);psi_alpha_hat = min(psi_alpha_hat,1-delta);
    eta_alpha_hat = max(eta_alpha_hat,delta);eta_alpha_hat = min(eta_alpha_hat,1-delta);
    
    
    %% Step 6: obtain likelihood
    if 1
        psi_mtx = (repmat(psi_alpha_i,1,n).^( (Z+1)/2)).*(repmat(1-psi_alpha_i,1,n).^( (1-Z)/2));
        eta_mtx = (repmat(eta_alpha_i,1,n).^( (1-Z)/2)).*(repmat(1-eta_alpha_i,1,n).^( (1+Z)/2));
        
        psi_prod = zeros(K,n);
        eta_prod = zeros(K,n);
        pos_ll = ones(1,n);
        neg_ll = ones(1,n);
        for k = 1:K
            g_idx = find(clusters==k);
            psi_prod(k,:) = prod(psi_mtx(g_idx,:),1);
            eta_prod(k,:) = prod(eta_mtx(g_idx,:),1);
            pos_ll = pos_ll.*(psi_alpha_hat(k)*psi_prod(k,:)+(1-psi_alpha_hat(k))*eta_prod(k,:) );
            neg_ll = neg_ll.*( (1-eta_alpha_hat(k))*psi_prod(k,:)+eta_alpha_hat(k)*eta_prod(k,:) );
        end
        ll_hat = mean( log(p_hat*pos_ll+(1-p_hat)*neg_ll));
        
    end
    
    %end
end