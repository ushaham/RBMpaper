function [b_hat, alpha_hat,psi_alpha,eta_alpha,psi_alpha_i,eta_alpha_i] = ...
        estimate_params_correlated_model_v_3(Z,clusters,psi,eta,psi_y_z,eta_y_z,psi_z_f,eta_z_f)
    
    [m,n]= size(Z);
    R = cov(Z');
    
    
    %% estimate v_in, v_out by matrix completion
    R_c = rank_1_matrix_completion(R,clusters);
    [v_out,~] = eigs(R_c,1);
    not_same_cluster = bsxfun(@minus,clusters',clusters)~=0;
    %same_cluster = 1-not_same_cluster;
    %same_cluster(logical(eye(m)))=0;
    
    %get constant C for first eigenvector min(C*V*V'-R)
    R_v = v_out*v_out';
    Y = R( logical(not_same_cluster) );
    X = R_v( logical(not_same_cluster) );
    [~,C] = evalc('lsqr(X,Y)');
    v_out = v_out*sqrt(C);
    v_out = sign(sum(sign(v_out)))*v_out;
    
    v_in = zeros(m,1);
    for k = 1:max(clusters)
        
        idx = find(clusters==k);
        if length(idx)>2
            l = length(idx);
            R_in = R(idx,idx);
            R_c = rank_1_matrix_completion(R_in,1:l);
            [v_in(idx),~] = eigs(R_c,1);
            
            %get constant C for first eigenvector min(C*V*V'-R)
            
            R_v = v_in(idx)*v_in(idx)';
            Y = R_in( logical(tril(ones(l))-eye(l)) );
            X = R_v( logical(tril(ones(l))-eye(l)) );
            [~,C] = evalc('lsqr(X,Y)');
            v_in(idx) = v_in(idx)*sqrt(C);
            v_in(idx) = sign(sum(sign(v_in(idx))))*v_in(idx);
        else
            v_in(idx) = v_out(idx);
        end
    end
    
    
    %% check - reconstruct R
    R_r = zeros(m);
    for i = 1:m
        for j = 1:m
            if clusters(i)==clusters(j)
                R_r(i,j) = v_in(i)*v_in(j);
            else
                R_r(i,j) = v_out(i)*v_out(j);
            end
        end
    end
    
    
    
    %% estimate the parameters of the classifiers
    delta = 0.01;
    b_vec = -1+delta:delta:1-delta;
    b_vec = -0.1:delta:0.1;
    alpha_vec = -1+delta:delta:1-delta;
    %alpha_vec = -0.1:delta:0.1;
    
    %initialize
    psi_alpha_im = zeros(max(clusters),1);
    eta_alpha_im = zeros(max(clusters),1);
    psi_alpha_i_im = zeros(m,1);
    eta_alpha_i_im = zeros(m,1);
    im_pos_ll = zeros(max(clusters),n);
    im_neg_ll = zeros(max(clusters),n);
    ll_b = zeros(1,length(b_vec));
    ll = zeros(1,length(alpha_vec));
    alpha_hat_im = zeros(max(clusters),length(b_vec));
    
    groups = zeros(1,max(clusters));
    clust_vec = clusters;
    for k = 1:max(clusters)
        groups(k) =  clust_vec(1);
        clust_vec(clust_vec==groups(k))=[];
    end
    
    for b_idx = 1:length(b_vec)
        tic
        b_hat = b_vec(b_idx);
        %b_hat = 0;
        p_hat = (1+b_hat)/2;
        
        %get values of psi_i,eta_i for all classifiers
        psi_hat = 0.5*(1+mean(Z,2)+v_out*sqrt( (1-b_hat)/(1+b_hat)));
        eta_hat = 0.5*(1-mean(Z,2)+v_out*sqrt( (1+b_hat)/(1-b_hat)));
        psi_hat = min(psi_hat,1-delta);
        eta_hat = min(eta_hat,1-delta);
        psi_hat = max(psi_hat,delta);
        eta_hat = max(eta_hat,delta);
        
        
        
        % go over all groups
        for k = 1:max(clusters)
            
            %find elements inside group
            g_idx = find(clusters==k);
            %g_num = find(unique(clusters,'stable')==k);
            %g_num = groups(k);
            
            if length(g_idx)>1
                
                % go over all values of E[alpha]
                for alpha_idx = 1:length(alpha_vec)
                    alpha_hat = alpha_vec(alpha_idx);
                    %alpha_hat = 0;
                    
                    %get values of all psi_alpha_i and eta_alpha_i
                    [psi_alpha,eta_alpha,psi_alpha_i,eta_alpha_i] = ...
                        get_internal_group_values(alpha_hat,Z(g_idx,:),v_in(g_idx),psi_hat(g_idx),eta_hat(g_idx));
                    
                    %obtain likelihood
                    p_pos = p_hat*psi_alpha+(1-p_hat)*(1-eta_alpha);
                    p_neg = p_hat*(1-psi_alpha)+(1-p_hat)*eta_alpha;
                    
                    ll(alpha_idx) = mean( log(...
                        p_pos*prod( (repmat(psi_alpha_i,1,n).^( (Z(g_idx,:)+1)/2))...
                        .*( repmat(1-psi_alpha_i,1,n).^( (1-Z(g_idx,:))/2)),1)+...
                        (1-p_pos)*prod( (repmat(eta_alpha_i,1,n).^( (1-Z(g_idx,:))/2))...
                        .*( repmat(1-eta_alpha_i,1,n).^( (1+Z(g_idx,:))/2)),1)...
                        ));
                end
                
                %take the maximum value of k,
                [~,max_idx] = max(ll);
                [psi_alpha_im(k),eta_alpha_im(k),psi_alpha_i_im(g_idx),eta_alpha_i_im(g_idx)] = ...
                    get_internal_group_values(alpha_vec(max_idx),Z(g_idx,:),v_in(g_idx),psi_hat(g_idx),eta_hat(g_idx));
                
                %get intermediate probabilites
                im_pos_ll(k,:) = psi_alpha_im(k)*prod( (repmat(psi_alpha_i_im(g_idx),1,n).^( (Z(g_idx,:)+1)/2))...
                    .*( repmat(1-psi_alpha_i_im(g_idx),1,n).^( (1-Z(g_idx,:))/2)),1)+...
                    (1-psi_alpha_im(k))*prod( (repmat(eta_alpha_i_im(g_idx),1,n).^( (1-Z(g_idx,:))/2))...
                    .*( repmat(1-eta_alpha_i_im(g_idx),1,n).^( (1+Z(g_idx,:))/2)),1);
                
                im_neg_ll(k,:) = (1-eta_alpha_im(k))*prod( (repmat(psi_alpha_i_im(g_idx),1,n).^( (Z(g_idx,:)+1)/2))...
                    .*( repmat(1-psi_alpha_i_im(g_idx),1,n).^( (1-Z(g_idx,:))/2)),1)+...
                    (eta_alpha_im(k))*prod( (repmat(eta_alpha_i_im(g_idx),1,n).^( (1-Z(g_idx,:))/2))...
                    .*( repmat(1-eta_alpha_i_im(g_idx),1,n).^( (1+Z(g_idx,:))/2)),1);
                
                %save itermediate value for alpha
                alpha_hat_im(k,b_idx) = alpha_vec(max_idx);
            else
                alpha_hat_im(k,b_idx) = b_hat;
                psi_alpha_im(g_idx) = 1;
                eta_alpha_im(g_idx) = 1;
                %alpha_hat = b_hat;
                
                im_pos_ll(k,:) = (psi_hat(g_idx).^((1+Z(g_idx,:))/2)).*( (1-psi_hat(g_idx)).^( (1-Z(g_idx,:))/2) );
                im_neg_ll(k,:) = ( (1-eta_hat(g_idx)).^((1+Z(g_idx,:))/2)).*( (eta_hat(g_idx)).^( (1-Z(g_idx,:))/2) );
                
            end
            
        end
        
        %obtain likelihood for b
        ll_b(b_idx) = mean(log(p_hat*prod(im_pos_ll,1)+(1-p_hat)*prod(im_neg_ll,1)));
        toc
    end
    
    %take the final value of b, mean alpha and all the statistics of the
    %problem
    [~,max_b_idx] = max(ll_b);
    b_hat = b_vec(max_b_idx);
    alpha_hat = alpha_hat_im(:,max_b_idx);
    
    for k = 1:max(clusters)
        g_idx = find(clusters==k);
        %g_num = groups(k);
        %g_num = find(unique(clusters,'stable')==k);        
        psi_hat = 0.5*(1+mean(Z,2)+v_out*sqrt( (1-b_hat)/(1+b_hat)));
        eta_hat = 0.5*(1-mean(Z,2)+v_out*sqrt( (1+b_hat)/(1-b_hat)));
        if length(g_idx)>2
            [psi_alpha(k),eta_alpha(k),psi_alpha_i(g_idx),eta_alpha_i(g_idx)] = ...
                get_internal_group_values(alpha_hat(k),Z(g_idx,:),v_in(g_idx),psi_hat(g_idx),eta_hat(g_idx));
        else
            psi_alpha(k) = 1;
            eta_alpha(k) = 1;
            psi_alpha_i(g_idx) = psi_hat(g_idx);
            eta_alpha_i(g_idx) = eta_hat(g_idx);
        end
    end
end

%%
% function [psi_alpha,eta_alpha,psi_alpha_i,eta_alpha_i] = ...
%              get_internal_group_values(alpha_hat,Z,v_in,eta_hat,psi_hat)
% Input: alpha_hat - value of intermediate
%        Z - Inputs of values in group
%        v_in - values of vector

function [psi_alpha,eta_alpha,psi_alpha_i,eta_alpha_i] = ...
        get_internal_group_values(alpha_hat,Z,v_in,eta_hat,psi_hat)
    delta = 0.01;
    
    %get values of all psi_alpha_i and eta_alpha_i
    psi_alpha_i = 0.5*(1+mean(Z,2)+v_in*sqrt( (1-alpha_hat)/(1+alpha_hat)));
    eta_alpha_i = 0.5*(1-mean(Z,2)+v_in*sqrt( (1+alpha_hat)/(1-alpha_hat)));
    %psi_alpha_i = min(psi_alpha_i,1-delta);eta_alpha_i = min(eta_alpha_i,1-delta);
    psi_alpha_i = min(psi_alpha_i,1-delta);eta_alpha_i = min(eta_alpha_i,1-delta);
    psi_alpha_i = max(psi_alpha_i,delta);eta_alpha_i = max(eta_alpha_i,delta);
    
    %get valus of psi_alpha and eta_alpha by LS - Ax = B;
    B = psi_hat +eta_alpha_i-1;
    A = psi_alpha_i+eta_alpha_i-1;
    %psi_alpha = lsqr(A,B);
    [~,psi_alpha] = evalc('lsqr(A,B)');
    psi_alpha = min(psi_alpha,1-delta);
    psi_alpha = max(psi_alpha,delta);
    %res_psi(alpha_idx) = norm(A*psi_alpha-B);
    
    
    B = eta_hat +psi_alpha_i-1;
    A = psi_alpha_i+eta_alpha_i-1;
    %eta_alpha = lsqr(A,B);
    [~,eta_alpha] = evalc('lsqr(A,B)');
    eta_alpha = min(eta_alpha,1-delta);
    eta_alpha = max(eta_alpha,delta);
    %res_eta(alpha_idx) = norm(A*eta_alpha-B);
    
end