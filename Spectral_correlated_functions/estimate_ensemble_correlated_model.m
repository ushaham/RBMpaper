function [y_hat,psi_hat,eta_hat] =  estimate_ensemble_correlated_model(Z,clusters)
    %create ensemble learner based on correlated model
    % Input: 
    % Z - original prediction matrix
    % clusters - clusters of correlated classifiers
    % b - class imbalance
    % Output: 
    % y_hat - prediction of modified ensemble
    
    %% In order to estimate the parameters of each classifier more accurately - perform 
    %% matrix completion
    
    delta = 0.01;
    if 0 
    R = cov(Z');
    R_c = rank_1_matrix_completion(R,clusters);
    %get first eigenvector
    [V, ~] = eigs(R,1);
    V = V*sign(sum(sign(V)));
    
    %get constant C for first eigenvector min(C*V*V'-R)
    m = size(Z,1);
    R_v = V*V';
    Y = R( logical(tril(ones(m))-eye(m)) );
    X = R_v( logical(tril(ones(m))-eye(m)) );
    [~,C] = evalc('lsqr(X,Y)');
    V_hat = V*sqrt(C);
    
    %estimate psi and eta
    psi_hat = 0.5*(1+mean(Z,2)+V_hat*sqrt( (1-b)/(1+b)));
    eta_hat = 0.5*(1-mean(Z,2)+V_hat*sqrt( (1+b)/(1-b)));
    
    psi_hat = min(1-delta,psi_hat);
    eta_hat = min(1-delta,eta_hat);
    pi_hat = (psi_hat+eta_hat)/2;
    
    end
    %% first stage - go over all groups and receive prediction of every group
    y_pred = zeros(max(clusters),length(Z));
    if 1 
    
    for k = 1:max(clusters)
        
        %find classifiers of group k
        group_idx = find(clusters==k);
        %length(group_idx)
        if length(group_idx)>5
            
            b_hat = estimate_class_imbalance_restricted_likelihood(Z(group_idx,:),delta);
            
            %get paraeters of group k
            [V,psi_hat,eta_hat] = estimate_ensemble_parameters(Z(group_idx,:),b_hat,delta);
        
            %get prediction of group k 
            alpha = log( (psi_hat.*eta_hat)./((1-psi_hat).*(1-eta_hat)) );
            beta =  log( (psi_hat.*(1-psi_hat))./(eta_hat.*(1-eta_hat)) );
            y_pred(k,:) = sign(alpha'*Z(group_idx,:)+sum(beta));            
        else
            %if not enough classifiers in group to do spectral clustering -
            %just vote.
            y_pred(k,:) = sign(sum(Z(group_idx,:),1));
            y_pred(k,y_pred(k,:)==0)=1;
        end
    end
    
     
    %% stage two - perform spectral analysis on predictions
    b_hat = estimate_class_imbalance_restricted_likelihood(y_pred,delta);
    [V,psi_hat,eta_hat] = estimate_ensemble_parameters(y_pred,b_hat,delta);
        
    %get prediction of group k 
    alpha = log( (psi_hat.*eta_hat)./((1-psi_hat).*(1-eta_hat)) );
    beta =  log( (psi_hat.*(1-psi_hat))./(eta_hat.*(1-eta_hat)) );
    y_hat = sign(alpha'*y_pred+sum(beta)); 
    else
       %alternative way - just choose the best one from each group 
       for k = 1:max(clusters)
        group_idx = find(clusters==k);
        [~,best_in_group] = max(pi_hat(group_idx));
        y_pred(k,:) = Z(group_idx(best_in_group),:);
        
       end
       
           %% stage two - perform spectral analysis on predictions
        [V,psi_hat_g,eta_hat_g] = estimate_ensemble_parameters(y_pred,b,0.01);
        
        %get prediction of group k 
        alpha = log( (psi_hat_g.*eta_hat_g)./((1-psi_hat_g).*(1-eta_hat_g)) );
        beta =  log( (psi_hat_g.*(1-psi_hat_g))./(eta_hat_g.*(1-eta_hat_g)) );
        y_hat = sign(alpha'*y_pred+sum(beta)); 

    end
end
