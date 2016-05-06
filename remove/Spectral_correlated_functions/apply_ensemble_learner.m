function y_hat = apply_ensemble_learner(Z,clusters,psi_z_f,eta_z_f,psi_y_z,eta_y_z)
   
    %
    K = max(clusters);
    n = size(Z,2);
    A = zeros(K,n);
    B = zeros(K,n);
    Pr_pos = zeros(K,n);
    Pr_neg = zeros(K,n);
    for k = 1:K
       g_idx = find(clusters==k); 
       A(k,:) = prod(repmat(psi_z_f(g_idx),1,n).^((1+Z(g_idx,:))/2).*repmat(1-psi_z_f(g_idx),1,n).^((1-Z(g_idx,:))/2),1);
       B(k,:) = prod(repmat(eta_z_f(g_idx),1,n).^((1-Z(g_idx,:))/2).*repmat(1-eta_z_f(g_idx),1,n).^((1+Z(g_idx,:))/2),1);
       Pr_pos(k,:) = psi_y_z(k)*A(k,:)+(1-psi_y_z(k))*B(k,:);
       Pr_neg(k,:) = (1-eta_y_z(k))*A(k,:)+eta_y_z(k)*B(k,:);
    end
    
    sum_log_pos = sum(log(Pr_pos),1);
    sum_log_neg = sum(log(Pr_neg),1);
    
    %Pr_pos = sum(log(repmat(psi_y_z,1,n).*A + repmat(1-psi_y_z,1,n).*B),1);
    %Pr_neg = sum(log(repmat(1-eta_y_z,1,n).*A + repmat(eta_y_z,1,n).*B),1);
    
    y_hat = sign(sum_log_pos-sum_log_neg);    
    
    
end