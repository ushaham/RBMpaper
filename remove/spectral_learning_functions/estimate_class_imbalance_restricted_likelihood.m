function b_hat = estimate_class_imbalance_restricted_likelihood(Z,delta)
    % b_hat = estimate_class_imbalance_restricted_likelihood(Z,delta)
    %
    % Estimate the class imbalance using the restricted likelihood method
    %
    % Input: 
    % Z -m x n matrix of binary data
    % delta - bounds away the class imbalance, psi and eta estimations
    %        b_hat in [-1+delta,1-delta], psi,eta in [delta,1-delta]
    %
    % Output: 
    % b - Class Y imbalance Pr(Y=1)-Pr(Y=-1)
    %
    % Written by Ariel Jaffe and Boaz Nadler, 2015
        
    %get number of classifiers
    m = size(Z,1);
    
    %estimate first moment
    mu = mean(Z,2);
        
    %estimate second moment
    R = cov(Z');
    
    %Create a single dimension matrix from R
    R = estimate_rank_1_matrix(R);
    
    %get first eigenvector
    [V, ~] = eigs(R,1);
    V = V*sign(sum(sign(V)));
    
    %get constant C for first eigenvector min(C*V*V'-R)
    R_v = V*V';
    Y = R( logical(tril(ones(m))-eye(m)) );
    X = R_v( logical(tril(ones(m))-eye(m)) );
    [~,C] = evalc('lsqr(X,Y)');
    V = V*sqrt(C);
    
    %Scan over b in [-1+delta, 1-delta]
    b_min = -1+delta;
    b_max =  1-delta;
    res   = delta;
    b_tilde_vec = b_min:res:b_max;
    
    pi_tilde = zeros(m,length(b_tilde_vec));
    delta_tilde = zeros(m,length(b_tilde_vec));
    psi_tilde = zeros(m,length(b_tilde_vec));
    eta_tilde = zeros(m,length(b_tilde_vec));
    restricted_ll = zeros(1,length(b_tilde_vec));
        
    for k = 1:length(b_tilde_vec)
        
        if ~mod(k,100)
            %disp(k)
        end
        
        %get values of pi,delta,psi and eta which correspond to b_tilde_vec(k)
        pi_tilde(:,k) = ((V/(1-b_tilde_vec(k)^2))+1)/2;
        delta_tilde(:,k) = (mu - (2*pi_tilde(:,k)-1)*b_tilde_vec(k))/2;        
        psi_tilde(:,k) = min(pi_tilde(:,k)+delta_tilde(:,k),1-delta);
        eta_tilde(:,k) = min(pi_tilde(:,k)-delta_tilde(:,k),1-delta);        
        psi_tilde(:,k) = max( psi_tilde(:,k),delta);
        eta_tilde(:,k) = max(eta_tilde(:,k),delta);
        
        %estimate log likelihood function: log(p*Pr(f|y=1)+(1-p)Pr(f|y=-1)
        ll_pos = zeros(size(Z));
        ll_neg = zeros(size(Z));
        p_k = (1+b_tilde_vec(k))/2;
        
        for i = 1:m
            
            %find positive and negative index
            pos_idx = find(Z(i,:)==1);
            neg_idx = find(Z(i,:)==-1);
            
            % find for each element Pr(f|y=1)
            ll_pos(i,pos_idx) = psi_tilde(i,k);
            ll_pos(i,neg_idx) = 1-psi_tilde(i,k);
            ll_neg(i,pos_idx) = 1-eta_tilde(i,k);
            ll_neg(i,neg_idx) = eta_tilde(i,k);
        end
        
        %estimate log likelihood function: log(p*Pr(f|y=1)+(1-p)Pr(f|y=-1)
        ll_vec = log(p_k*prod(ll_pos)+(1-p_k)*prod(ll_neg));        
        restricted_ll(k) = mean(ll_vec);        
    end
    
    
    %get maximum of restricted likelihood function
    [~,max_idx] = max(restricted_ll);
    b_hat = b_tilde_vec(max_idx);
        
end