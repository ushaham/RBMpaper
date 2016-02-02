function [V_hat,psi_hat,eta_hat] = estimate_ensemble_parameters(Z,b,delta)
    % [V_hat,psi_hat,eta_hat] = estimate_ensemble_parameters(Z,b)
    % 
    % Estimate the sensitivities (psi) and specificities (eta) of the
    % classifiers
    %
    % Input: 
    % Z - Prediction matrix
    % b - class imbalance
    % delta - restrict psi,eta between delta and 1-delta
    % Output: 
    % V_hat - first eigenvector of the covariance matrix Z
    % psi_hat - estimated sensitivity of m classifiers
    % eta_hat - estimated specificity of m classifiers.
    %
    % Written by Ariel Jaffe and Boaz Nadler, 2015
    
    m = size(Z,1);
    
    %estimate mean
    mu = mean(Z,2);
    
    %estimate covariance matrix 
    R = cov(Z');
    
    % estimate the diagonal values of a single rank matrix
    R = estimate_rank_1_matrix(R);
    
    %get first eigenvector
    [V, ~] = eigs(R,1);
    V = V*sign(sum(sign(V)));
    
    %get constant C for first eigenvector min(C*V*V'-R)
    R_v = V*V';
    Y = R( logical(tril(ones(m))-eye(m)) );
    X = R_v( logical(tril(ones(m))-eye(m)) );
    [~,C] = evalc('lsqr(X,Y)');
    V_hat = V*sqrt(C);
    
    %estimate psi and eta
    psi_hat = 0.5*(1+mu+V_hat*sqrt( (1-b)/(1+b)));
    eta_hat = 0.5*(1-mu+V_hat*sqrt( (1+b)/(1-b)));
    
    psi_hat = min(1-delta,psi_hat);
    eta_hat = min(1-delta,eta_hat);
    
    psi_hat = max(delta,psi_hat);
    eta_hat = max(delta,eta_hat);
    
    
end