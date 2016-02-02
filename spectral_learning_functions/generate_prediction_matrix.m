function [y,Z] = generate_prediction_matrix(m,n,b,psi,eta)
    % function [y,Z] = generate_prediction_matrix(m,n,b,psi,eta)
    % Generates random binary data - true label vector y and prediction
    % matrix Z.
    %
    % input: 
    % b - class imbalance of class y
    % m - Number of classifiers
    % n - Number of instances
    % psi - Sensitivity of m classifiers
    % eta - specificity of n classifiers
    %
    % output:
    % y - vector of true labels 
    % Z - prediction matrix of m classifiers for n instances
    %
    % Written by Ariel Jaffe and Boaz Nadler, 2015
    
    %generate true lable vector y: Pr(y_j = 1) = p
    p = (1+b)/2;
    y = randsrc(1,n,[1 -1 ; p (1-p)]);
    
    %get indices for true and negative elements
    pos_idx = find(y==1);
    neg_idx = find(y==-1);
    
    %generate prediction matrix Z_ij according to y_j and psi_i/eta_i 
    Z = zeros(m,n);
    Z(:,pos_idx) = binornd(1,repmat(psi,1,length(pos_idx)));
    Z(:,neg_idx) = 1-binornd(1,repmat(eta,1,length(neg_idx)));
    Z = 2*Z-1;
    
    
end