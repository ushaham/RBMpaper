function T = compute_classifier_3D_tensor(Z)
%function T = Compute_Classifier_3D_Tensor(Z)
%
% Computes tensor of joint covariance, from the prediction matrix Z
% Input: 
% Z - m x n matrix of binary classifier outputs
%
% Output: 
% T - 3D Tensor of joint covariance E[(f_i-mu_i)(f_j-mu_j)(f_k-mu_k)]
%
% Written by Ariel Jaffe and Boaz Nadler, 2015

[m n] = size(Z); 

% estimate third moment tenzor
T = zeros(m,m,m);

mu = mean(Z,2); 

for k_a = 1:m-2
    for k_b = k_a+1:m-1

        for k_c = k_b+1:m
            
            T(k_a,k_b,k_c) = n/(n-1)/(n-2) * sum( (Z(k_a,:)-mu(k_a) ).*(Z(k_b,:) - mu(k_b)).*(Z(k_c,:) - mu(k_c)) );
            T(k_a,k_c,k_b) = T(k_a,k_b,k_c);
            T(k_b,k_a,k_c) = T(k_a,k_b,k_c);
            T(k_b,k_c,k_a) = T(k_a,k_b,k_c);
            T(k_c,k_a,k_b) = T(k_a,k_b,k_c);
            T(k_c,k_b,k_a) = T(k_a,k_b,k_c);            
        end
    end
end
