function alpha = estimate_alpha(V,T)
%function alpha3 = Estimate_alpha(V,T)
% Estimate scalar parameter alpha
%
% Input: 
% V - First eigenvector of covariance matrix V
% T - Tensor of joint covariance
%
% Output:
% alpha - 
% min sum_{ijk} (Tijk - alpha3 * v_ijk)^2
% d/d alpha gives sum (vijk Tijk) / sum(vijk^2)
%
% Written by Ariel Jaffe and Boaz Nadler, 2015

m = length(V); 

s1 = 0; s2 = 0; 
for i=1:(m-2)
    for j=(i+1):(m-1)
        for k=(j+1):m
            s1 = s1 + T(i,j,k)*V(i)*V(j)*V(k);
            s2 = s2 + (V(i)*V(j)*V(k))^2; 
        end
    end
end

alpha = s1 / s2; 
