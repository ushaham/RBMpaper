function R = estimate_rank_1_matrix(Q)
% function [R] = estimate_rank_1_matrix(R)
%
% Estimate the diagonal entries of matrix Q by assuming a rank-1 structure
%
% Input: 
% Q - matrix of m x m with off-diagonal rank one entries
%
% Ooutput:
% R - same matrix with estimated diagonal entries
%
% Written by Ariel Jaffe and Boaz Nadler, 2015
    
    m = size(Q,1);
    
    %number of equations
    N = 3*nchoosek(m,3);
    
    A = zeros(N,m);
    B = zeros(N,1);
    ctr = 0;
    for i = 1:m
       for j = i+1:m
           for k = j+1:m
               ctr = ctr+1;
               A(ctr,k)=Q(i,j);
               B(ctr) = Q(j,k)*Q(i,k);
               ctr = ctr+1;
               A(ctr,i)=Q(j,k);
               B(ctr) = Q(i,j)*Q(i,k);
               ctr = ctr+1;
               A(ctr,j)=Q(i,k);
               B(ctr) = Q(i,j)*Q(j,k);
            
           end
       end
    end
    
    [~,X] = evalc('lsqr(A,B)');
        
    R = Q;
    R(logical(eye(size(R)))) = X;
end