%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate_score_function
% every cell in the matrix receives a 'score' as to the
% compatability of the cell to a rank-1 matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function S = generate_score_function(R,T)
    
    % generate score function:
    % input: covariance matrix R
    %        T -
    % output: Score function S
    
% go over all quadriplets    
m = size(R,1);
S = zeros(m);
ctr = 0;
sum_score = 0;

for i = 1:m
    %disp([ 'i = ' num2str(i)]); 
    for j = 1:m
        %disp([ 'i =' num2str(i) ', j = ' num2str(j)]); 
        for k = 1:m
            for l = 1:m
                if (length(unique([T(i) T(j) T(k) T(l)]))==4)
                    score = abs(R(i,k)*R(j,l)-R(i,l)*R(j,k));
                    sum_score = sum_score+score;
                    ctr = ctr+1;
                    S(i,k) = S(i,k)+score;
                    S(j,l) = S(j,l)+score;
                    S(i,l) = S(i,l)+score;
                    S(j,k) = S(j,k)+score;
                end                
            end
        end
    end
end
%score_avg =   sum_score/ctr;  
end