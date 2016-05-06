%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Update score matrix
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function S = update_score_matrix(S,R_old,R_new)
    
    % for each of the elements in update_idx change the 
    % score matrix
    
    updated_idx = find(R_old~=R_new);
    
    m = size(R_old,1);
    
    for idx = 1:length(updated_idx)
        
        %get score and column of current idx
        i = mod(updated_idx(idx)-1,m)+1;
        j = floor((updated_idx(idx)-1)/m)+1;
        
        for k = 1:m
           for l = 1:m
               
               if (length(unique([i j k l]))==4)
                   %change score of S_{kl}
                   score_change = abs(R_new(i,j)*R_new(k,l)-R_new(i,l)*R_new(k,j))-...
                   abs(R_old(i,j)*R_old(k,l)-R_old(i,l)*R_old(k,j));
               
                   S(i,k) = S(i,k)+score_change;
                   S(j,l) = S(j,l)+score_change;
                   S(i,l) = S(i,l)+score_change;
                   S(j,k) = S(j,k)+score_change;                   
               end
           end
        end
    end    
end