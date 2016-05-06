%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function R_c = rank_1_matrix_completion(R,T)
    %  function: matrix_completion:
    %   Input: R - symmetrical covariance matrix R
    %          T - groups of elements in R.
    
    m = size(R,1);
    R_c = R;
    for i = 1:m
        for j = 1:m
            
            %check if need completion
            if (T(i) == T(j))
                
                %go over all matrix
                A = [];
                B = [];
                for k = 1:m
                    for l = 1:m
                        if (T(k)~=T(j)) && (T(k)~=T(l)) && (T(i)~=T(l))
                            B = [B ; R(i,l)*R(k,j)];
                            A = [A ; R(k,l)];
                        end                        
                    end
                end
                
                %estimate R(i,j)
                [~,R_c(i,j)] = evalc('lsqr(A,B)');
            end
            
            
        end
    end