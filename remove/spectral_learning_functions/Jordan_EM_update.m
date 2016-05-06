function pi_em = Jordan_EM_update(f,mu,y_real)
    
    pos_idx = find(y_real ==1);
    neg_idx = find(y_real==-1);
    
    %conver f to Z
    [m n] = size(f);
    Z = zeros(n,2,m);
    idx_mtx = (f+3)/2; 
    for k_a = 1:n
        for k_b = 1:10
            Z(k_a,idx_mtx(k_b,k_a),k_b)=1;
        end
    end
    k = 2;
    Nround = 10;
    
    for iter = 1:Nround
        q = zeros(n,k);
        for j = 1:n
            for c = 1:k
                for i = 1:m
                    if Z(j,:,i)*mu(:,c,i) > 0
                        q(j,c) = q(j,c) + log(Z(j,:,i)*mu(:,c,i));
                    end
                end
            end
            q(j,:) = exp(q(j,:));
            q(j,:) = q(j,:) / sum(q(j,:));
        end
        
        for i = 1:m
            mu(:,:,i) = (Z(:,:,i))'*q;
            
            mu(:,:,i) = AggregateCFG(mu(:,:,i),0);
            for c = 1:k
                mu(:,c,i) = mu(:,c,i)/sum(mu(:,c,i));
            end
        end
        
        [I J] = max(q');
        J = 2*(J-1.5);
        %error2_predict(iter) = mean(y_real ~= J);
        
    end
    %Get properties of meta classifiers
    psi_j = length(find(J(pos_idx)==1))/length(pos_idx);
    eta_j = length(find(J(neg_idx)==-1))/length(neg_idx);
    pi_em = (psi_j+eta_j)/2;   
    
end