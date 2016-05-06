

function [R_c,group_idx] = get_contaminated_elements(S,R,threshold)
    
    m = size(S,1);
    %T = 1:m;
    R_c = R;
    S_c = S;
    
    total_score = sum(sum(S_c));
    normalized_score = sum(sum(S_c))/(m*(m-1));
    group_idx = 1:m;
    R_c = rank_1_matrix_completion(R_c,group_idx);
    [~,D] = eigs(R_c,2);
    eigenvalue = diag(D);
    ratio = abs(eigenvalue(2)/eigenvalue(1));
    
    
    %while  (total_score(end)>threshold)
    while  (ratio>threshold)
        %while (1)
        %imagesc(S_c);grid on;colorbar;
        %close all;
        %find index of highest score value
        g_row =0;g_col = 0;
        while (g_row==g_col)
            [val,idx] = max(S_c(:));
            row = mod(idx-1,m)+1;
            col = floor((idx-1)/m)+1;
            
            %find group of row and column
            g_row = group_idx(row);
            g_col = group_idx(col);
            
            S_c(row,col)=0;
        end
        
        %update groups
        
        
        %change the group number of the higher to the lower one.
        max_idx = find(group_idx==max(g_row,g_col));
        group_idx(max_idx) = min(g_row,g_col);
        
        %take one off the indices of the higher groups
        group_idx(group_idx>max(g_col,g_row)) =...
            group_idx(group_idx>max(g_col,g_row))-1;
        
        
        %re-asses the values of the
        R_c_new = rank_1_matrix_completion(R_c,group_idx);
        %S_c = generate_score_function(R_c,1:15);
        S_c = update_score_matrix(S_c,R_c,R_c_new);
        R_c = R_c_new;
        
        [V,D] = eigs(R_c,2);
        eigenvalue = [eigenvalue  diag(D)];
        total_score = [total_score  sum(sum(S_c))];
        
        %calc number of elements
        num_elements = m*(m-1);
        for k = 1:max(group_idx)
            num_elements = num_elements-...
                sum(group_idx == k)*(sum(group_idx == k)-1);
        end
        normalized_score = [normalized_score ; total_score(end)/num_elements];
        ratio = abs(eigenvalue(2,end)/eigenvalue(1,end));
    end
    
    if 0
        subplot(3,1,1)
        plot(1:6,eigenvalue(2,:)./eigenvalue(1,:),'s');
        grid on;
        subplot(3,1,2)
        plot(1:6,total_score,'s');
        grid on;
        subplot(3,1,3)
        plot(1:6,normalized_score,'s');
        grid on;
        
        
        
    end
    
end