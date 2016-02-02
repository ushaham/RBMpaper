function [f,psi,eta] = generate_correlated_prediction_matrix(y,group_size,group_acc,inter_acc)
    global k
    % generate correlated prediction matrix according to the following
    % input:
    % y          -  vector of true labels
    % group_size -  vector containing the size of correlated groups
    % group_acc  -  The limits of accuracies for the intermediate paramter Z
    %               of each group
    % inter_acc  -  The limits of accuracies for the single classifier of
    %               each group with reference to the intermediate parameter
    %
    % output:
    % f          -  Correlated prediction matrix
    % psi        -  vector containing sensitivity of each classifier
    % eta        -  vector containing specificity of each classifier
    
    %get number of classifiers
    m = sum(group_size);
    n = length(y);
    n_groups = length(group_size);
    
    %get indices of positive and negative elements in y
    pos_idx = find(y==1);
    neg_idx = find(y==-1);
    
    %%  generate intermediate variables z
    Z = zeros(n_groups,n);
    rng(1000);
    psi_y_z = group_acc(1)+diff(group_acc)*rand(n_groups,1);
    eta_y_z = group_acc(1)+diff(group_acc)*rand(n_groups,1);
    rng('shuffle');
    for i = 1:n_groups
        Z(i,pos_idx) = randsrc(1,length(pos_idx),[1 -1 ; psi_y_z(i) 1-psi_y_z(i)]);
        Z(i,neg_idx) = randsrc(1,length(neg_idx),[1 -1 ; 1-eta_y_z(i) eta_y_z(i)]);
    end
    
    %% generate group data
    f = zeros(m,n);
    psi = zeros(m,1);
    eta = zeros(m,1);
    rng(100);
    psi_z_f = inter_acc(1)+diff(inter_acc)*rand(m,1);
    eta_z_f = inter_acc(1)+diff(inter_acc)*rand(m,1);
    rng('shuffle');
    %rand('seed',k)
    ctr = 1;
    for i = 1:n_groups
        z_pos_idx = find(Z(i,:)==1);
        z_neg_idx = find(Z(i,:)==-1);
        for j = 1:group_size(i)
            f(ctr,z_pos_idx) = randsrc(1,length(z_pos_idx),[1 -1 ; psi_z_f(ctr) 1-psi_z_f(ctr)]);
            f(ctr,z_neg_idx) = randsrc(1,length(z_neg_idx),[1 -1 ; 1-eta_z_f(ctr) eta_z_f(ctr)]);
            
            
            %get effective psi and eta
            psi(ctr) = psi_y_z(i)*psi_z_f(ctr)+(1-psi_y_z(i))*(1-eta_z_f(ctr));
            eta(ctr) = eta_y_z(i)*eta_z_f(ctr)+(1-eta_y_z(i))*(1-psi_z_f(ctr));
            
            ctr = ctr+1;
        end
    end
    
    
end



