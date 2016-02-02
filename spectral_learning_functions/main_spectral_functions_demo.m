%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% main_spectral_functions_demo: This demo demonstrates the use of
%      spectral methods for unsupervised learning.
%      written by Ariel Jaffe and Boaz Nadler, 2015
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%parameters
n_vec = round(10.^(3:0.25:4.5));    %number of instances
m = 10;                             %number of classifiers
acc_limits = [0.5 0.8];             %limits of accuracy of different classifiers
b = 0;                              %class imbalance of class y
delta = 0.01;                       %limit of class imbalance estimation [-1+delta,1-delta]
num_itr = 1000;                      %number of iterations

%initialize b_mse_tensor and b_mse_rl- mean square error of
%class imbalance estimation for both methods (tensor and restricted
%likelihood)
b_mse_tensor = zeros(num_itr,length(n_vec));
b_mse_rl = zeros(num_itr,length(n_vec));

% generate sensitivity and specificity vectors for m classifiers
psi = acc_limits(1)+ diff(acc_limits)*rand(m,1);
eta = acc_limits(1)+ diff(acc_limits)*rand(m,1);

for i = 1:num_itr
    disp(num2str(i));
    for j = length(n_vec)
        
        % generate true label vector y (according to b) and prediction matrix Z
        % according to y, psi and eta
        [y,Z] = generate_prediction_matrix(m,n_vec(j),b,psi,eta);
        
        % estimate class imbalance with the restricted likelihood method
        b_hat_rl = estimate_class_imbalance_restricted_likelihood(Z,delta);
        
        % estimate class imbalance with the tensor method
        b_hat_t = estimate_class_imbalance_tensor(Z,delta);
        
        % estimate sensitivity and specificity of ensemble
        [V_hat,psi_hat,eta_hat] = estimate_ensemble_parameters(Z,b_hat_rl);
        
        %mse
        b_mse_rl(i,j) = (b_hat_rl-b)^2;
        b_mse_tensor(i,j) = (b_hat_t-b)^2;
    end
end

fig_handle = create_figure_2(n_vec,b_mse_tensor,b_mse_rl);
print(fig_handle,'compare_mse.png','-dpng') 