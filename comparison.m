function [y_vote, y_sml, y_rl, y_corr] = comparison(f)
addpath('./spectral_learning_functions');
addpath('./Spectral_correlated_functions');

m = size(f,1);
n = size(f,2);

f = double(f);
f(f==0)=-1;

%% vote
disp 'calculating vote'
y_vote =  sign(sum(f,1));

%% restricted likelihood

disp 'calculating RL'
delta = 0.01;
b_hat = estimate_class_imbalance_restricted_likelihood(f,delta);
[V,psi_hat,eta_hat] = estimate_ensemble_parameters(f,b_hat,delta);
alpha = log( (psi_hat.*eta_hat)./((1-psi_hat).*(1-eta_hat)) );
beta =  log( (psi_hat.*(1-psi_hat))./(eta_hat.*(1-eta_hat)) );
y_rl = sign(alpha'*f+sum(beta));

%% sml

disp 'SML'
R = cov(f');
R = estimate_rank_1_matrix(R);
[V,D] = eigs(R,1);
V = sign(sum(sign(V)))*V;
y_sml = sign(V'*f);


%% corr

disp 'calculating CORR'
R = cov(f');
S = generate_score_function(R,1:m);

% perform model selection using spectral properties (works only if
% there are only group sizes>1)
[V,D] = eigs(R,m-1);
%K = find(abs(diff(diag(D)))>0.5,1,'last');
min_K = 3;
residual = inf(1,m-min_K+1);
ll_hat = inf(1,m-min_K+1);
for i = min_K:m

    clusters = spectral_cluster(S,i,1,0);
    %[b_hat, alpha_hat,psi_alpha_hat,eta_alpha_hat,psi_alpha_i,eta_alpha_i,ll_hat(i)] = ...
    %estimate_params_correlated_model_v_4(f,clusters);
    residual(i) = get_residual(R,clusters);
end
[~,min_res] = min(residual);
K = min_res;
% use model selection to do estimation of the ensemble params and final
% estimate
clusters = spectral_cluster(S,K,1,0);
[b_hat, alpha_hat,psi_alpha_hat,eta_alpha_hat,psi_alpha_i,eta_alpha_i,ll_hat] = ...
    estimate_params_correlated_model_v_4(f,clusters);
y_corr = apply_ensemble_learner(f,clusters,psi_alpha_i,eta_alpha_i,psi_alpha_hat,eta_alpha_hat);

%% 
y_vote(y_vote==-1)=0;
y_sml(y_sml==-1)=0;
y_rl(y_rl==-1)=0;
y_corr(y_corr==-1)=0;