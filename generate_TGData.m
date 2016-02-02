% This script generates binary data of from a truncated Gaussian model

b = 0; acc_lim = [0.5,0.8]; d=15; n=10000;

% accuracy of each classifier
acc = acc_lim(1)+rand(d,1)*diff(acc_lim);

% mean vector
for j = 1:d
    if (acc(j)<0.5)
        MU(j) = -1* sqrt( log (1/ (2*(acc(j)))));        
    else
        MU(j) = sqrt( log (1/ (2*(1-acc(j)))));        
    end   
end 

% covariance matrix
SIGMA=0.5*ones(d) + 0.5*eye(d);

for j = 1:5

    %create true labels y
    y = randsrc(1,n,[1 -1; (1+b)/2 (1-b)/2]);
    Z = mvnrnd(y'*MU,SIGMA);
    Z = (sign(Z)+1)/2;
    y(y==-1)=0;
    f = Z';
    y = y';
    str = strcat('datasets/simulated/TG/dataset', num2str(j),'.mat');
    save(str, 'f', 'y')
end

%% plot correlation matrix
c1 = corr(f(:,y==0)');
figure
imagesc(c1)
h = colorbar;
caxis([-.2,1])
set(gca, 'fontsize', 15)
set(gca,'xtick',0:((length(c1)>5)+1):length(c1));
set(gca,'ytick',0:((length(c1)>5)+1):length(c1));

%% convert to CUBAM format
for i = 1:5
    str = strcat('datasets/simulated/TG/dataset', num2str(i),'.mat');
    writeDatasetInCubamFormat(str)
end

%% compute Bayes optimal error

disp 'computing Bayes error'
n = 10^6;
% sampling 10^6 vectors from the above tree
y = randsrc(1,n,[1 -1; (1+b)/2 (1-b)/2]);
Z = mvnrnd(y'*MU,SIGMA);
Z = (sign(Z)+1)/2;
y(y==-1)=0;
f = Z';
y = y';
    
disp 'estimating posterior probabilities'
% compute approximate posterior $p(y|x) for the first 1000 vectors in
% bigData
num = 200;
posterior = zeros(num,1);
for i=1:num
   disp(i)
   v = Z(i,:);
   inds = find(ismember(Z,v,'rows'));
   posterior(i) = mean(y(inds));   
end
BayesAcc = mean(round(posterior)==y(1:num));
fprintf('Bayes accuracy: %2.2f \n' ,BayesAcc);

str = strcat('datasets/simulated/TG/model.mat');
    save(str, 'BayesAcc', 'MU', 'SIGMA')
    
% Bayes error: 0.85     