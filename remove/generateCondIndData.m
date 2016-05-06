% This script generates binary data of conditionally independent
% classifiers using a Naive Bayes type of graphical model

% number of  classifiers
nClassifiers = 15;
% number of observations
nObs = 10000;

% sample sensitivity and sensitivity parameters
classifierParams = 0.5+0.5*rand(2,nClassifiers);
classifierParams(:,6:end) = 0.5;

% generate true labels
class1Prior = 0.5;
labels = rand(nObs,1)<class1Prior;

%% generate 5 datasets

for j = 1:5
    orgData = rand(nObs, nClassifiers);
    for i = 1:nObs
        label = labels(i);
        if label==0
           orgData(i,:) = orgData(i,:)>classifierParams(1,:);
        else % label ==1 
            orgData(i,:) = orgData(i,:)<classifierParams(2,:);
        end
    end

    f = orgData';
    y = labels;
    str = strcat('datasets/simulated/condInd/dataset', num2str(j),'.mat');
    save(str, 'f', 'y', 'classifierParams')
end

%% plot correlation matrix
c1 = corr(orgData(y==0,:));
figure
imagesc(c1)
h = colorbar;
caxis([-.2,1])
set(gca, 'fontsize', 15)
set(gca,'xtick',0:((length(c1)>5)+1):length(c1));
set(gca,'ytick',0:((length(c1)>5)+1):length(c1));
%% convert to CUBAM format
for i = 1:5
    str = strcat('datasets/simulated/condInd/dataset', num2str(i),'.mat');
    writeDatasetInCubamFormat(str)
end

%% compute Bayes optimal error

disp 'computing Bayes error'
nObs = 10^6;
% sampling 10^6 vectors from the above tree
bigData = rand(nObs, nClassifiers);
labels = rand(nObs,1)<class1Prior;
for i = 1:nObs
    label = labels(i);
    if label==0
       bigData(i,:) = bigData(i,:)>classifierParams(1,:);
    else % label ==1 
        bigData(i,:) = bigData(i,:)<classifierParams(2,:);
    end
end
disp 'estimating posterior probabilities'
% compute approximate posterior $p(y|x) for the first 1000 vectors in
% bigData
num = 200;
posterior = zeros(num,1);
for i=1:num
   disp(i)
   v = bigData(i,:);
   inds = find(ismember(bigData,v,'rows'));
   posterior(i) = mean(labels(inds));   
end
BayesAcc = mean(round(posterior)==labels(1:num));
fprintf('Bayes accuracy: %2.2f \n' ,BayesAcc);

str = strcat('datasets/simulated/condInd/model.mat');
    save(str, 'classifierParams', 'BayesAcc')
    
% Bayes error: 0.94    