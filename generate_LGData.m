% This script generates binary data of from a Layered-Graph model


% Generate a full network structure
%[V_psi, V_eta, V_weights] = generateNetworkStructure(15,0.8,0.9,[4 5 15]);
[V_psi, V_eta, V_weights] = generateNetworkStructure(15,0.6,0.9,[5 5 15]);

% Create a matrix to dilute the number of edges
dilution = rand(size(V_psi)) < 0.4; dilution(1,:) = ones(1,size(V_psi,2));
%dilution = rand(size(V_psi)) < 0.95; dilution(1,:) = ones(1,size(V_psi,2));


% Make sure that there are no nodes without a parent
number_of_parentless_children = sum(sum(dilution .* (V_psi + V_eta)) == 0);
dilution(:,sum(dilution .* (V_psi + V_eta)) == 0) = ones(size(dilution,1), number_of_parentless_children);


view(biograph(V_psi.*dilution,[],'showweights','on'))
view(biograph(V_psi,[],'showweights','on'))

%% Generate 5 LG 1-4-5-15 datasets, based on the structure
for j = 1:5
    [f,y] = generateNetworkDependentData(15,10000,0.5,[5 5 15], ...
            V_psi .* dilution, ...
            V_eta .* dilution, ...
            (V_weights .* dilution) ./ repmat(sum(V_weights .* dilution), size(V_weights,1),1));
        % V_weights needs to be re-weighted so that the columns sum to 1

    str = strcat('datasets/simulated/LG/dataset', num2str(j),'.mat');
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
    str = strcat('datasets/simulated/LG/dataset', num2str(i),'.mat');
    writeDatasetInCubamFormat(str)
end

%% compute Bayes optimal error

disp 'computing Bayes error'
nObs = 10^6;
% sampling 10^6 vectors from the above tree
[f,y] = generateNetworkDependentData(15,nObs,0.5,[5 5 15], ...
            V_psi .* dilution, ...
            V_eta .* dilution, ...
            (V_weights .* dilution) ./ repmat(sum(V_weights .* dilution), size(V_weights,1),1));
        % V_weights needs to be re-weighted so that the columns sum to 1

f = f';
disp 'estimating posterior probabilities'
% compute approximate posterior $p(y|x) for the first 1000 vectors in
% bigData
num = 200;
posterior = zeros(num,1);
for i=1:num
   disp(i)
   v = f(i,:);
   inds = find(ismember(f,v,'rows'));
   posterior(i) = mean(y(inds));   
end
BayesAcc = mean(round(posterior)==y(1:num));
fprintf('Bayes accuracy: %2.2f \n' ,BayesAcc);

str = strcat('datasets/simulated/LG/model.mat');
    save(str, 'BayesAcc')
    
% Bayes error: 0.95     