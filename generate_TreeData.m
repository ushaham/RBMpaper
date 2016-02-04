% This script generates binary data of from a tree model


%% Generate Tree 1-3-5 structure
d=15; n=10000; prob_y_is_1 = 0.5;
min_psi=[0.8 0.6]; min_eta=[0.8 0.6];
%min_psi=[0.8 0.5]; min_eta=[0.8 0.5];
num_nodes_per_layer=[3 5];
t = generateTreeStructure(d,n,min_psi,min_eta,num_nodes_per_layer);

%% Print Tree structure
idxs = t.breadthfirstiterator;
tree_text = 'Node  1:\tTrue Label\n';
for i=idxs(2:end)
    tree_text = [tree_text sprintf('Node %2d,\tParent %2d:\tPsi=%.02f\tEta=%.02f\n', ...
            i,t.getparent(i),t.get(i))];
end
fprintf(tree_text);

%% Generate 5 Tree 1-3-5 datasets, based on the structure
for j = 1:5
    [f,y] = generateTreeDependentData(t, d, n, prob_y_is_1);
    str = strcat('datasets/simulated/Tree/dataset', num2str(j),'.mat');
    save(str, 'f', 'y', 't')
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
    str = strcat('datasets/simulated/Tree/dataset', num2str(i),'.mat');
    writeDatasetInCubamFormat(str)
end

%% compute Bayes optimal error

disp 'computing Bayes error'
nObs = 10^6;
% sampling 10^6 vectors from the above tree
[f,y] = generateTreeDependentData(t, d, nObs, prob_y_is_1);
Z = f';

disp 'estimating posterior probabilities'
a = (0:14)';
b = 2.^a;
counts = zeros(2^15,2);
for i = 1:10^6
   x =  Z(i,:);
   label = y(i);
   ind = x*b + 1;
   counts(ind, label+1) = counts(ind, label+1)+1;
end

preds = zeros(10000,1);
for i=1:10000
   disp(i)
   x = Z(i,:);
   ind = x*b +1;
   preds(i) = counts(ind,2)>counts(ind,1);
end
BayesAcc = mean(preds==y(1:10000));
fprintf('Bayes accuracy: %2.2f \n' ,BayesAcc);

str = strcat('datasets/simulated/Tree/model.mat');
    save(str, 'BayesAcc', 't')
    
% Bayes error: 0.96           