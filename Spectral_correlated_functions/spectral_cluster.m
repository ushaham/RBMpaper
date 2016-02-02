function clusters = spectral_cluster(S,k,normalize,verbose)
% S -data matrix ,nXn symmetric and positive
% k - number of clusters
% normalize - one for normalized laplacain (recommended) zeor for regular
% verbose - prodice images if 1
if ~exist('normalize','var')
    normalize = 1;
end

if ~exist('verbose','var')
    verbose = 0;
end

D = diag(sum(S));
L = D-S;
L = 0.5*(L+L');
if normalize
    L = D^(-0.5)*L*D^(-0.5);
end
[eigVec,eigVal] = eig(L);
eigVal = diag(eigVal);
[eigVal,idx] = sort(eigVal);
eigVec = eigVec(:,idx);
clusters = kmeans(eigVec(:,1:k),k,'Replicates',500);
if verbose
   figure;
   plot(eigVal,'*-')
   figure;
   imagesc(S)
   figure;
   [~,idx] = sort(clusters);
   imagesc(S(idx,idx));
end