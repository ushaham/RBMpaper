function out = forwardPass(data, vishid, hidbiases)
numhid = size(vishid,2);
[numcases numdims numbatches]=size(data);
out = zeros(numcases,numhid,numbatches);

hidbias = repmat(hidbiases,numcases,1);

for i = 1:numbatches,
   
   %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   batch = data(:,:,i);
   poshidprobs = 1./(1 + exp(-batch*vishid - hidbias));
   out(:,:,i) = round(poshidprobs);
   out(:,:,i) = poshidprobs>rand(size(poshidprobs));
end
 
