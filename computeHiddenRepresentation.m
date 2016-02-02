function hidRep = computeHiddenRepresentation (vishid,hidbiases, data)
hidbias = repmat(hidbiases,size(data,1),1); 
poshidprobs = 1./(1 + exp(-data*(vishid)...
    - hidbias));  
hidRep = round(poshidprobs);