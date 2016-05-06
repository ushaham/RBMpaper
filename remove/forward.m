function posteriorProbs = forward(stack, data, mode, nit)

numLayers = size(stack,2);
numcases = size(data,1);
orgData = data;

if strcmp(mode, 'deterministic')
    for i = 1:numLayers
        hidbiases = stack{i}.hidbiases;
        hidbias = repmat(hidbiases,numcases,1); 
        % obtain hidden probabilities
        poshidprobs = 1./(1 + exp(-data*stack{i}.vishid - hidbias)); 
        % obtain hidden states
        hidStates = round(poshidprobs);
        data = hidStates;
    end
    posteriorProbs = poshidprobs;
else % mode.equals 'stochastic'
    probs = zeros(size(data,1), nit);
    for i = 1:nit
        data = orgData;
        for j = 1:numLayers
                hidbiases = stack{j}.hidbiases;
                hidbias = repmat(hidbiases,numcases,1); 
                poshidprobs = 1./(1 + exp(-data*stack{j}.vishid - hidbias)); 
                hidStates = poshidprobs > rand(size(poshidprobs));
                data = hidStates;
        end
        probs(:,i) = poshidprobs;
    end
    posteriorProbs = mean(probs,2);
end

