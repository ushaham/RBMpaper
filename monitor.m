function output = monitor(monitorInput)

vishid = monitorInput.vishid;
visbiases = monitorInput.visbiases;
hidbiases = monitorInput.hidbiases;


trainData = monitorInput.trainData;
validationData = monitorInput.validationData;

numTrainPts = size(trainData,1);
numValPts = size(validationData,1);

trainData = trainData > rand(size(trainData)); 
validationData = validationData > rand(size(validationData)); 

trainVisbias = repmat(visbiases,numTrainPts,1);
validationVisbias = repmat(visbiases,numValPts,1);

trainHidbias = repmat(1*hidbiases,numTrainPts,1); 
validationHidbias = repmat(1*hidbiases,numValPts,1); 

%%%%%%%%% START OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainPoshidprobs = 1./(1 + exp(-trainData*(1*vishid) - trainHidbias));  
validationPoshidprobs = 1./(1 + exp(-validationData*(1*vishid) - validationHidbias));  
%%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainPoshidstates = trainPoshidprobs > rand(size(trainPoshidprobs));
validationPoshidstates = validationPoshidprobs > rand(size(validationPoshidprobs));

trainNegdata = 1./(1 + exp(-trainPoshidstates*vishid' - trainVisbias));
validationNegdata = 1./(1 + exp(-validationPoshidstates*vishid' - validationVisbias));

trainNegdata = trainNegdata > rand(size(trainNegdata)); 
validationNegdata = validationNegdata > rand(size(validationNegdata)); 

trainNeghidprobs = 1./(1 + exp(-trainNegdata*(1*vishid) - trainHidbias));

%% reconstruction error
output.recErrTraining = sum(sum((trainData-trainNegdata).^2))/numTrainPts;
output.recErrValidation = sum(sum((validationData-validationNegdata).^2))/ numValPts;
 
%% free energy
trainX = trainData*(1*vishid) + trainHidbias;
validationX = validationData*(1*vishid) + validationHidbias;

output.freeEnergyTraining = -sum(trainData*visbiases' + sum(log(1+exp(trainX)),2))/numTrainPts;
output.freeEnergyValidation = -sum(validationData*visbiases' + sum(log(1+exp(validationX)),2)) / numValPts;