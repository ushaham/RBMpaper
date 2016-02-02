function data = readData2b(dataSetName, iDivideData)
if nargin<2 
    iDivideData = 0;
end

load(dataSetName);
%f: d x n matrix of -1,1
%y: labels -1 or 1

[numClassifiers, numObs] = size(f);

p = randperm(numObs);
%f = f(:,p);
%y = y(p);


batchSize = 80;
data.batchSize = batchSize;
numObs = floor(numObs/batchSize)*batchSize;

orgData=f';
orgData(orgData==-1)=0;
orgData = orgData(1:numObs, 1:numClassifiers);
labels = (y(1:numObs))';
labels(labels==-1)=0;
data.labels = labels;



% arrange in minibatches for RBM training
numBatches = numObs/batchSize;
numValBatches = floor(numBatches/10);

numTrainBatches = numBatches - numValBatches;
allData =  reshape(orgData', numClassifiers, batchSize, numBatches);
allData = permute(allData,[2 1 3]);
trainDataTable = (orgData(1:numTrainBatches*batchSize,:));
trainData = reshape(trainDataTable', numClassifiers, batchSize, numTrainBatches);
validationDataTable = (orgData(numTrainBatches*batchSize+1:...
     (numTrainBatches + numValBatches)*batchSize,:));
validationData = reshape(validationDataTable', numClassifiers, batchSize, numValBatches);

data.trainData = permute(trainData,[2 1 3]);
data.validationData = permute(validationData,[2 1 3]);
data.numTrainBatches = numTrainBatches;
data.numValBatches = numValBatches;
data.trainDataTable = trainDataTable;
data.validationDataTable = validationDataTable;
data.trainingLabels = labels(1:numTrainBatches*batchSize);
data.validationLabels = labels(numTrainBatches*batchSize+1:...
     (numTrainBatches + numValBatches)*batchSize);
data.allDataTable = orgData;
data.labels = labels;
if ~iDivideData
    data.batchdata = allData;
else data.batchdata = data.trainData;
end