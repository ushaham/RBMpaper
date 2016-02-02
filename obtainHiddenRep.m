function [data] = obtainHiddenRep(rbmInput, rbmOutput)

data = rbmInput.data;
vishid = rbmOutput.vishid;
hidbiases = rbmOutput.hidbiases;

data.batchdata = forwardPass(data.batchdata, vishid, hidbiases);
data.trainData = forwardPass(data.trainData, vishid, hidbiases);
data.validationData = forwardPass(data.validationData, vishid, hidbiases);
data.trainDataTable = forwardPass(data.trainDataTable, vishid, hidbiases);
data.validationDataTable = forwardPass(data.validationDataTable, vishid, hidbiases);
data.allDataTable = forwardPass(data.allDataTable, vishid, hidbiases);



 
