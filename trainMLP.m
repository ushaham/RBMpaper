function testPred = trainMLP (trainDataName, testDataName)
addpath('/Users/urishaham/Documents/Uri''s studies/research/Ex 24');
addpath('/Users/urishaham/Documents/Uri''s studies/research/Ex 24/minFunc');
%clear all;clc
%% setup
inputSize = 15;
numClasses = 2;
hiddenSizeL1 = 4;    % Layer 1 Hidden Size
hiddenSizeL2 = 2;    % Layer 2 Hidden Size
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda1 = 3e-3; % weight decay parameter for the AE 
lambda2 = 1e-4; % weight decay parameter for the softmax
lambda3 = 1e-4; % weight decay parameter for the BP   
beta = 0;              % weight of sparsity penalty term 
hiddenActivation = 'sigmoid';
outputActivation = 'sigmoid';
inputDropoutProbAE = 0; % for the AE
hiddenDropoutProbAE = 0; % for the AE
inputDropoutProbBP = 0; % for the BP
hiddenDropoutProbBP = 0; % for the BP
iTiedWeights = 0;
iStochAE = 0;
iStochBP = 0;
batchSize = 20000;

maxIter = 400; %% maybe omit this?
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = maxIter;% Maximum number of iterations of L-BFGS to run 
options.display = 'on';

%% Load the MNIST database

datasetName = trainDataName;
iDivideData = 0;
data = readData2a(datasetName, iDivideData);

trainData = data.allDataTable';
trainLabels = double(data.labels');
trainLabels(trainLabels == 0) = 2; % Remap 0 to 2 since our labels need to start from 1


% check AE gradients
checkStackedAECost()
%%======================================================================
%% Train the first sparse autoencoder

%  Randomly initialize the parameters
sae1Theta = initializeParameters(hiddenSizeL1, inputSize, hiddenActivation);
[sae1OptTheta, cost] = minFunc( @(p) superAutoencoderCost(p, inputSize, hiddenSizeL1, ...
                                             lambda1, sparsityParam, beta, ...
                                             hiddenActivation, outputActivation,...
                                             inputDropoutProbAE, hiddenDropoutProbAE,...
                                             iTiedWeights, iStochAE, batchSize, trainData), ...
                                   sae1Theta, options);
                               
[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, hiddenActivation, trainData);

%% Train the second sparse autoencoder

%  Randomly initialize the parameters
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1, hiddenActivation);

[sae2OptTheta, cost] = minFunc( @(p) superAutoencoderCost(p, hiddenSizeL1, hiddenSizeL2, ...
                                             lambda1, sparsityParam, beta, ...
                                             hiddenActivation, outputActivation,...
                                             inputDropoutProbAE, hiddenDropoutProbAE,...
                                             iTiedWeights, iStochAE, batchSize, sae1Features), ...
                                   sae2Theta, options);
                               
[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, hiddenActivation, sae1Features);
                                    
                                    
%% Train the softmax classifier                                    
                                    
% lambda2=0;                                    
%  Randomly initialize the parameters
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);

softmaxModel = softmaxTrain(hiddenSizeL2, numClasses, lambda2, ...
                            sae2Features, trainLabels, options);
saeSoftmaxOptTheta = softmaxModel.optTheta(:);

%% Finetune softmax model
% Initialize the stack using the parameters learned
stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
% from here
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

hidden1Activation = hiddenActivation;
hidden2Activation = hiddenActivation;

[stackedAEOptTheta, cost] = minFunc( @(p) stackedAECost(p, inputSize, hiddenSizeL2, ...
                                              numClasses, netconfig, ...
                                              lambda3, hidden1Activation, hidden2Activation,...
                                              inputDropoutProbBP, hiddenDropoutProbBP, iStochBP, batchSize,...
                                              trainData, trainLabels), ...
                              stackedAETheta, options);
                          
 %% 
%% STEP 6: Test 


% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set
[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, hidden1Activation, hidden2Activation, trainData);

acc = mean(trainLabels(:) == pred(:));
fprintf('Before Finetuning Train Accuracy: %0.3f%%\n', acc * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, hidden1Activation, hidden2Activation, trainData);

acc = mean(trainLabels(:) == pred(:));
fprintf('After Finetuning (Ng) Train Accuracy: %0.3f%%\n', acc * 100);


datasetName = testDataName;
iDivideData = 0;

data = readData2a(datasetName, iDivideData);
testData = data.allDataTable';
testLabels =  double(data.labels');
testLabels(testLabels == 0) = 2; % Remap 0 to 2 since our labels need to start from 1

[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, hidden1Activation, hidden2Activation, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, hidden1Activation, hidden2Activation, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning (Ng) Test Accuracy: %0.3f%%\n', acc * 100);

testPred = pred;
