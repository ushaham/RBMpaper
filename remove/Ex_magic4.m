
%% read data
close all hidden; clc; clear;
%datasetName = 'datasets/condInd.mat';
datasetName = 'datasets/magic/magic_block_4.mat';
iDivideData = 0;
data = readData2a(datasetName, iDivideData);

%% setup
rbmInput.restart=1;

% the following are configurable hyperparameters for RBM

rbmInput.reg_type = 'l2';
rbmInput.weightPenalty = 1e-2;%1e-2 for 1,2,4, 1e-1 for 3,5
rbmInput.epsilonw      = 1e-1;%1e-1;   % Learning rate for weights 
rbmInput.epsilonvb     = 1e-1;%1e-1;   % Learning rate for biases of visible units 
rbmInput.epsilonhb     = 1e-1;%1e-1;   % Learning rate for biases of hidden units 
rbmInput.CD=5;   
rbmInput.initialmomentum  = 0.5;%0.5;%0.5;
rbmInput.finalmomentum    = 0.9;%0.9;%0.9;
rbmInput.maxEpoch = 50;
rbmInput.decayLrAfter = 40; % either 0 or 1
rbmInput.decayMomentumAfter = 40; % when to switch from initial to final momentum
rbmInput.iIncreaseCD = 0;
% monitor free energy and likelihood change (on validation set) with time
rbmInput.iMonitor = 1;


%% train
sizes = [];
rbmInput.data = data;
rbmInput.numhid = size(data.allDataTable,2); % non-configurable for RBM1
stack = cell(1,1);
layerCounter = 1;
addLayers = 1;
while addLayers
    % train RBM
    rbmOutput = rbmV2(rbmInput);
    % collect params
    stack{layerCounter}.vishid = rbmOutput.vishid;
    stack{layerCounter}.hidbiases = rbmOutput.hidbiases;
    stack{layerCounter}.visbiases = rbmOutput.visbiases;

    % SVD to determine number of hidden nodes
    [U,D,V]  = svd (stack{layerCounter}.vishid);
    cumsum(diag(D)) /sum( diag(D))
    numhid = min(find(cumsum(diag(D))/sum(diag(D))>0.95));
    fprintf ('need %1.0f hidden units\n', numhid);
    disp 'paused, press any key to continue'
%    pause;

    % Re-train RBM
    sizes = [sizes, numhid];
    rbmInput.numhid = numhid;
    rbmOutput = rbmV2(rbmInput);
    % collect params
    stack{layerCounter}.vishid = rbmOutput.vishid;
    stack{layerCounter}.hidbiases = rbmOutput.hidbiases;
    stack{layerCounter}.visbiases = rbmOutput.visbiases;
    figure
    imagesc(stack{layerCounter}.vishid)
    colorbar;
    title (strcat('Weight matrix of RBM ', num2str(layerCounter)));
    xlabel('visible nodes')
    ylabel('hidden nodes')

    % setup for next RBM
    rbmInput.data = obtainHiddenRep(rbmInput, rbmOutput);

    % stopping criterion
    if numhid ==1
        addLayers = 0;
    end
    layerCounter = layerCounter + 1;
end

numLayers = size(stack,2);
fprintf ('trained a deep net with %1.0f layers, of sizes:\n', numLayers);
disp(sizes)
%% obtain posterior probabilities
% deterministic
mode = 'deterministic';
posteriorProbsDet = forward (stack, data.allDataTable, mode);


% stochastic
mode = 'stochastic';
nit = 100;
posteriorProbsStoch = forward (stack, data.allDataTable, mode, nit);

%% predict labels
labels = data.labels;

% deterministic mode:
predictedLabels = round(posteriorProbsDet);
% check if predictedLables need to be flipped
m = mean(predictedLabels == data.allDataTable(:,1));
if (m<0.5)
    predictedLabels = 1-predictedLabels;
end
acc = mean(labels==predictedLabels);
inds1 = labels==1;
inds0 = labels==0;
sensitivity = mean(predictedLabels(inds1));
specificity = 1-mean(predictedLabels(inds0));

balAcc_rbmDet = (sensitivity + specificity)/2;
disp 'Deterministic mode:'
fprintf (1,'sensitivity: %0.3f%%\n',100*sensitivity);
fprintf (1,'specificity: %0.3f%%\n',100*specificity);
fprintf (1,'accuracy: %0.3f%%\n',100*acc);
fprintf (1,'balanced accuracy: %0.3f%%\n',100*balAcc_rbmDet);

%stochastic mode:
predictedLabels = round(posteriorProbsStoch);
% check if predictedLables need to be flipped
m = mean(predictedLabels == data.allDataTable(:,1));
if (m<0.5)
    predictedLabels = 1-predictedLabels;
end
acc = mean(labels==predictedLabels);
inds1 = labels==1;
inds0 = labels==0;
sensitivity = mean(predictedLabels(inds1));
specificity = 1-mean(predictedLabels(inds0));

balAcc_rbmStoch = (sensitivity + specificity)/2;
disp 'Stochastic mode:'
fprintf (1,'sensitivity: %0.3f%%\n',100*sensitivity);
fprintf (1,'specificity: %0.3f%%\n',100*specificity);
fprintf (1,'accuracy: %0.3f%%\n',100*acc);
fprintf (1,'balanced accuracy: %0.3f%%\n',100*balAcc_rbmStoch);

%% compare to other models
load (datasetName);
[y_vote, y_sml, y_rl, y_corr] = comparison(f);
inds1 = labels==1;
inds0 = labels==0;
% vote
sensitivity = mean(y_vote(inds1));
specificity = 1-mean(y_vote(inds0));
balAcc_vote = (sensitivity + specificity)/2;
% sml
sensitivity = mean(y_sml(inds1));
specificity = 1-mean(y_sml(inds0));
balAcc_sml = (sensitivity + specificity)/2;
% rl
sensitivity = mean(y_rl(inds1));
specificity = 1-mean(y_rl(inds0));
balAcc_rl = (sensitivity + specificity)/2;
% corr
sensitivity = mean(y_corr(inds1));
specificity = 1-mean(y_corr(inds0));
balAcc_corr = (sensitivity + specificity)/2;

%%
% calculate best
b = repmat(y, size(f,1), 1);
best = max(mean(f==b,2))*100;
%% conclusion
disp (strcat('RBM result (det): ', num2str(balAcc_rbmDet)))
disp (strcat('RBM result (stoch): ', num2str(balAcc_rbmStoch)))
disp (strcat('vote result: ', num2str(balAcc_vote)))
disp (strcat('SML result: ', num2str(balAcc_sml)))
disp (strcat('RL result: ', num2str(balAcc_rl)))
disp (strcat('CORR result: ', num2str(balAcc_corr)))
disp (strcat('Best result: ', num2str(best)))
% RBM result (det):0.8529
% RBM result (stoch):0.8529
% vote result:0.8516
% SML result:0.8475
% RL result:0.8471
% CORR result:0.8529
