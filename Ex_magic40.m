
%% read data
close all hidden; clc; clear;
iDivideData = 0;

%% setup
rbmInput.restart=1;

% the following are configurable hyperparameters for RBM

rbmInput.reg_type = 'l2';
rbmInput.weightPenalty = 2e-2;%1e-2 for 1,2,4, 1e-1 for 3,5
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
rbmInput.iMonitor = 0;

results = zeros(6,40);
for i = [10    13      21    31  ]% 1:40
    % readData
    iDivideData = 0;
    str = strcat('datasets/real/magic/magic_block_', num2str(i),'.mat'); 
    data = readData2a(str, iDivideData);
    
    % train
    sizes = [];
    rbmInput.data = data;
    rbmInput.numhid = size(data.allDataTable,2); % non-configurable for RBM1
    stack = cell(1,1);
    layerCounter = 1;
    addLayers = 1;
    while addLayers
        % train RBM
        A = rbmInput.data.allDataTable;
        y = rbmInput.data.labels;
        c1 = corr(A(y==0,:));
        imagesc(c1)
        h = colorbar;
        caxis([-.2,1])
        %title ('conditional correlation matrix, class = 0')
        set(gca, 'fontsize', 15)
        set(gca,'xtick',0:((length(c1)>5)+1):length(c1));
        set(gca,'ytick',0:((length(c1)>5)+1):length(c1));
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
        % pause;

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
    load (str);
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

    results(1,i) = balAcc_rbmDet;
    results(2,i) = balAcc_rbmStoch;
    results(3,i) = balAcc_vote;
    results(4,i) = balAcc_sml;
    results(5,i) = balAcc_rl;
    results(6,i) = balAcc_corr;
end

%%
%results = 100*results;

RBMdet = results(1,:);
RBMstoch = results(2,:);
Vote = results(3,:);
SML = results(4,:);
RL = results(5,:);
L_SML = results(6,:);


CUBAM = 100*[ 0.70058352,  0.65157072,  0.71193707,  0.70138196,  0.69392606,...
        0.64933047,  0.6998905 ,  0.70328757,  0.70708138,  0.71149561,...
        0.69479273,  0.68676444,  0.68991805,  0.69170084,  0.70248576,...
        0.70816106,  0.69003731,  0.66343751,  0.70882995,  0.71071677,...
        0.70329014,  0.66027976,  0.69272474,  0.68396626,  0.66065356,...
        0.70534032,  0.65572311,  0.70054298,  0.64192026,  0.64798221,...
        0.69062511,  0.7004162 ,  0.70533181,  0.6898154 ,  0.63427686,...
        0.65281634,  0.66359079,  0.70512494,  0.65067359,  0.65042024];

%RBMdet = [0.873532356	0.878371178	0.872339779	0.836072636	0.882321258	0.789448016	0.871955295	0.787257575	0.885980064	0.788242748	0.88183383	0.878398628	0.877369682	0.881516509	0.887369486	0.792298493	0.866849917	0.853832423	0.854652736	0.828821225	0.772634951	0.879283742	0.886884709	0.889875487	0.853673112	0.772878424	0.853810637	0.878389638	0.847137918	0.848579571];
%RBMstoch = [0.869955466	0.873509539	0.870360853	0.83545133	0.876920416	0.796460303	0.871284301	0.799992368	0.879520041	0.797685112	0.878849159	0.873721232	0.871468318	0.877209903	0.880871528	0.802309397	0.864129858	0.853171988	0.855232134	0.842222921	0.818014593	0.872181298	0.878914938	0.883270615	0.854328001	0.793289158	0.857676626	0.875549999	0.846731189	0.84788347];
%Vote = [0.804874295	0.796083328	0.800698768	0.802539576	0.800702028	0.805528514	0.799898271	0.802996465	0.799461156	0.806404119	0.799495067	0.803190702	0.801097553	0.796204267	0.800188602	0.800644605	0.802490461	0.798933497	0.805115164	0.799149246	0.806101132	0.803086938	0.804369733	0.795099228	0.793333473	0.79779032	0.802374916	0.804747068	0.807499885	0.802593809];
%SML = [0.814175948	0.803890047	0.812291355	0.811839404	0.811701912	0.814230791	0.80951465	0.813595086	0.809465567	0.81802126	0.808692601	0.814602708	0.810696439	0.803802357	0.810385693	0.808419933	0.811484243	0.808732565	0.814506352	0.808408354	0.815201153	0.812394863	0.813472707	0.805298838	0.802814113	0.808091291	0.812971236	0.815964592	0.81859582	0.810697983];
%RL = [0.822681137	0.814268712	0.819275196	0.820614253	0.819199575	0.823352285	0.819164739	0.821689613	0.817990557	0.828140979	0.818205818	0.822271157	0.820995885	0.81240645	0.819465075	0.817797849	0.818476887	0.816920029	0.822092034	0.817586684	0.824500933	0.820011471	0.822301092	0.812697113	0.811986164	0.815780168	0.821925574	0.825193669	0.825404617	0.81982011];
%L_SML = [0.809553974	0.799166174	0.795067918	0.810584271	0.826000897	0.79273055	0.805859193	0.805202238	0.805566151	0.797576404	0.790896899	0.808530714	0.809200838	0.796803547	0.812779115	0.801414206	0.817594681	0.803470452	0.823378051	0.812056513	0.822800833	0.808781471	0.813027567	0.798995947	0.827770326	0.799499382	0.804966643	0.819305344	0.8099009	0.820629095];
%SUP = [0.906814052	0.904526169	0.907730674	0.910569926	0.909505907	0.912047597	0.90661041	0.910067562	0.91049185	0.910027012	0.912384383	0.904367986	0.904783299	0.905413064	0.913918125	0.906085942	0.904717202	0.901715705	0.911922815	0.910607059	0.910998836	0.911092953	0.906821789	0.90899296	0.907003424	0.908391902	0.903551874	0.908780476	0.910505138	0.910525754];

close all
figure
plot(RBMdet, Vote,'o', RBMdet, RL, 'o', RBMdet, L_SML, 'o', RBMdet, CUBAM, 'o');
xlim([74,84]);
ylim([63,84]);
set(gca,'ytick',63:5:84);
refline(1,0)
xlabel('RBM accuracy');
ylabel('other methods accuracy')
set(gca, 'fontsize', 15)
legend('Vote', 'DS', 'L-SML', 'CUBAM')

results
save('magic40Results', 'results')