function rbmOutput = rbm(rbmInput)

% Code based on publicly available code provided by Ruslan Salakhutdinov 

batchdata = rbmInput.data.batchdata;

decayLrAfter = rbmInput.decayLrAfter; 
epsilonw_0 = rbmInput.epsilonw;     % Learning rate for weights 
epsilonvb_0 = rbmInput.epsilonvb;   % Learning rate for biases of visible units 
epsilonhb_0 = rbmInput.epsilonhb;   % Learning rate for biases of hidden units 
initialmomentum = rbmInput.initialmomentum;
finalmomentum = rbmInput.finalmomentum;
lambda  =  rbmInput.weightPenalty;   
CD=rbmInput.CD; 
maxEpoch = rbmInput.maxEpoch;
decayMomentumAfter = rbmInput.decayMomentumAfter;
numhid = rbmInput.numhid;
iIncreaseCD = rbmInput.iIncreaseCD;

[numcases numdims numbatches]=size(batchdata);
restart = rbmInput.restart;

if restart ==1,
  restart=0;
  epoch=1;

% Initializing symmetric weights and biases. 
  vishid     = 0.01*randn(numdims, numhid);
  hidbiases  = zeros(1,numhid);
  visbiases  = zeros(1,numdims);

  poshidprobs = zeros(numcases,numhid);
  neghidprobs = zeros(numcases,numhid);
  posprods    = zeros(numdims,numhid);
  negprods    = zeros(numdims,numhid);
  vishidinc  = zeros(numdims,numhid);
  hidbiasinc = zeros(1,numhid);
  visbiasinc = zeros(1,numdims);
  batchposhidprobs=zeros(numcases,numhid,numbatches);
end

if rbmInput.iMonitor
    freeEnergyTraining = zeros(maxEpoch,1);
    freeEnergyValidation = zeros(maxEpoch,1);
    recErrTraining = zeros(maxEpoch,1);
    recErrValidation = zeros(maxEpoch,1);
end


for epoch = epoch:maxEpoch
 fprintf(1,'epoch %d\r',epoch);
 
 if iIncreaseCD
     CD = ceil(epoch/10);
 end

 if epoch > decayLrAfter
   factor = 10^-ceil((epoch - decayLrAfter)/10); 
   epsilonw = epsilonw_0*factor;
   epsilonvb = epsilonvb_0*factor;
   epsilonhb = epsilonhb_0*factor;
 else
   epsilonw = epsilonw_0;
   epsilonvb = epsilonvb_0;
   epsilonhb = epsilonhb_0;
 end 

 
 errsum=0;
 
 for batch = 1:numbatches,
   fprintf(1,'epoch %d batch %d\r',epoch,batch); 

   visbias = repmat(visbiases,numcases,1);
   hidbias = repmat(hidbiases,numcases,1); 
   %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   data = batchdata(:,:,batch);
   data = data > rand(numcases,numdims);  

   poshidprobs = 1./(1 + exp(-data*vishid - hidbias));    
   posprods    = data' * poshidprobs;
   poshidact   = sum(poshidprobs);
   posvisact = sum(data);

   %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   poshidprobs_temp = poshidprobs;

   %%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   for cditer=1:CD
     poshidstates = poshidprobs_temp > rand(numcases,numhid);
     negdata = 1./(1 + exp(-poshidstates*vishid' - visbias));
     negdata = negdata > rand(numcases,numdims); 
     poshidprobs_temp = 1./(1 + exp(-negdata*vishid - hidbias));
   end 
   neghidprobs = poshidprobs_temp;     

   negprods  = negdata'*neghidprobs;
   neghidact = sum(neghidprobs);
   negvisact = sum(negdata); 

   %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   err= sum(sum( (data-negdata).^2 ));
   errsum = err + errsum;

   if epoch>decayMomentumAfter,
     momentum=finalmomentum;
   else
     momentum=initialmomentum;
   end;

   %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    if strcmp(rbmInput.reg_type, 'l2')% l_2 regularization
        vishidinc = momentum*vishidinc + ...
                    epsilonw*( (posprods-negprods)/numcases - lambda*vishid);  
        vishid = vishid + vishidinc;
    else % l_1 regularization
        vishidinc = momentum*vishidinc + ...
                    epsilonw*((posprods-negprods)/numcases); 
        vishid = softThresholding(vishid + vishidinc, lambda*epsilonw);
    end
               
    visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
    hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);

    visbiases = visbiases + visbiasinc;
    hidbiases = hidbiases + hidbiasinc;
    %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
 end
 
 fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum);
 if rbmInput.iMonitor
     monitorInput.trainData = rbmInput.data.trainDataTable;
     monitorInput.validationData = rbmInput.data.validationDataTable;
     monitorInput.visbiases = visbiases;
     monitorInput.hidbiases = hidbiases;
     monitorInput.vishid = vishid;
     
     monitorOutput = monitor(monitorInput);
     freeEnergyTraining(epoch) = monitorOutput.freeEnergyTraining;
     freeEnergyValidation(epoch) = monitorOutput.freeEnergyValidation;
     recErrTraining(epoch) = monitorOutput.recErrTraining;
     recErrValidation(epoch) = monitorOutput.recErrValidation;
 end
end
 
 % RBM output
 rbmOutput.vishid = vishid;
 rbmOutput.visbiases = visbiases;
 rbmOutput.hidbiases = hidbiases;
 rbmOutput.batchposhidprobs = batchposhidprobs;
 rbmOutput.poshidstates = poshidstates;
 
 if rbmInput.iMonitor
     rbmOutput.freeEnergyTraining = freeEnergyTraining;
    rbmOutput.freeEnergyValidation = freeEnergyValidation;
    rbmOutput.recErrTraining = recErrTraining;
    rbmOutput.recErrValidation = recErrValidation;
    figure
    plot(1:maxEpoch, freeEnergyTraining, 1:maxEpoch, freeEnergyValidation);
    title ('Free Energies')
    legend('Training', 'Validation')
    xlabel ('Epoch')
    ylabel ('Avg Free Energy on validation data')
    figure
    plot(1:maxEpoch, recErrTraining, 1:maxEpoch, recErrValidation);
    title ('Reconstruction Error')
    legend('Training', 'Validation')
    xlabel ('Epoch')
    ylabel ('Avg Reconstruction Error on validation data')
end
 


