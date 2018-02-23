% Adapted from MATLAB tutorial on Transfer Learning at:
%https://www.mathworks.com/help/nnet/examples/transfer-learning-using-alexnet.html
Ds = imageDatastore('dataset1',...
       'IncludeSubfolders',true,'LabelSource','foldernames');
[Train,Test] = splitEachLabel(Ds,0.7);
orinet = alexnet;
alexlayers = orinet.Layers(1:end-3);
classes = numel(categories(Train.Labels));
layers = [
    alexlayers
    fullyConnectedLayer(classes,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

minbSize = 10;
iternum = floor(numel(Train.Labels)/minbSize);
options = trainingOptions('sgdm',...
    'MiniBatchSize',minbSize,...
    'MaxEpochs',4,...
    'InitialLearnRate',1e-4,...
    'Verbose',false);

transfernet = trainNetwork(Train,layers,options);
YTest = classify(transfernet,Test);
TTest = Test.Labels;
Conf = confusionmat(TTest,YTest)
imagesc(Conf)
colorbar
accuracy = mean(YTest == TTest)