%digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos',...'nndatasets','DigitDataset');
%ImgDs = imageDatastore('data',...
  %     'IncludeSubfolders',true,'LabelSource','foldernames');
%%[Train,Test] = splitEachLabel(ImgDs,0.70);
Train = imageDatastore('train',...
         'IncludeSubfolders',true,'LabelSource','foldernames');
Test = imageDatastore('test',...
         'IncludeSubfolders',true,'LabelSource','foldernames');
Train.countEachLabel
Test.countEachLabel

layers = [imageInputLayer([256 256 1]); 
    convolution2dLayer(3,8,'Padding',1)
    dropoutLayer(0.10)
    reluLayer
    crossChannelNormalizationLayer(5)
    maxPooling2dLayer(2,'Stride',2) 
    convolution2dLayer(3,16,'Padding',1)
    dropoutLayer(0.10)
    reluLayer
    crossChannelNormalizationLayer(5)
    maxPooling2dLayer(2,'Stride',2)
    %fullyConnectedLayer(9)
    %reluLayer
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];
      
options = trainingOptions('sgdm','MaxEpochs',50,...
	'InitialLearnRate',0.0005);

convnet = trainNetwork(Train,layers,options);

[YTest, WEIGHTS] = classify(convnet,Test);
Lab_tst = YTest;
TTest = Test.Labels;
abbr_categories={'BED','COR','KIT'};
categories={'bedroom','corridor','kitchen'};
accuracy = sum(YTest == TTest)/numel(TTest)   
cm=confusionmat(TTest,YTest)
fig_handle = figure; 
imagesc(cm); 
     
