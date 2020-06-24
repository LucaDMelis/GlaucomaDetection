pathtiTrain = "C:\Users\lucaf\Desktop\EAI\Ophtamologia\Tesi\originales";

%% Rete sperimentale
inLayer = imageInputLayer([256 256 3]);
midLayers = [ convolution2dLayer(11, 3); reluLayer(); maxPooling2dLayer(2);...
    convolution2dLayer(5, 96); reluLayer(); maxPooling2dLayer(2); convolution2dLayer(3, 192);...
    maxPooling2dLayer(2); convolution2dLayer(3, 192);maxPooling2dLayer(2)];
outLayers = [fullyConnectedLayer(2); softmaxLayer();fullyConnectedLayer(2); softmaxLayer();classificationLayer()];

ChenLayer = [inLayer; midLayers; outLayers];
inputSize = ChenLayer(1).InputSize

%% Datastore e Augmentation
imds = imageDatastore(pathtiTrain,'IncludeSubfolders',true,'LabelSource','foldernames');
[trainImgs,testImgs] = splitEachLabel(imds,0.15,0.85,'randomized');

pixelRange = [-10 10];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

trainAug= augmentedImageDatastore(inputSize(1:2),trainImgs, 'DataAugmentation',imageAugmenter);
testAug = augmentedImageDatastore(inputSize(1:2),testImgs); %validation

miniBatchSize = 1;
valFrequency = floor(numel(trainAug.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',20, ...
    'InitialLearnRate',0.0001, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');
landnet = trainNetwork(trainAug, ChenLayer, options);

[YPred,probs] = classify(landnet, testAug);
accuracy = mean(YPred == testImgs.Labels)

confusionchart(testImgs.Labels,YPred)
