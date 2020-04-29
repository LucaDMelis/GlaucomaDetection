close all;
pathToData = "C:\Users\lucaf\Desktop\EAI\Ophtamologia\Tesi\data3";

% Acquisizione del datastore 
imds = imageDatastore(pathToData,'IncludeSubfolders',true,'LabelSource','foldernames');
[trainImgs,valImgs,testImgs] = splitEachLabel(imds,0.8,0.1,0.1,'randomized');

% Importazione di una rete neurale(Googlenet)
net = googlenet;
%layers = net.Layers;
inputSize = net.Layers(1).InputSize;

% Per trovare i layer da sostituire
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer] 

numClasses = numel(categories(imds.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);



newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])

layers = lgraph.Layers;
connections = lgraph.Connections;

% Opzionale
%layers(1:10) = freezeWeights(layers(1:10));

% Training della rete
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainImgs, ...
    'DataAugmentation',imageAugmenter);
% Modifiche sulla grandezza delle immagini

trainAug= augmentedImageDatastore([224 224 3],trainImgs,'DataAugmentation',imageAugmenter);
valAug = augmentedImageDatastore([224 224 3],valImgs, 'DataAugmentation',imageAugmenter); %test
testAug = augmentedImageDatastore([224 224 3],testImgs); %validation

miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',testAug, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');
landnet = trainNetwork(trainAug, lgraph, options);

% Valutazione della rete
[YPred,probs] = classify(landnet, testAug);
accuracy = mean(YPred == testImgs.Labels)

% Display four sample validation images with predicted labels and the predicted
% probabilities of the images having those labels.
idx = randperm(numel(testImgs.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(testImgs,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end

