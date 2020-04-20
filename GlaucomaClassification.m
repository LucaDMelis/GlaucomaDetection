close all;
pathToData = "C:\Users\lucaf\Desktop\EAI\Ophtamologia\Tesi\data3";

% Importazione di una rete neurale(Alexnet)
net = alexnet;
layers = net.Layers;

% Modifica degli ultimi layer
layers(end-2) = fullyConnectedLayer(2);
layers(end) = classificationLayer();

% Acquisizione del datastore 
imds = imageDatastore(pathToData,'IncludeSubfolders',true,'LabelSource','foldernames');
[trainImgs,valImgs,testImgs] = splitEachLabel(imds,0.8,0.1,0.1,'randomized');

% Modifiche sulla grandezza delle immagini
imageAugmenter = imageDataAugmenter('RandRotation',[-45,45],'RandXTranslation',[-10 10], 'RandYTranslation',[-10 10], ...
    'RandXReflection', 1);
trainAug= augmentedImageDatastore([227 227],trainImgs,'DataAugmentation',imageAugmenter);
valAug = augmentedImageDatastore([227 227],valImgs,'DataAugmentation',imageAugmenter);
testAug = augmentedImageDatastore([227 227],testImgs,'DataAugmentation',imageAugmenter);

minibatch = read(trainAug);
imshow(imtile(minibatch.input))  
% Training della rete
options = trainingOptions('adam','InitialLearnRate',0.0001,'Plots', 'training-progress', 'MaxEpochs', 10, 'MiniBatchSize', 10,...
'Shuffle','every-epoch', 'ValidationData', {valAug valImgs.Labels}, 'ValidationFrequency', 10);
landnet = trainNetwork(trainAug, layers, options);

% Valutazione della rete
testPred = classify(landnet, testAug);
acc = nnz(testPred == testImgs.Labels)/length(testPred)
