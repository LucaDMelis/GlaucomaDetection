

layers = [
    imageInputLayer([256 256 3],"Name","imageinput")
    convolution2dLayer([11 11],3,"Name","conv1","Padding","same")
    reluLayer("Name","relu_1")
    maxPooling2dLayer([3 3],"Name","maxpool_1","Padding","same")
    convolution2dLayer([5 5],96,"Name","conv2","Padding","same")
    reluLayer("Name","relu_2")
    maxPooling2dLayer([3 3],"Name","maxpool_2","Padding","same")
    convolution2dLayer([3 3],192,"Name","conv3","Padding","same")
    maxPooling2dLayer([3 3],"Name","maxpool_3","Padding","same")
    convolution2dLayer([3 3],192,"Name","conv4","Padding","same")
    maxPooling2dLayer([3 3],"Name","maxpool_4","Padding","same")
    fullyConnectedLayer(10,"Name","fc_1")
    fullyConnectedLayer(10,"Name","fc_2")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")
];

%% Plot Layers
plot(layerGraph(layers));
