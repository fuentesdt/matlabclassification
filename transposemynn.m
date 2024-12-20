

myreshapelayer = dlhdl.layer.reshapeLayer(7,7,32,Name="reshape1")


transposelayers = [
    featureInputLayer(10,Name=input)
    fullyConnectedLayer(1568)
    myreshapelayer 

    reluLayer
    %batchNormalizationLayer
    transposedConv2dLayer(3,32)

    %maxPooling2dLayer(2,'Stride',2)
    transposedConv2dLayer(2,32,"Stride",[2 2],"Weights",ones(2,2,32),"Bias",ones(1,1,32))

    reluLayer
    %batchNormalizationLayer
    transposedConv2dLayer(3,16)

    %maxPooling2dLayer(2,'Stride',2)
    transposedConv2dLayer(2,16,"Stride",[2 2],"Weights",ones(2,2,32),"Bias",ones(1,1,16))

    reluLayer
    %batchNormalizationLayer
    transposedConv2dLayer(3,8)
    
    %softmaxLayer
    %classificationLayer
    ];
