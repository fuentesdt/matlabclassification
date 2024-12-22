
%myreshapelayer = dlhdl.layer.reshapeLayer(7,7,32,Name="reshape1")

forwardlayers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    %maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(2,8,'Stride',2,"Weights",ones(2,2,8,8),"Bias",zeros(1,1,8),"WeightLearnRateFactor",0,"BiasLearnRateFactor",0)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    %maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(2,16,'Stride',2,"Weights",ones(2,2,16,16),"Bias",zeros(1,1,16),"WeightLearnRateFactor",0,"BiasLearnRateFactor",0)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    ];

transposelayers = [
    featureInputLayer(10,Name='input')
    fullyConnectedLayer(1568)
    %myreshape:oadayer 
    reshapeLayer("reshape")

    reluLayer
    %batchNormalizationLayer
    transposedConv2dLayer(3,32)

    %maxPooling2dLayer(2,'Stride',2)
    transposedConv2dLayer(2,32,"Stride",[2 2],"Weights",ones(2,2,32,32),"Bias",zeros(1,1,32))

    reluLayer
    %batchNormalizationLayer
    transposedConv2dLayer(3,16)

    %maxPooling2dLayer(2,'Stride',2)
    transposedConv2dLayer(2,16,"Stride",[2 2],"Weights",ones(2,2,16,16),"Bias",zeros(1,1,16))

    reluLayer
    %batchNormalizationLayer
    transposedConv2dLayer(3,8)
    
    %softmaxLayer
    %classificationLayer
    ];


forwardnet = dlnetwork(forwardlayers )
analyzeNetwork(forwardnet )
%transposenet = dlnetwork(transposelayers)

%% 
% Compute the singular value decomposition of |A|, returning the six largest 
% singular values and the corresponding singular vectors. Specify a fourth output 
% argument to check convergence of the singular values.
load west0479
A = west0479;

[U,S,V,cflag] = svds(A);
cflag

s = svds(@(x,tflag) Afun(x,tflag,A),size(A),10)

function y = Afun(x,tflag,A)
   if strcmp(tflag,'notransp')
       y =  A * x;
   else
       y = A' *x ; 
   end

end


