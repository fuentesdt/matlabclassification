% TrainABasicConvolutionalNeuralNetworkForClassificationExample

%myreshapelayer = dlhdl.layer.reshapeLayer(7,7,32,Name="reshape1")

forwardlayers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same','Weights',net.Layers(2).Weights,'Bias',net.Layers(2).Bias)
    reluLayer
    
    %maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(2,8,'Stride',2,'Weights',ones(2,2,8,8),'Bias',zeros(1,1,8),'WeightLearnRateFactor',0,'BiasLearnRateFactor',0)
    
    convolution2dLayer(3,16,'Padding','same','Weights',net.Layers(6).Weights,'Bias',net.Layers(6).Bias)
    reluLayer
    
    %maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(2,16,'Stride',2,'Weights',ones(2,2,16,16),'Bias',zeros(1,1,16),'WeightLearnRateFactor',0,'BiasLearnRateFactor',0)
    
    convolution2dLayer(3,32,'Padding','same','Weights',net.Layers(10).Weights,'Bias',net.Layers(10).Bias)
    reluLayer
    
    fullyConnectedLayer(10)
    ];

forwardnet = dlnetwork(forwardlayers )
analyzeNetwork(forwardnet )
analyzeNetwork(net )

transposelayers = [
    %featureInputLayer(10,Name='input')
    %fullyConnectedLayer(1568,Name='full')
    %myreshapeLayer 
    %reshapeLayer('reshape')
    %functionLayer(@(X) reshape(X, [7 7 32 1]),Name='reshape',Description='reshape')
    %functionLayer(@(X) dlarray(X,"SSBC"),Formattable=true,Acceleratable=true)
    imageInputLayer([7 7 32])

    reluLayer
    %batchNormalizationLayer
    transposedConv2dLayer(3,16,'Cropping','same','Weights',net.Layers(10).Weights,'Bias',zeros(1,1,16))

    %maxPooling2dLayer(2,'Stride',2)
    transposedConv2dLayer(2,16,'Stride',[2 2],'Weights',ones(2,2,16,16),'Bias',zeros(1,1,16))

    reluLayer
    %batchNormalizationLayer
    transposedConv2dLayer(3,8,'Cropping','same','Weights',net.Layers(6).Weights,'Bias',zeros(1,1,8))

    %maxPooling2dLayer(2,'Stride',2)
    transposedConv2dLayer(2,8,'Stride',[2 2],'Weights',ones(2,2,8,8),'Bias',zeros(1,1,8))

    reluLayer
    %batchNormalizationLayer
    transposedConv2dLayer(3,1,'Cropping','same','Weights',net.Layers(2).Weights,'Bias',zeros(1,1,1))
    
    ];


analyzeNetwork(transposelayers)
transposenet = dlnetwork(transposelayers)





%% 
% Compute the singular value decomposition of |A|, returning the six largest 
% singular values and the corresponding singular vectors. Specify a fourth output 
% argument to check convergence of the singular values.
% load west0479
% A = west0479;
% 
% [U,S,V,cflag] = svds(A);
% cflag

%% xrandmy = rand(1,784);
%% yout = myAfun(xrandmy,'notransp',net,forwardnet,transposenet)
%% yrandmy = rand(10,1);
%% xout = myAfun(yrandmy,'transp',net,forwardnet,transposenet)
s = svds(@(x,tflag) myAfun(x,tflag,net,forwardnet,transposenet),[10 784],10)

function y = myAfun(x,tflag,net,forwardnet,transposenet)
   if strcmp(tflag,'notransp')
       %y =  A * x;
       XX = reshape(x,[28 28 1]);
       dlX = dlarray(XX,"SSC");
       dlY = forward(forwardnet ,dlX);
       y=extractdata(dlY(:));
   else
       %y = A' *x ; 
       XX = reshape(net.Layers(13).Weights'*x,[7 7 32]);
       dlX = dlarray(XX,"SSC");
       dlY = forward(transposenet ,dlX);
       y=double(extractdata(dlY(:)));
   end

end


