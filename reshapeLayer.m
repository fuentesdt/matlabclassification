classdef reshapeLayer < nnet.layer.Layer

    properties

    end

    properties (Learnable)

    end

    methods
        function layer = reshapeLayer(name)
            layer.Name = name;
        end

        function [Z] = predict(layer, X)
            Z = reshape(X,7,7,32,[]);
        end
    end
    
end
