classdef reshapeLayer < nnet.layer.Formattable

    properties

    end

    properties (Learnable)

    end

    methods
        function layer = reshapeLayer(name)
            layer.Name = name;
        end

        function [Z] = predict(layer, X)
            Z = reshape(X,7,7,32,1);
        end
    end
    
end
