function internalState = leaky1_esn(totalstate , esn , varargin )
% Update internal state using the leaky integrator neuron model with equation
% \mathbf{x}_{n+1} = (1 - \gamma) \mathbf{x}_{n} + 
%                     + \gamma f(\rho \mathbf{W} \mathbf{x}_{n} 
%                                + s^{in} \mathbf{W}^{in}\mathbf{u}_{n+1} 
%                                + s^{fb} \mathbf{W}^{fb} \mathbf{y}_{n})
% (notation from paper H. Jaeger, M. Lukosevicius, D. Popovici (2007): 
%  Optimization and Applications of Echo State Networks with Leaky Integrator Neurons. 
%  Neural Networks 20(3), 335-352, 2007)
% \gamma is esn.timeConstant
%
% input arguments:
% totalstate: the previous totalstate, vector of size 
%     (esn.nInternalUnits + esn.nInputUnits + esn.nOutputUnits) x 1
% esn: the ESN structure
%
% output: 
% internalState: the updated internal state, size esn.nInternalUnits x 1
%
% Created from leaky_esn, July 1, 2007, H. Jaeger


    previousInternalState = totalstate(1:esn.nInternalUnits, 1);
    internalState = (1 -  esn.timeConstants) .* previousInternalState + esn.timeConstants .* ...
        feval(esn.reservoirActivationFunction ,...
        [ esn.internalWeights, esn.inputWeights, esn.feedbackWeights * diag(esn.feedbackScaling )] ...
        * totalstate) ; 
    
    %%%% add noise to the Esn 
internalState = internalState + esn.noiseLevel * (rand(esn.nInternalUnits,1) - 0.5) ; 