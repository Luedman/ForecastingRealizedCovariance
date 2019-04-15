function [inputSequence, outputSequence] = generate_NARMA_sequence(sequenceLength, memoryLength)
%  Generates a sequence using a nonlinear autoregressive moving average
% (NARMA) model. The sequence at the beginning includes a  ramp-up
% transient, which should be deleted if necessary. The NARMA equation to be
% used must be hand-coded into this function (at bottom)

% inputs: 
% sequenceLength: a natural number, indicating the length of the
% sequence to be generated
% memoryLength: a natural number indicating the dependency length
%
% outputs: 
% InputSequence: array of size sequenceLength x 2. First column contains 
%                uniform noise in [0,1] range, second column contains bias 
%                input (all 1's)             
% OutputSequence: array of size sequenceLength x 1 with the NARMA output
%
% usage example:
% [a b] = generate_linear_sequence(1000,10) ; 
%
% Created April 30, 2006, D. Popovici
% Copyright: Fraunhofer IAIS 2006 / Patent pending
% Revision H. Jaeger Feb 23, 2007

%%%% create input 
inputSequence = [ones(sequenceLength,1) rand(sequenceLength,1)];

% use the input sequence to drive a NARMA equation

outputSequence = 0.1*ones(sequenceLength,1); 

for i = memoryLength + 1 : sequenceLength
    % insert suitable NARMA equation on r.h.s., this is just an ad hoc
    % example
    outputSequence(i,1) = 0.7 * inputSequence(i-memoryLength,2) + 0.1 ...
        + (1 - outputSequence(i-1,1)) * outputSequence(i-1,1);
end

