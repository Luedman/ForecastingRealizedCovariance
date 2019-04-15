function plot_outputWeights(outweights, figNr)
% plot the output weights (of a trained ESN). For each output channel, a
% new color is used.
%
% input arguments:
% outweights: a weight matrix of size nOutputUnits x (nInputUnits +
% nInternalUnits)
% 
% figNr: either [] or an integer. If [], a new figure is created, otherwise
% the plot is displayed in a figure window with number figNr
%
% Created June 6, 2006, H. Jaeger


if isempty(figNr)    
    nFigure = figure ;
else
    nFigure = figNr;
end

figure(figNr); clf;
plot(outweights');

title('output weights'); 
