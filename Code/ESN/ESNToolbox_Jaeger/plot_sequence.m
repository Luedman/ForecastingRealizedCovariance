function plot_sequence(teacherSequence, predictedSequence, nPoints, caption)
% PLOT_SEQUENCE plots each dimension from the both the teacherSequence and
% the sequence predicted by the network. The first nPoints values from each
% sequence are plotted. 
%
% inputs: 
% teacherSequence: matrix of size nDataPoints x nDimensions
% predictedSequence: matrix of size nDataPoints x nDimensions
% nPoints: a natural number. 
%           The first nPoints from the input time series are plotted
% caption: a string containing the caption of the figure

%
% Created April 30, 2006, D. Popovici
% Copyright: Fraunhofer IAIS 2006 / Patent pending
% Revision 1, Feb 23, 2007, H. Jaeger

if nargin == 3
    caption = '' ; 
end


nFigure = figure ;

nDimensions = length(teacherSequence(1,:)) ; 

for iPlot = 1 : nDimensions
    subplot(nDimensions,1,iPlot) ; 
    %%%% set the caption of the figure    
    title(caption);
    hold on ; 
    plot(teacherSequence(1:nPoints, iPlot),'r') ; 
    plot(predictedSequence(1:nPoints, iPlot)) ; 
end
