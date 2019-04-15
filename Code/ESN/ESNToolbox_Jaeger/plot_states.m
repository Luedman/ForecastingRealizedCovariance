function plot_states(stateMatrix, states, nPoints, figNr, titletext)

% PLOT_STATES plots the internal states of the esn
%
% inputs:
% stateMatrix = matrix of size (nTrainingPoints) x
% (nInputUnits + nInternalUnits )
% stateMatrix(i,j) = internal activation of unit j after the
% i-th training point has been presented to the network
% states = vector of size 1 x n , containing the indices of the internal
% units we want to plot
% nPoints = natural number containing the number of points to plot
% figNr: either [] or an integer. If [], a new figure is created, otherwise
% the plot is displayed in a figure window with number figNr
% titletext: a string which is displayed as title over first panel
%
% example  : plot_states(stateMatrix,[1 2 3 4],200) plots the first 200
% points from the traces of the first 4 units

%
% Created April 30, 2006, D. Popovici
% Copyright: Fraunhofer IAIS 2006 / Patent pending
% Revision 1, June 6, 2006, H. Jaeger
% Revision 2, July 2, 2007, H. Jaeger
% Revision 3, Aug 17, 2007, H. Jaeger


if isempty(figNr)
  figure ; clf;
else
  figure(figNr); clf;
end


nStates = length(states) ;

xMax = ceil(sqrt(nStates)) ;
yMax = ceil(nStates /xMax);

for iPlot = 1 : nStates
  subplot(xMax,xMax,iPlot) ;
  plot(stateMatrix(1:nPoints, states(1,iPlot)));
  if iPlot == 1
    title(titletext);
  end
end

