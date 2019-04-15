function timeConstants = makeTimeConstantsFromRange(lowerBoundary, upperBoundary, esn)
% helper function, creates a timeConstant vector (to be assigned to
% esn.timeConstants) from a lower and upper boundary (which must be in the
% interval (0,1] and may be identical)
%
% Created July 1, 2007, H.Jaeger

range = upperBoundary - lowerBoundary;
timeConstants = (0:esn.nInternalUnits-1)' / (esn.nInternalUnits-1);
timeConstants = timeConstants * range + lowerBoundary;
