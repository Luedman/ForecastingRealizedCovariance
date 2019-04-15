function [normData, shifts] = normalizeData0Mean(data)
% Normalizes a multivariate dataset data (where each data vector is a row in data) 
% by shifting such that in each column mean becomes 0. 
%
% 
% Input arg: 
% data: a dataset, a real-valued array of size N by dim 
%
% Outputs:
% normData: the dataset normalized to columns with zero mean.
% shifts: a row vector of length dim giving the shiftconstants, signed s.th.
%         normData = data + shifts
%
% Created by H. Jaeger, Sep 29, 2010
  
shifts = - mean(data);
normData = data + repmat(shifts, size(data,1), 1); 

