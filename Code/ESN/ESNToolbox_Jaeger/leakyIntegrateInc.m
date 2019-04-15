function timeseriesLI = leakyIntegrateInc(timeseries, LR, inc)
% Perform a leaky integration on a (possibly multidim) timeseries. 
% It is assumed that the timeseries was sampled with stepsize inc. 
% The applied formula is the linear discretization of 
%    d LI(x) / dt = - LR LI(x) + x, which is
%    (LI(x))(t+inc) = (1 - inc LR) (LI(x))(t) + inc x(t)
%
% Inputs:
% timeseries: array of size timeseries-dimension x runLength
% LR: leaking rates; column vector of length timeseries-dimension, 
%     or scalar
%
%
% Outputs:
% timeseriesLI: the leaky-integrated timeseries (using all zero
%               integrated state for starting the leaky
%               integration)
%
% Created Feb 28 2010 HJaeger

[dim runLength] = size(timeseries);


if not((size(LR,1) == dim && size(LR,2) == 1) || ...
    (size(LR,1) == 1 && size(LR,2) == 1))
  error('leaking rate input to leakyIntegrateInc: dimension error');
end

if inc * LR > 1
    error('inc * LR > 1');
end


if dim > 1 && size(LR,1) == 1
  leakingRates = LR * ones(dim,1);
else
  leakingRates = LR;
end

retainRates = 1 - inc * leakingRates;

timeseriesLI = zeros(dim, runLength);
datapointLI = zeros(dim, 1); % initialization
for n = 1:runLength
  datapointLI = inc * timeseries(:,n) + retainRates .* datapointLI;
  timeseriesLI(:,n) = datapointLI;
end


