function analogSignal = unitCodedToAnalog(unitCodedSignal)
% re-transforms unit coded signal (created by rawToUnitCoded) into analog signal
% unitCodedSignal is a matrix whose row index is time and whose columns correspond to 
% membership responses of coding units to analog (one-dim) signal
%
% Created by H. Jaeger, Nov 3, 2006
% Revision March 7, 2007 H. Jaeger (normalize input signal to unit
% component sum)
% Revision 2, July 1, 2007, H. Jaeger (renamed some vars)

unitCodedSignalNormalized = unitCodedSignal;
% unitCodedSignalNormalized = max(0, unitCodedSignalNormalized);
sumSig = sum(unitCodedSignalNormalized');


[l, p] = size(unitCodedSignal);

for n = 1:l
    if sumSig(n) > 0
        unitCodedSignalNormalized(n,:) = unitCodedSignalNormalized(n,:)  / sumSig(n);
    end
end

decoder = 0:p - 1;
decoder = decoder';
stretchedSignal = unitCodedSignalNormalized * decoder;
analogSignal = stretchedSignal / (p - 1);
