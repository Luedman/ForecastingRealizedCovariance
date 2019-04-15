function unitCodedSignal = analogToUnitCoded(analogSignal, nUnits)
% transforms a 1-dimensional Signal analogSignal with range [0,1] into
% nUnits-dimensional signal
% (each dim ranging in [0,1]) through nUnits many triangular, equally
% spaced membership functions. The component sum of each signal time point
% vector is normalized to 1.
%
% unitCodedSignal has size (length(analogSignal) , nUnits)
%
% Created by H. Jaeger, Nov 3, 2006
% Revision 1, Junly 1, 2007, H. Jaeger (renamed this funcation and some vars)

l = length(analogSignal);
unitCodedSignal = zeros(l, nUnits);
stretchedSignal = analogSignal * (nUnits - 1);

for i = 1:l
    for p = 1:nUnits
        unitCodedSignal(i, p) = unitTriangle(stretchedSignal(i,1) - p + 1);
    end  
end

function membershipValue = unitTriangle(x)
% computes the membership function of a unit triangle centered at zero
if abs(x) > 1
    membershipValue = 0;
else
    membershipValue = 1 - abs(x);
end