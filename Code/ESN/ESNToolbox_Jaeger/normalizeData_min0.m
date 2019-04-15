function [normData, shifts] = normalizeData_min0(data)
% Normalizes a multivariate dataset data (where each data vector is a row in data) 
% by shifting such that in each column the min becomes 0.
% If the min and the max in some column coincide, this column is not
% changed. 
% 
% Input arg: 
% data: a dataset, either real-valued array of size N by dim or a cell array of size
%    [nrSamples, 1], where the i-th cell is a real-valued array of size N_i by dim 
%
% Outputs:
% normData: the dataset normalized to columns with min = 0. Each
%    column in normData is computed from the corresponding column in data by 
%    normalizedColumn = originalColum + shiftconstant. If
%    the input is a cell structure, the same scalefactors and shiftconstants are
%    applied across all cells, such that the *global* min of normData
%    becomes 0.
% shifts: a row vector of lenght dim giving the shiftconstants
%
% Created by H. Jaeger, June 21, 2006

if isnumeric(data)
    dim = size(data,2);
    mins = min(data); maxs = max(data);
    scalingsInv = maxs - mins;
    shifts = zeros(1,dim);
    normData = data;
    for d = 1:dim
        if scalingsInv(1,d) > 0
            shifts(1,d) = -mins(1,d);
            normData(:,d) = data(:,d) + shifts(1,d);    
        end
    end
elseif iscell(data)
    dim = size(data{1,1},2);
    nrSamples = size(data,1);
    %check if all cells have same dim
    for n = 1:nrSamples
        if size(data{n,1},2) ~= dim
            error('all cells must have same row dim');
        end
    end
    mins = min(data{1,1});
    maxs = max(data{1,1});
    for n = 1:nrSamples
        mins = min(mins, min(data{n,1}));
        maxs = max(maxs, max(data{n,1}));
    end
    scalingsInv = maxs - mins;
    shifts = zeros(1,dim);
    normData = data;
    for d = 1:dim
        if scalingsInv(1,d) > 0
            shifts(1,d) = -mins(1,d);
            for n = 1:nrSamples
                normData{n,1}(:,d) = data{n,1}(:,d) + shifts(1,d);  
            end
        end
    end   
else error('input data must be array or cell structure');
end



