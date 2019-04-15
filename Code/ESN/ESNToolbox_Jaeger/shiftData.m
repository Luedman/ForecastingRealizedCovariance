function shiftedData = shiftData(data, shifts)
% Shifts data by shifts.
%
% Input args:
%
% data: a dataset, either real-valued array of size N by dim, or a cell array of size
%    [nrSamples, 1], where the i-th cell is a real-valued array of size N_i
%    by dim.
% shifts: a row vector of reals of size [1 dim]
%
% Outputs:
%
% shiftedData: a dataset of same size as data, obtained from data by  
%   column-wise shifting:  newColumn = oldColumn + shiftconstant
%
% Created by H. Jaeger, June 21, 2006
if isnumeric(data)
    shiftedData = data;
    dim = size(data,2);    
    for d = 1:dim        
        shiftedData(:,d) = data(:,d) + shifts(1,d);         
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
    shiftedData = data;    
    for d = 1:dim        
        for n = 1:nrSamples
            shiftedData{n,1}(:,d) = data{n,1}(:,d) + shifts(1,d);  
        end        
    end   
else error('input data must be array or cell structure');
end


