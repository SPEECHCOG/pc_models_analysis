function [offset,MI] = alignTwoFeatures(F1,F2,maxlen)
% Finds relative time shift of information in two sets of feature matrices or
% tensors (3-d) F1 and F2.
% 
% negative offset means that F2 is delayed by that number of frames with
% respect to F1. Positive offset means that F2 is earlier than F1 by that
% number of frames.
%
% Code: Okko Räsänen, 2020. okko.rasanen@tuni.fi

if nargin <3
    maxlen = 300;
end
    


if(size(F1,3) > 1)
    s = [];
    y = [];
    for k = 1:min(size(F1,1),maxlen)
        s = [s;squeeze(F1(k,:,:))];
        y = [y;squeeze(F2(k,:,:))];
    end
elseif(iscell(F1) && iscell(F2))
    s = [];
    y = [];
    for k = 1:min(length(F1),maxlen)
        s = [s;F1{k}];
        y = [y;F2{k}];
    end
    
elseif(ndims(F1) == 2 && ndims(F2) == 2)
    s = F1;
    y = F2;
end

idx_s = kmeans(s,64);
idx_y = kmeans(y,64);

MI = zeros(31,1);
for lag = -15:15
    MI(lag+16) = mutInfcross(idx_s,idx_y,lag);    
end

[a,b] = max(MI);

offset = b-16;