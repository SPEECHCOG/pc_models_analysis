function [I,M,SP] = mutInfcross(seq,seq2,dist)
%
% Computes mutual information of symbols at distance k in the sequence
%
% Compute probability of each symbol
% With a positive lag, seq1(t) vs. seq2(t+dist) (i.e., seq1 is preceding seq
% by dist elements


lambda = max(seq);

symbol_count = zeros(lambda,1);

for k = 1:lambda
    symbol_count(k) = sum(seq == k);
end



% Symbol probability

symbol_prob = symbol_count./sum(symbol_count)+0.000001;


lambda2 = max(seq2);


symbol_count2 = zeros(lambda2,1);

for k = 1:lambda2
    symbol_count2(k) = sum(seq2 == k);
end


% Symbol probability

symbol_prob2 = symbol_count2./sum(symbol_count2)+0.000001;

% Compute transition matrix at distance dist 

M = zeros(lambda,lambda2);

if(dist > 0)
for k = 1:length(seq)-dist
    loc1 = seq(k);    
    
    loc2 = seq2(k+dist);    
    
    M(loc1,loc2) = M(loc1,loc2)+1;    
end
else
    for k = abs(dist)+1:length(seq)
    loc1 = seq(k);    
    
    loc2 = seq2(k+dist);    
    
    M(loc1,loc2) = M(loc1,loc2)+1;    
    end    
end

SP = symbol_prob*symbol_prob2';

% Pair joint probability

%M = M./sum(M(:))+0.0000000001;
%I = sum(sum(M.*log2(M./SP))); %/log2(lambda);

% New variant
M = M./sum(M(:));

tmp = log2(M./SP);
tmp(isinf(tmp)) = 0;
tmp = M.*tmp;
M = tmp;
I = sum(tmp(:));









