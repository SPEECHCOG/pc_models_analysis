function ZS = loadZSData2017(languages,audiopath)
% function ZS = loadZSData2017(languages,audiopath)
% 
% Returns file paths for train and test files for ZS2017 challenge.
%
% Inputs:
%
%   languages = {'english','mandarin','french','LANG1','LANG2'} or any
%               subset
%
%   audiopath = location of ZS2017 audio folders "train" and "test"   


if nargin <1
    languages = {'english','mandarin','french','LANG1','LANG2'};
elseif(~iscell(languages))
    tmp = languages;    
    languages = cell(1,1);
    languages{1} = tmp;
end

if nargin <2
    audiopath = '/Users/rasaneno/speechdb/zerospeech2020/2017/';
end

ZS = struct();



for lang = languages    
    d = [audiopath '/' lang{1} '/train/'];    
    tmp = dir([d '/*.wav']);    
    
    ZS.(lang{1}).train.filename = cell(length(tmp),1);
    for k = 1:length(tmp)
       ZS.(lang{1}).train.filename{k} = [d tmp(k).name]; 
    end
    
    
    d = [audiopath '/' lang{1} '/test/1s/'];    
    tmp = dir([d '/*.wav']);    
    
    ZS.(lang{1}).test.filename_1 = cell(length(tmp),1);
    for k = 1:length(tmp)
       ZS.(lang{1}).test.filename_1{k} = [d tmp(k).name]; 
    end
    
    d = [audiopath '/' lang{1} '/test/10s/'];    
    tmp = dir([d '/*.wav']);    
    
    ZS.(lang{1}).test.filename_10 = cell(length(tmp),1);
    for k = 1:length(tmp)
       ZS.(lang{1}).test.filename_10{k} = [d tmp(k).name]; 
    end
    
     d = [audiopath '/' lang{1} '/test/120s/'];    
    tmp = dir([d '/*.wav']);    
    
    ZS.(lang{1}).test.filename_120 = cell(length(tmp),1);
    for k = 1:length(tmp)
       ZS.(lang{1}).test.filename_120{k} = [d tmp(k).name]; 
    end   
    
end


