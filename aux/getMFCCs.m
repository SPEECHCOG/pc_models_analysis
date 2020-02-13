function F = getMFCCs(data_train,train_flag,NoiseType,SNR,WinLen,WinShift,CMVN,Dynamic_flag) 
%function F = getMFCCs(data_train,train_flag,NoiseType,SNR,WinLen,WinShift,CMVN,Dynamic_flag) 
%
% 

if nargin <8
    Dynamic_flag=0;
end

if nargin <7
    CMVN = 0;
end

if nargin <6
    WinShift = 0.01;
end
if nargin <5
    WinLen = 0.025;
end

MFCC_flag=1;
MMFCC_flag=0;

% Cepstrum Mean and Normalization (CMVN). If set, then output feature will be Cepstrum Mean and
% Variance Normalized (CMVN). The CMVN is carried out in
% utterance-by-utterance basis (i.e. file-by-file).

if(CMVN)
    CMVN_flag=1;
else
    CMVN_flag=0;
end

% Specification about Fs, Window Length and Window Shift
SigOprFs=16000;                  % Signal Operating Sampling Frequency

% Noise parameters

%NoiseType = 'white';   % (babble, pink, volvo and white)
%SNR = 10;        % SNR in dB  (Choose SNR=1000 for clean testing)
NoiseDBPath = '/Users/orasanen/speechdb/NoiseX_16khz/';
%train_flag = 0; % set 1 if clean training

F = cell(length(data_train),1);
N = length(data_train);

for k = 1:length(data_train);
       
    filu = data_train{k};
    
    try
        [input,fs] = audioread(filu);
    catch exception
        
        if(strfind(filu,'TIMIT'))
            [input,fs] = readsph(filu);
        else
            rethrow(exception);
        end
    end
    
    if(fs ~= 16000)
        input = resample(input,16000,fs);
        fs = 16000;
    end
    
    a = find(input == 0);
        
    O = make_MFCC_And_MMFCC2_features(input,WinLen,WinShift,fs,SigOprFs,MFCC_flag,MMFCC_flag,Dynamic_flag,CMVN_flag,NoiseType,SNR,NoiseDBPath,train_flag);
    
    F{k} = O;
        
    procbar(k,N);
    
end
fprintf('\n');