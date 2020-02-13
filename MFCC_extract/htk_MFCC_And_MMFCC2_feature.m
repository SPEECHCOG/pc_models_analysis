% This programme is used for generating MFCC and MMFCC features to use in
% HTK framework.
% This programme has to run seperately once fro training data and once for
% testing data

clear
close all


% If train flag is 1 then, training. Note that, we always perform training using clean speech data.
% If train flag is 0 then, testing. We can perform clean and robust testing
% using appropriate SNR level
train_flag=0;


% Noise specification (only used when train_flag=0)
NoiseDBPath='../MFCC&MMFCC2/NoiseX_16kHz/';  % Specify the path of noise database directory
NoiseType='babble';  % This specifies the noise type (babble, pink, volvo and white)
SNR=20;              % SNR in dB  (Choose SNR=1000 for clean testing)



% If we use MFCC, then MFCC_flag=1 and MMFCC_flag=0
% Else if we use MMFCC, then MFCC_flag=0 and MMFCC_flag=1
MFCC_flag=0;
MMFCC_flag=1;


% Dynamic parameter. If this flag is set, then the output vector will be 39
% dimensional as E+FeatureVec+Vel+Acc. If this flag is off, then the output
% will be only 12 dimensional feature.
Dynamic_flag=1;


% Cepstrum Mean and Normalization (CMVN). If set, then output feature will be Cepstrum Mean and
% Variance Normalized (CMVN). The CMVN is carried out in
% utterance-by-utterance basis (i.e. file-by-file).
CMVN_flag=1;


% Specification about Fs, Window Length and Window Shift
SigInpFs=44100;                  % Sampling Frequency of Input Signal in Hz
SigOprFs=16000;                  % Signal Operating Sampling Frequency in Hz
WinLen=0.032;                    % 32ms
WinShift=0.010;                  % 10 ms


if (train_flag==1)
    
    % For Training 
    fin=fopen('Train.wavlist');      
    fout=fopen('~/HTK_My/HTK_TIMIT1/HTK_TIMIT/work/TRAIN.mfclist');
    
    NoOfFiles= 1903;
        
else
    
    % For Testing 
    fin=fopen('Test.wavlist');      
    fout=fopen('~/HTK_My/HTK_TIMIT1/HTK_TIMIT/work/TEST.mfclist');
    
    NoOfFiles= 1903;
    
end



for i=1:NoOfFiles   % Here we need to correct; train -> 4620, test -> 1680
    i;
    filein=fgetl(fin);
    fileout=fgetl(fout);
    
    if mod(i,100)==0
        i
    end
    
    finput=fopen(filein,'r');
    input=fread(finput,'int16');
    input = input(513:end);
    fclose(finput);

    O= make_MFCC_And_MMFCC2_features(input,WinLen,WinShift,SigInpFs,SigOprFs,MFCC_flag,MMFCC_flag,Dynamic_flag,CMVN_flag,NoiseType,SNR,NoiseDBPath,train_flag);
    writehtk_new(fileout,O,32e-3,9);
         
%     plot(input)
%     i
%     pause
%     close;
    
end

fclose(fin);
fclose(fout);
