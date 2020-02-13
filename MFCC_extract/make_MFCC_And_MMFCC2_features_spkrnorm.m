% This function is used to evaluate the features which is subsequently used
% by the ACORNS recognition systems, such as CM, DP-N Gram and NMF. Also,
% this can be used by standard HTK.
% It can generate either MFCC features or MMFCC features

function O= make_MFCC_And_MMFCC2_features_spkrnorm(input,WinLen,WinShift,SigInpFs,SigOprFs,MFCC_flag,MMFCC_flag,Dynamic_flag,CMVN_flag,NoiseType,SNR,NoiseDBPath,train_flag,talker_id,shift)

% input -> Input speech signal with sampl freq=44.1 kHz
% WinLen -> Window length in second 
% WinShift -> Window shift in second
% SigInpFs -> Sampling frequency (in Hz) of input speech signal (Default=44100)
% SigOprFs -> Sampling frequency (in Hz) of operating speech signal (Default=16000)
% MFCC_flag -> If set, then output feature = MFCC parameters. Otherwise put
% it off.
% MMFCC_flag -> If set, then output feature = MMFCC parameters. Otherwise put
% it off.
% Dynamic_flag -> If set, then output will be 39-dimensional feature of
% E+feature+Vel+Acc.
% CMVN_flag -> If set, then output feature will be Cepstrum Mean and
% Variance Normalized (CMVN). The CMVN is carried out in
% utterance-by-utterance basis (i.e. file-by-file).
% NoiseType -> This specifies the noise type to corrupt the test speech
% signal (babble, pink, volvo and white)
% SNR -> SNR in dB  (Choose SNR=1000 for clean testing)
% NoiseDBPath -> Specify the path of noise database directory

% O <- Feature. 12 dimensional if Dynamic_flag=0. Else, 39 dimensional if
% Dynamic_flag=1.



% -------------------- Input speech signal processing ---------------------


% If input signal Fs is not 44100, then error
% if (SigInpFs~=44100)
%    
%    error('Input Signal Fs is not equal to 44100 Hz');
%    
% end

% If input signal operating Fs is not 16000, then error

if (SigOprFs~=16000)
   
   error('Operating Fs is not equal to 16000 Hz');
   
end


s = input;
s=s/max(abs(s));
s=s-mean(s);

s = [zeros(WinLen*16000,1);s;zeros(WinLen*16000,1)];

% % testing
% wavwrite(input/max(abs(input)),44100,'original.wav');
% plot(input);
% figure; plot(s,'r');
% wavwrite(s,16000,'downsampled.wav');
% pause
% close all;

% ------------------- Corrupting by additive noise ------------------------

% Loading the noise file




if (train_flag == 1)  % Then traning and no input noise (clean condition training)
    
    s_noisy=s;
    
else
    
    
str1=cat(2,NoiseDBPath,NoiseType);
str1=cat(2,str1,'_16kHz.wav');
Noise=wavread(str1);

% Processing for input speech signal

Noise_For_Input=Noise(1:length(s));
    
    UV_Noise_For_Input=Noise_For_Input/std(Noise_For_Input);
    
    noise_var=(var(s) / (10^(SNR/10)));
    Noise_sample= (noise_var^(0.5)) * UV_Noise_For_Input;
    s_noisy=s+Noise_sample;  
    
end



clear s;


% wavwrite(s_noisy,16000,'downsampled_noisy.wav');


% -------------------------------------------------------------------------



    
% -------------------------------------------------------------------------

   

% ---------- Window length, shift and window type specification -----------

WinLenInSamples=WinLen*SigOprFs;      % Window length in samples
WinShiftInSamples=WinShift*SigOprFs;  % Window shift in samples

Nshift=WinShiftInSamples;

HamWin=hamming(WinLenInSamples)';
% -------------------------------------------------------------------------



% -------------- Specification of MFCC and MMFCC parameters ---------------
% Choice of either MFCC or MMFCC parameters
if ( (MFCC_flag==1) && (MMFCC_flag==0) )
    [O E]=MFCC_feature_sprknorm(s_noisy,SigOprFs,WinLenInSamples,Nshift,HamWin,talker_id,shift);
    
end

if ( (MFCC_flag==0) && (MMFCC_flag==1) )
    [O E]=MMFCC2_feature(s_noisy,SigOprFs,WinLenInSamples,Nshift,HamWin);
end

if ( (MFCC_flag==0) && (MMFCC_flag==0) )
    error('Both the MFCC_flag and MMFCC_flag are same');
end

if ( (MFCC_flag==1) && (MMFCC_flag==1) )
    error('Both the MFCC_flag and MMFCC_flag are same');
end
% -------------------------------------------------------------------------





% ---------------- If Dynamic Parameters are required ---------------------

if (Dynamic_flag == 1),
    
    % Energy and Dynamic features 
    
    E_And_Feature=[log10(E') O];
    
    E_And_Feature(1,:) = 1;
    Delta_Feature=zeros(size(E_And_Feature));
    Delta_Delta_Feature=zeros(size(E_And_Feature));
    
    [J col1]=size(Delta_Feature);
    
    for j=1:J
        
        if (j==1)
            Delta_Feature(j,:) = E_And_Feature(j+1,:) - E_And_Feature(j,:);
        elseif (j==J)
            Delta_Feature(j,:)=E_And_Feature(j,:) - E_And_Feature(j-1,:);
        else
            Delta_Feature(j,:)=0.5 * (E_And_Feature(j+1,:) - E_And_Feature(j-1,:));
        end
        
    end
    
    for j=1:J
        
        if (j==1)
            Delta_Delta_Feature(j,:) = Delta_Feature(j+1,:) - Delta_Feature(j,:);
        elseif (j==J)
            Delta_Delta_Feature(j,:) = Delta_Feature(j,:) - Delta_Feature(j-1,:);
        else
            Delta_Delta_Feature(j,:) = 0.5 * (Delta_Feature(j+1,:) - Delta_Feature(j-1,:));
        end
        
    end
    
    % Now, we create the feature vector for the speech input file
    O=[E_And_Feature Delta_Feature Delta_Delta_Feature];    
else   
    O = [log10(E') O];            
    O(1,:) = 1;
end

% -------------------------------------------------------------------------




% -------------- If Cepstrum Mean and Variance Normalized (CMVN) ----------

if (CMVN_flag==1)
    
    Mu=mean(O(2:end,:));
    Std=std(O(2:end,:));       
    
    [J p]=size(O);
    
    Mu_Mat=nncopy(Mu,J,1);
    Std_Mat=nncopy(Std,J,1);
    
    O= (O - Mu_Mat) ./ Std_Mat; 
    
end
% -------------------------------------------------------------------------




