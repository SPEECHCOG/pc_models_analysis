% This is a function which provides MFCC features

function [O E]=MFCC_feature_sprknorm(s,SigOprFs,WinLenInSamples,Nshift,HamWin,talker_id,shift)



% Number of triangular filters (Default value)
M=26;                         

% Dimension of the feature vector (Default value)
Q=12; 

% Compressing coefficient
a=1;

% Warping constant
K=700;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% Feature Preparation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% ------------------------ DCT Matrix -------------------------------------

% DCT matrix without the dc term 

dct_matrix = dct_func_without_dc_term(M);

dct_matrix = dct_matrix (1:Q,:);
% -------------------------------------------------------------------------



% ------------------- Triangular Filter Bank Paramaters -------------------

[StartFreq_warp, CentreFreq_warp, TerminatingFreq_warp]=WarpedFeatureParam(M,SigOprFs,WinLenInSamples,K);
% -------------------------------------------------------------------------



% ------------------------- Speech Frames Preparation ---------------------

framecount=0;

start=1;
finish=WinLenInSamples;

while (finish < (length(s)- WinLenInSamples))
    
    framecount=framecount+1;
    
           
    start=start+Nshift;
    finish=finish+Nshift;
        
end

s_M=zeros(framecount,WinLenInSamples);



framecount=0;

start=1;
finish=WinLenInSamples;
sig=s(start:finish);

while (finish < (length(s)- WinLenInSamples))
    
    framecount=framecount+1;
    
    s_M(framecount,:) = sig;      
       
    start=start+Nshift;
    finish=finish+Nshift;
    sig=s(start:finish);
    
end
% -------------------------------------------------------------------------




% ------------------- All the processing stages ---------------------------

% Energy of the speech frames
E=sum((s_M .* s_M)');


% Hamming windowed Power spectrum calculation
PS_M=power_spectrum(s_M,HamWin);


f_basic = 0:8000/size(PS_M,2):8000-1/size(PS_M,2);
f_new = f_basic.*exp(shift);

% Make warping matrix

MAPPER = zeros(length(f_basic));
for j = 2:length(f_basic);
    row = j;        % Old frequency bin
    
    [dists,inds] = sort(abs(f_new(j)-f_basic),'ascend');
    if(dists(1) == 0)
        inds = inds(1);
        dists = 1;
    else
    dists = 1./dists(1:2);
    inds = inds(1:2);
    dists = dists./sum(dists);
    end
    MAPPER(row,inds) = dists;
end

for k = 1:size(MAPPER,2)
   MAPPER(:,k) =  MAPPER(:,k)./sum(MAPPER(:,k));
end

MAPPER(1,1) = 1;





% MAPPER now describes where the new frequencies are mapped to 

%PS_M_new = zeros(size(PS_M));
%for j = 1:size(PS_M,1)
%    PS_M_new(j,:) = PS_M(j,:)*MAPPER;
%end



PS_M = PS_M * MAPPER; 



%save PS_M.mat PS_M%
%pause;

PS_M(isnan(PS_M)) = 0;
PS_M(PS_M == 0) = 0.0001;





% Filter banks (FB) using FB parameters
FB_warp=triangular_filter_bank(M, WinLenInSamples, StartFreq_warp, CentreFreq_warp, TerminatingFreq_warp);
%figure;plot(FB_warp','b','LineWidth',1);

% Filter bank energies (FBE) using FB
FBE_ini= PS_M * FB_warp' ;



% Compressed filter bank energies (Com_FBE)
Com_FBE_ini = log10( (a* FBE_ini) + ((1-a) * (FBE_ini .* FBE_ini)) );

% DCT transform to get the feature 
FCC_ini= dct_matrix * Com_FBE_ini';
FCC_ini=FCC_ini';


% The feature vector for the speech input file
O=FCC_ini;
% -------------------------------------------------------------------------






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% All the necessary function %%%%%%%%%%%%%%%%%%%%%%%%%%%%




% ---------------- Function to get DCT matrix -----------------------------

% This function provides the DCT matrix without DC term

function dct_matrix = dct_func_without_dc_term(M)

%   M  -> M point dct (which is determined by the number of trainagular
%   filters

%   dct_matrix <- DCT matrix of (M-1)*M without the dc term 


T=zeros(M-1,M);

for q=1:M-1
    for m=0:M-1
        T(q,m+1)=cos(q*(m+0.5)*(pi/M));
    end
end

dct_matrix=T;
% -------------------------------------------------------------------------



% ------------------ Function to generate power spectrum ------------------

% This function provides the power spectrum of the speech signal matrix

function PS_M=power_spectrum(s_M,HamWin)

% PS_M <- Power spectrum matrix  (row index is the speech frame index) 

% s_M ->  speech signal matrix
% HamWin -> Hamming window

[row col]=size(s_M);

PS_M=zeros(row,(length(HamWin)/2)+1);


for j=1:row
    
    sig_w=s_M(j,:) .* HamWin;                  % For original signal    
    sig_fft=fft(sig_w);
    sxx_sig= (abs(sig_fft)).^2;
    
    sxx_sig_part=sxx_sig(1:length(sxx_sig)/2 + 1); 
    
    PS_M(j,:)=sxx_sig_part;
    
end
% -------------------------------------------------------------------------



% ------------------- Function to get the FB parameters -------------------

% This file is to provide the structure of filter parameters

function [StartFreq, CentreFreq, TerminatingFreq]=WarpedFeatureParam(M,Fs,WinLenInSamples,K)


% M -> Number of triangular filters (we start with uniform filter)
% Fs -> Sampling frequency
% WinLenInSamples -> Window length in samples
% K -> 'K' is the warping coefficient (The default value is K=K which is
% used for MFCC)

% StartFreq <- start frequency index
% CentreFreq <- centre frequency index
% TerminatingFreq <- terminating frequency index



% ------ Triangular Filters ------------

NyquistFreq=Fs/2;

DFTLength=WinLenInSamples;       % We use DFT length as the length of Window.


% Warped scale frequency

Max_Warp_Freq= 2595 * log10(1+NyquistFreq/K);  % K=700 for MFCC

del=Max_Warp_Freq/(M+1);

OmegaCentreFreq_Warp=zeros(1,M);
OmegaStartFreq_Warp=zeros(1,M);
OmegaTerminatingFreq_Warp=zeros(1,M);

for i=1:M
    
    % Centre Freq, Start Freq, Terminating Freq
    if i==1
        OmegaCentreFreq_Warp(i)=del;
        OmegaStartFreq_Warp(i)=0;
        OmegaTerminatingFreq_Warp(i)=OmegaCentreFreq_Warp(i)+del;        
    else
        OmegaCentreFreq_Warp(i)=OmegaCentreFreq_Warp(i-1)+del;
        OmegaStartFreq_Warp(i)=OmegaCentreFreq_Warp(i-1);
        OmegaTerminatingFreq_Warp(i)=OmegaCentreFreq_Warp(i)+del;
    end
    
    if (OmegaTerminatingFreq_Warp(M) > Max_Warp_Freq)
        OmegaTerminatingFreq(M)=Max_Warp_Freq;
    end
    
end


% Coming back to the normal frequency

OmegaCentreFreq=zeros(1,M);
OmegaStartFreq=zeros(1,M);
OmegaTerminatingFreq=zeros(1,M);

for i=1:M
    
    OmegaCentreFreq(i)=K * (10^(OmegaCentreFreq_Warp(i)/2595) - 1);
    OmegaStartFreq(i)=K * (10^(OmegaStartFreq_Warp(i)/2595) - 1);
    OmegaTerminatingFreq(i)=K * (10^(OmegaTerminatingFreq_Warp(i)/2595) - 1);
    
    if (OmegaTerminatingFreq(M) > NyquistFreq)
        OmegaTerminatingFreq(M)=NyquistFreq;
    end
end




% Freq on the DFT scale

CentreFreq=zeros(1,M);
StartFreq=zeros(1,M);
TerminatingFreq=zeros(1,M);

for i=1:M
    
    CentreFreq=round(OmegaCentreFreq * (DFTLength/Fs));
    StartFreq=round(OmegaStartFreq * (DFTLength/Fs));
    TerminatingFreq=round(OmegaTerminatingFreq * (DFTLength/Fs));
    
    StartFreq(1)=0;
    TerminatingFreq(M)=DFTLength/2;
    
end

% -------------------------------------------------------------------------



% --------------- Function to get the triangular FB -----------------------

% This function provides triangular filter banks

function FB = triangular_filter_bank(M, WinLenInSamples, StartFreq, CentreFreq, TerminatingFreq)

% M -> number of triangular filters
% WinLenInSamples -> Window length in number of samples for which the DFT
% is calculated.
% StartFreq -> start frequency index
% CentreFreq -> centre frequency index
% TerminatingFreq -> terminating frequency index

% FB <- A matrix containg the filter banks




% Window function

DFTLength=WinLenInSamples;

TriangleWindow=zeros(M,DFTLength/2+1);

for i=1:M
    
    DummyWindow=zeros(1,DFTLength/2+1);
    
    for k=0:DFTLength/2
        
        if ( (StartFreq(i) <= k) && (k <= CentreFreq(i)))
            DummyWindow(k+1)= 2 * (TerminatingFreq(i)*CentreFreq(i) - StartFreq(i)*CentreFreq(i) ...
                - TerminatingFreq(i)*StartFreq(i) + StartFreq(i)^2 )^(-1) * (k-StartFreq(i));
        elseif ( (CentreFreq(i) < k) && (k <= TerminatingFreq(i)))
            DummyWindow(k+1)= 2 * (StartFreq(i)*CentreFreq(i) - StartFreq(i)*TerminatingFreq(i) ...
                - CentreFreq(i)*TerminatingFreq(i) + TerminatingFreq(i)^2 )^(-1) * (CentreFreq(i)-k) ...
                + 2 * (TerminatingFreq(i)-StartFreq(i))^(-1);
        end
        
    end
    
    TriangleWindow(i,:)=DummyWindow;
    
end


FB=TriangleWindow;

% -------------------------------------------------------------------------

