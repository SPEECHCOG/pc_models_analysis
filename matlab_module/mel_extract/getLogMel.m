function [F,E] = getLogMel(signal,wl,ws,opfreq,mel_bands,is_log, ...
    usespecsub)

    if nargin <7
        usespecsub = 0;
    end
    if nargin<6
        is_log = true;
    end

    if nargin<5
        mel_bands = 24;
    end

    if nargin <4    
    opfreq = 16000;
    end

    if nargin <3
        ws = 0.0125*opfreq;
    else
        ws = round(ws*opfreq);
    end
    if nargin <2
        wl = 0.025*opfreq;
    else
        wl = round(wl*opfreq);
    end

    ww = hamming(wl);

    [MEL,ME,MS]= melbankm(mel_bands,wl,16000,0,0.5,'u');
    
    x = signal;

    x = [zeros(round(wl/2),1);x;zeros(round(wl/2),1)];

    if(usespecsub)
       x = specsub(x,fs);
    end


    F = zeros(round(length(x)/ws)-2,size(MEL,1));
    E = zeros(round(length(x)/ws)-2,1);
    j = 1;
    for loc = 1:ws:length(x)-wl+1
       y = x(loc:loc+wl-1).*ww;
       tmp = abs(fft(y));
       tmp = tmp(2:wl/2);
       y = MEL*tmp;
       if is_log
           y = 20*log10(y);
       end

       F(j,:) = y;
       E(j) = sum(tmp);
       j = j+1;       
    end      
   
end