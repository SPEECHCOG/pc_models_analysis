function MEL = getMelSpectrogram(data, window_length, ... 
    window_shift, bands, is_log)
%%%
% @author María Andrea Cruz Blandón
% @date 28.02.2020
% This script calculate the mel spectrogram for a data set. It returns
% the cell MEL with the bands-dimensional Mel spectrograms of each 
% file in data
%%%

MEL = cell(length(data),1);
N = length(data);

for k=1:length(data)
    file_path = data{k};
    
    try
        [input,fs] = audioread(file_path);
    catch exception
        
        if(contains(file_path,'TIMIT'))
            [input,fs] = readsph(file_path);
        else
            rethrow(exception);
        end
    end
    
    if(fs ~= 16000)
        input = resample(input,16000,fs);
        fs = 16000;
    end
    win_len_samples = round(fs*window_length);
    win_shift_samples = round(fs*window_shift);
    
    mel_ = melSpectrogram(input, fs, 'WindowLength', win_len_samples, ...
        'OverlapLength', win_shift_samples, 'NumBands', bands);
    if is_log
        % Compute a stabilized log to get log-magnitude mel-scale 
        % spectrograms. (Tensorflow api doc)
        MEL{k} = log(mel_ + eps);
    else
        MEL{k} = mel_;
    end
    
    procbar(k,N);
end

end