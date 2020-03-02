
clear all

% Add paths
curdir = fileparts(which('ZS2020_demo_pipeline.m'));
addpath([curdir '/MFCC_extract/']);
addpath([curdir '/mel_extract/']);
addpath([curdir '/mel_extract/lib_voicebox/']);
addpath([curdir '/aux/']);
addpath([curdir '/misc/']);

% Read configuration:
configuration = readConfiguration();
config_feats = configuration.feature_extraction;

% Load filenames of each language
ZS = loadZSData2017(config_feats.languages,config_feats.audio_path); 

% Extract features and create tensors for model training per each language

for language=config_feats.languages
    % identify method for feature extraction.
    if strcmp(config_feats.method.name, 'mfcc')
        % Extract MFCCs (CMVN but no deltas)
        F_train = getMFCCs(ZS.(language{1}).train.filename,1,'white', ...
                           1000,config_feats.window_length, ...
                           config_feats.window_shift,1,0);
    elseif strcmp(config_feats.method.name, 'mel')
        F_train = getMelSpectrogram(ZS.(language{1}).train.filename, ...
            config_feats.window_length, config_feats.window_shift, ...
            config_feats.method.bands, false);
    elseif strcmp(config_feats.method.name, 'logmel')
        F_train = getMelSpectrogram(ZS.(language{1}).train.filename, ...
            config_feats.window_length, config_feats.window_shift, ...
            config_feats.method.bands, true);
    end
    
    % Sample size
    seqlen = round(config_feats.sample_length / config_feats.window_shift);
    
    % Features size
    feature_size = size(F_train{1},2);
  
    % total number of frames
    totlen = sum(cellfun(@length,F_train)); 
    % "reset" after each audio, except last one. Half of a sample
    reset_size = round(seqlen/2);
    reset_sample = zeros(reset_size,feature_size);
    
    % Concatenate all feature data into one large matrix  
    % F_all total_frams x features_size 
    F_all = zeros(totlen,feature_size);
    wloc = 1;
    for k = 1:length(F_train)
        F_all(wloc:wloc+size(F_train{k},1)-1,:) = F_train{k};
        wloc = wloc+size(F_train{k},1);
        % introduce reset
        if k < length(F_train)
            F_all(wloc:wloc+reset_size-1,:) = reset_sample;
            wloc = wloc+reset_size;
        end
    end

    % F_all contains now all features frames of the training set
    % Make sure there are no NaNs or Infs
    F_all(isnan(F_all)) = 0;
    F_all(isinf(F_all)) = 0;

    
    % NOTE:
    % Hack to cut training data: allow max 2 GB training data, because 
    % scipy cannot read larger than that .mat files. This doesn't cut data 
    % on Mandarin, but does so on English.
    maxlen = 2e9/(size(F_all,2)*8)-1;
    if(size(F_all,1) > maxlen)
        F_all = F_all(1:maxlen-1,:);
    end


    % For predictive modeling, create also temporally shifted versions of 
    % the data: short shift (50 ms, medium shift, 150 ms, and long shift, 
    % 400 ms).
    shift1 = 5; %
    shift2 = 15;
    shift3 = 40;

    % Shift original features values by the given amounts
    F_all_out1 = F_all(shift1+1:end,:);
    F_all_out2 = F_all(shift2+1:end,:);
    F_all_out3 = F_all(shift3+1:end,:);
    
    F_all_in1 = F_all(1:end-shift1,:);
    F_all_in2 = F_all(1:end-shift2,:);
    F_all_in3 = F_all(1:end-shift3,:);
    
    % Split features into sample_length(s) sequences and store in a tensor 
    % called X_in, that is of format [training_sample x time x feats_dim] 
    % (the original no-shift version, and the three shifted versions).

    X_in1 = zeros(floor(size(F_all_in1,1)/seqlen), ...
        seqlen,size(F_train{1},2));

    wloc = 1;
    for k = 1:size(X_in1)
       X_in1(k,:,:) = F_all_in1(wloc:wloc+seqlen-1,:);
       wloc = wloc+seqlen;
    end
    
    X_in2 = zeros(floor(size(F_all_in2,1)/seqlen), ...
        seqlen,size(F_train{1},2));

    wloc = 1;
    for k = 1:size(X_in2)
       X_in2(k,:,:) = F_all_in2(wloc:wloc+seqlen-1,:);
       wloc = wloc+seqlen;
    end
    
    X_in3 = zeros(floor(size(F_all_in3,1)/seqlen), ...
        seqlen,size(F_train{1},2));

    wloc = 1;
    for k = 1:size(X_in3)
       X_in3(k,:,:) = F_all_in3(wloc:wloc+seqlen-1,:);
       wloc = wloc+seqlen;
    end    

    X_out1 = zeros(floor(size(F_all_out1,1)/seqlen),seqlen, ...
                   size(F_train{1},2));

    wloc = 1;
    for k = 1:size(X_out1)
       X_out1(k,:,:) = F_all_out1(wloc:wloc+seqlen-1,:);
       wloc = wloc+seqlen;
    end

    X_out2 = zeros(floor(size(F_all_out2,1)/seqlen),seqlen, ...
                   size(F_train{1},2));

    wloc = 1;
    for k = 1:size(X_out2)
       X_out2(k,:,:) = F_all_out2(wloc:wloc+seqlen-1,:);
       wloc = wloc+seqlen;
    end


    X_out3 = zeros(floor(size(F_all_out3,1)/seqlen),seqlen, ...
                   size(F_train{1},2));

    wloc = 1;
    for k = 1:size(X_out3)
       X_out3(k,:,:) = F_all_out3(wloc:wloc+seqlen-1,:);
       wloc = wloc+seqlen;
    end


    % Save features for modeling.
    
    % Create folder structure for features
    full_feats_path = fullfile(config_feats.features_path, ...
        config_feats.method.folder_name, language{1});
    mkdir(full_feats_path);
    
    save(fullfile(full_feats_path, 'train_in1.mat'), 'X_in1');
    save(fullfile(full_feats_path, 'train_in2.mat'), 'X_in2');
    save(fullfile(full_feats_path, 'train_in3.mat'), 'X_in3');
    save(fullfile(full_feats_path, 'train_out1.mat'), 'X_out1');
    save(fullfile(full_feats_path, 'train_out2.mat'), 'X_out2');
    save(fullfile(full_feats_path, 'train_out3.mat'), 'X_out3');

    % Repeat the process for test data
    for duration=config_feats.durations
        if strcmp(config_feats.method.name, 'mfcc')
            % Extract MFCCs (CMVN but no deltas)
            F_test = getMFCCs( ...
                ZS.(language{1}).test.(['filename_' duration{1}]), ...
                1,'white', 1000,config_feats.window_length, ...
                config_feats.window_shift,1,0);
        elseif strcmp(config_feats.method.name, 'mel')
            F_test = getMelSpectrogram( ...
                ZS.(language{1}).test.(['filename_' duration{1}]), ...
                config_feats.window_length, config_feats.window_shift, ...
                config_feats.method.bands, false);
        elseif strcmp(config_feats.method.name, 'logmel')
            F_test = getMelSpectrogram( ...
                ZS.(language{1}).test.(['filename_' duration{1}]), ...
                config_feats.window_length, config_feats.window_shift, ...
                config_feats.method.bands, true);
        end
        
        % Concatenate into one long matrix
        totlen_test = sum(cellfun(@length,F_test));
        F_test_all = zeros(totlen_test,feature_size);
        % this one keeps track of from which signal and frame the features
        % vectors came from
        F_test_ind = zeros(totlen_test,2); 
        
        wloc = 1;
        for k = 1:length(F_test)
            F_test_all(wloc:wloc+size(F_test{k},1)-1,:) = F_test{k};
            % store signal ID
            [~,fil,~] = fileparts(ZS.(language{1}).test.(['filename_' ...
                duration{1}]){k});
            F_test_ind(wloc:wloc+size(F_test{k},1)-1,1) = str2num(fil); 
            % store frame ID
            F_test_ind(wloc:wloc+size(F_test{k},1)-1,2) = ... 
                1:size(F_test{k},1); 
            wloc = wloc+size(F_test{k},1);
            % introduce reset. -1 id for file as we should ignore those 
            % reset samples.
            if k < length(F_test)
                F_test_all(wloc:wloc+reset_size-1,:) = reset_sample;
                F_test_ind(wloc:wloc+reset_size-1,1) = -1;
                F_test_ind(wloc:wloc+reset_size-1,2) = -1;
                wloc = wloc+reset_size;
            end
        end

        % Fix Nans and Infs
        F_test_all(isnan(F_test_all)) = 0;
        F_test_all(isinf(F_test_all)) = 0;

        % Put into tensor of the same format as training data
        % We need to generate all the test files, then, we need to pad last
        % file if the test set is not mod 0 of the sample size.
        % last value of wloc is used.
        
        if mod(size(F_test_all,1), seqlen) ~= 0
            padding_size = seqlen - mod(size(F_test_all,1), seqlen);
            F_test_all(wloc:wloc+padding_size-1,:) = ...
                zeros(padding_size,feature_size);
            F_test_ind(wloc:wloc+padding_size-1,1) = -1;
            F_test_ind(wloc:wloc+padding_size-1,2) = -1;
        end
        
        X_test_in = zeros(size(F_test_all,1)/seqlen, ...
                          seqlen,size(F_train{1},2));
        X_test_ind = zeros(size(F_test_all,1)/seqlen,seqlen,2);

        wloc = 1;
        for k = 1:size(X_test_in)
           X_test_in(k,:,:) = F_test_all(wloc:wloc+seqlen-1,:);
           X_test_ind(k,:,:) = F_test_ind(wloc:wloc+seqlen-1,:);
           wloc = wloc+seqlen;
        end
        
        save(fullfile(full_feats_path, ['test_', duration{1} 's.mat']), ... 
            'X_test_in','X_test_ind');
    end
  
end
