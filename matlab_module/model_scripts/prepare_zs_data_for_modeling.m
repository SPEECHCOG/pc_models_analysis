
clear all

% Add paths
curdir = fileparts(which('ZS2020_demo_pipeline.m'));
addpath([curdir '/MFCC_extract/']);
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
    end
  
    % total number of frames
    totlen = sum(cellfun(@length,F_train)); 

    % Concatenate all feature data into one large matrix  
    F_all = zeros(totlen,size(F_train{1},2));
    wloc = 1;
    for k = 1:length(F_train)
        F_all(wloc:wloc+size(F_train{k},1)-1,:) = F_train{k};
        wloc = wloc+size(F_train{k},1);
    end

    % F_all contains now all features frames of the training set
    % Make sure there are no NaNs or Infs
    F_all(isnan(F_all)) = 0;
    F_all(isinf(F_all)) = 0;

    % Compute mean and SD across all frames 
    meme = nanmean(F_all); 
    devi = nanstd(F_all);

    % Mean and variance normalize all time steps  
    F_all = F_all-repmat(meme,size(F_all,1),1);
    F_all = F_all./repmat(devi,size(F_all,1),1);


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
    F_all_out1 = circshift(F_all,-shift1);
    F_all_out2 = circshift(F_all,-shift2);
    F_all_out3 = circshift(F_all,-shift3);


    % Split features into sample_length(s) sequences and store in a tensor 
    % called X_in, that is of format [training_sample x time x feats_dim] 
    % (the original no-shift version, and the three shifted versions).

    seqlen = round(config_feats.sample_length / config_feats.window_shift);

    X_in = zeros(round(size(F_all,1)/seqlen)-1,seqlen,size(F_train{1},2));

    wloc = 1;
    for k = 1:size(X_in)
       X_in(k,:,:) = F_all(wloc:wloc+seqlen-1,:);
       wloc = wloc+seqlen;
    end

    X_out1 = zeros(round(size(F_all,1)/seqlen)-1,seqlen, ...
                   size(F_train{1},2));

    wloc = 1;
    for k = 1:size(X_out1)
       X_out1(k,:,:) = F_all_out1(wloc:wloc+seqlen-1,:);
       wloc = wloc+seqlen;
    end

    X_out2 = zeros(round(size(F_all,1)/seqlen)-1,seqlen, ...
                   size(F_train{1},2));

    wloc = 1;
    for k = 1:size(X_out2)
       X_out2(k,:,:) = F_all_out2(wloc:wloc+seqlen-1,:);
       wloc = wloc+seqlen;
    end


    X_out3 = zeros(round(size(F_all,1)/seqlen)-1,seqlen, ...
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
    
    save(fullfile(full_feats_path, 'train_in.mat'), 'X_in');
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
        end
        
        % Concatenate into one long matrix
        F_test_all = zeros(totlen,size(F_test{1},2));
        % this one keeps track of from which signal and frame the features
        % vectors came from
        F_test_ind = zeros(totlen,2); 
        
        wloc = 1;
        for k = 1:length(F_test)
            F_test_all(wloc:wloc+size(F_test{k},1)-1,:) = F_test{k};
            % store signal ID
            F_test_ind(wloc:wloc+size(F_test{k},1)-1,1) = k; 
            % store frame ID
            F_test_ind(wloc:wloc+size(F_test{k},1)-1,2) = ... 
                1:size(F_test{k},1); 
            wloc = wloc+size(F_test{k},1);
        end

        % Fix Nans and Infs
        F_test_all(isnan(F_test_all)) = 0;
        F_test_all(isinf(F_test_all)) = 0;

        % Mean and variance normalize with the _same_ means and variances 
        % as the training data
        F_test_all = F_test_all-repmat(meme,size(F_test_all,1),1);
        F_test_all = F_test_all./repmat(devi,size(F_test_all,1),1);

        % Put into tensor of the same format as training data
        X_test_in = zeros(round(size(F_test_all,1)/seqlen)-1, ...
                          seqlen,size(F_train{1},2));
        X_test_ind = zeros(round(size(F_test_all,1)/seqlen)-1,seqlen,2);

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
