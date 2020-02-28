function configuration = readConfiguration()
%%%
% @author María Andrea Cruz Blandón
% @date 24.02.2020
% This scripts reads a configuration file in JSON format. It returns the
% read configuration in the cell variable configuration.
%%%

% use default configuration
default = false;

% read json file
try
    configuration = jsondecode(fileread('config.json'));
catch
    warning(['There was a problem reading the configuration file ' ...
             '<./config.json>. Default configuratoin will be used ' ...
             'instead.']);
    default = true;
    % create configuration
    configuration.submission.path = './';
    configuration.submission.name = 'submission';
    configuration.feature_extraction.audio_path = '../dataset/2017/';
    configuration.feature_extraction.features_path = '../features/';
    configuration.feature_extraction.languages = {'mandarin'};
    configuration.feature_extraction.durations = {'10'};
    configuration.feature_extraction.window_length = 0.025;
    configuration.feature_extraction.window_shift = 0.01;
    configuration.feature_extraction.sample_length = 2;
    % default feature extraction method: MFCC
    configuration.feature_extraction.method.name = 'mfcc';
    configuration.feature_extraction.method.delta = true;
    configuration.feature_extraction.method.delta_delta = true;
    configuration.feature_extraction.method.cmvn = true;
    configuration.feature_extraction.method.folder_name = 'mfcc';
    configuration.feature_extraction.method.bands = 0;
end

% verify JSON structure, avoid if default configuration is used.

if ~default
    % sumission structure
    try    
        if ~ischar(configuration.submission.path)
            warning(['submission.path should be string. ' ... 
                     'Default value will be used instead: ./']);
            configuration.submission.path = './';
        end
    catch
        warning(['submission.path does not exist. ' ... 
                 'Default value will be used instead: ./']);
        configuration.submission.path = './';
    end

    try    
        if ~ischar(configuration.submission.name)
            warning(['submission.path should be string. ' ... 
                     'Default value will be used instead: submission']);
            configuration.submission.name = 'submission';
        end
    catch
        warning(['submission.name does not exist. ' ... 
                 'Default value will be used instead: submission']);
        configuration.submission.name = 'submission';
    end

    % feature_extraction structure

    try    
        if ~ischar(configuration.feature_extraction.audio_path)
            warning(['feature_extraction.audio_path should be string. ' ... 
                     'Default value will be used instead: ' ...
                     '../dataset/2017/']);
            configuration.feature_extraction.audio_path = ...
                '../dataset/2017/';
        end
    catch
        warning(['feature_extraction.audio_pathe does not exist. ' ... 
                 'Default value will be used instead: ../dataset/2017/']);
        configuration.feature_extraction.audio_path = ...
            '../dataset/2017/';
    end

    try    
        if ~ischar(configuration.feature_extraction.features_path)
            warning(['feature_extraction.features_path should be ' ...
                     'string. Default value will be used instead: ' ...
                     '../features/']);
            configuration.feature_extraction.features_path = ... 
                '../features/';
        end
    catch
        warning(['feature_extraction.features_path does not exist. ' ... 
                 'Default value will be used instead: ../features/']);
        configuration.feature_extraction.features_path = '../features/';
    end

    try    
        if ~iscellstr(configuration.feature_extraction.languages)
            warning(['feature_extraction.languages should be an ' ... 
                     'array of strings. Default value will be used ' ...
                     'instead: {''mandarin''}']);
            configuration.feature_extraction.languages = {'mandarin'};
        end
        % If the array has been mapped as a column vector change it to 
        % row vector
        if size(configuration.feature_extraction.languages,2) == 1
            configuration.feature_extraction.languages = ...
                configuration.feature_extraction.languages';
        end
    catch
        warning(['feature_extraction.languages does not exist. ' ... 
                 'Default value will be used instead: {''mandarin''}']);
        configuration.feature_extraction.languages = {'mandarin'};
    end
    
    try    
        if ~iscellstr(configuration.feature_extraction.durations)
            warning(['feature_extraction.durations should be an ' ... 
                     'array of strings. Default value will be used ' ...
                     'instead: {''10''}']);
            configuration.feature_extraction.durations = {'10'};
        end
        % If the array has been mapped as a column vector change it to 
        % row vector
        if size(configuration.feature_extraction.durations,2) == 1
            configuration.feature_extraction.durations = ...
                configuration.feature_extraction.durations';
        end
    catch
        warning(['feature_extraction.durations does not exist. ' ... 
                 'Default value will be used instead: {''10''}']);
        configuration.feature_extraction.durations = {'10'};
    end

    try    
        if ~isfloat(configuration.feature_extraction.window_length)
            warning(['feature_extraction.window_length should be ' ... 
                     'float (length in seconds). ' ... 
                     'Default value will be used instead: 0.025']);
            configuration.feature_extraction.window_length = 0.025;
        end
    catch
        warning(['feature_extraction.window_length does not exist. ' ... 
                 'Default value will be used instead: 0.025']);
        configuration.feature_extraction.window_length = 0.025;
    end

    try    
        if ~isfloat(configuration.feature_extraction.window_shift)
            warning(['feature_extraction.window_shift should be float ' ...
                     '(length in seconds). ' ... 
                     'Default value will be used instead: 0.01']);
            configuration.feature_extraction.window_shift = 0.01;
        end
    catch
        warning(['feature_extraction.window_shift does not exist. ' ... 
                 'Default value will be used instead: 0.01']);
        configuration.feature_extraction.window_shift = 0.01;
    end
    
    try    
        if ~isfloat(configuration.feature_extraction.sample_length)
            warning(['feature_extraction.sample_length should be float' ...
                     ' (length in seconds). ' ... 
                     'Default value will be used instead: 2']);
            configuration.feature_extraction.sample_length = 2;
        end
    catch
        warning(['feature_extraction.sample_length does not exist. ' ... 
                 'Default value will be used instead: 2']);
        configuration.feature_extraction.sample_length = 2;
    end
    
    try    
        if ~ischar(configuration.feature_extraction.method.name)
            warning(['feature_extraction.method.name should be ' ...
                     'string. Default value will be used instead: ' ...
                     'mfcc']);
            configuration.feature_extraction.method.name = ... 
                'mfcc';
        end
    catch
        warning(['feature_extraction.method.name does not exist. ' ... 
                 'Default value will be used instead: mfcc']);
        configuration.feature_extraction.method.name = 'mfcc';
    end
    
    try    
        if ~ischar(configuration.feature_extraction.method.folder_name)
            warning(['feature_extraction.method.folder_name should be ' ...
                     'string. Method name will be used instead']);
            configuration.feature_extraction.method.folder_name = ... 
                 configuration.feature_extraction.method.name;
        end
    catch
        warning(['feature_extraction.method.folder_name does not ' ...
                 'exist. Method name will be used instead']);
        configuration.feature_extraction.method.folder_name = ...
            configuration.feature_extraction.method.name;
    end
    
    try    
        if strcmp(configuration.feature_extraction.method.name, 'mel') ...
                || strcmp(configuration.feature_extraction.method.name, ...
                'logmel')
            if mod(configuration.feature_extraction.method.bands,1) ~= 0
                warning(['feature_extraction.method.bands should be ' ...
                         'integer. 80 bands will be used instead']);
                configuration.feature_extraction.method.bands = 80;
            end
        end
    catch
        warning(['feature_extraction.method.bands does not ' ...
                 'exist. 80 bands will be used instead']);
        configuration.feature_extraction.method.bands = 80;
    end
end
end
