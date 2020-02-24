function addTrack1FeaturesToSubmission(ZS,submission_name,path_in)



ZSpath = [path_in '/' submission_name '/'];

languages = fields(ZS);

for langiter = 1:length(languages)
    
    if(isfield(ZS.(languages{langiter}).test,'features_1') && isfield(ZS.(languages{langiter}).test,'features_1_t'))
        filepath = [ZSpath sprintf('/2017/track1/%s/1s/',languages{langiter})];
        fprintf('Found features and timestamps for %s with 1s files. Writing to .txt files... (this might take a while)\n',languages{langiter});
        writeTrack1Outputs(ZS.(languages{langiter}).test.filename_1,ZS.(languages{langiter}).test.features_1,ZS.(languages{langiter}).test.features_1_t,filepath);
    end
    
    if(isfield(ZS.(languages{langiter}).test,'features_10') && isfield(ZS.(languages{langiter}).test,'features_10_t'))
        filepath = [ZSpath sprintf('/2017/track1/%s/10s/',languages{langiter})];
        fprintf('Found features and timestamps for %s with 10s files. Writing to .txt files... (this might take a while)\n',languages{langiter});
        writeTrack1Outputs(ZS.(languages{langiter}).test.filename_10,ZS.(languages{langiter}).test.features_10,ZS.(languages{langiter}).test.features_10_t,filepath);
    end
    
    if(isfield(ZS.(languages{langiter}).test,'features_120') && isfield(ZS.(languages{langiter}).test,'features_120_t'))
        filepath = [ZSpath sprintf('/2017/track1/%s/120s/',languages{langiter})];
        fprintf('Found features and timestamps for %s with 120s files. Writing to .txt files... (this might take a while)\n',languages{langiter});
        writeTrack1Outputs(ZS.(languages{langiter}).test.filename_120,ZS.(languages{langiter}).test.features_120,ZS.(languages{langiter}).test.features_120_t,filepath);
        
    end
    
end

    


