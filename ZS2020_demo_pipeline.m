
curdir = fileparts(which('ZS2020_demo_pipeline.m'));
addpath([curdir '/MFCC_extract/']);
addpath([curdir '/aux/']);
addpath([curdir '/misc/']);


audio_location = '/Users/rasaneno/speechdb/zerospeech2020/2017/';
submission_path = '/Users/rasaneno/rundata/ZS2020/';
result_location = '/Users/rasaneno/rundata/ZS2020/eval/';

% Demo pipeline

ZS = loadZSData2017('mandarin',audio_location); % Load Mandarin data

% Get MFCCs for test data with 10s chunks

ZS.mandarin.test.features_10 = getMFCCs(ZS.mandarin.test.filename_10,1,'white',1000,0.025,0.01,1,1);

% Create timestamps for the features with the same frame shift

ZS = synthesizeTimeStamps(ZS,0.01);

% Create template submission structure
createSubmissionTemplateZS2017('test_submission',submission_path)

% Add calculated features to the template
addTrack1FeaturesToSubmission(ZS,'test_submission',submission_path);

% Evaluate 
if(~exist(result_location,'dir'))
    mkdir(result_location);
end
ss = sprintf('source ~/.bash_profile;conda activate zerospeech2020;zerospeech2020-evaluate  -j 10 %s/%s/ %s 2017 -l mandarin track1 -dr 10s %s/ABX_tasks/',submission_path,submission_name,result_location,audio_location);

system(ss);






