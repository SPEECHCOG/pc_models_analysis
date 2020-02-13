% Demo pipeline for ZS2020

clear all

curdir = fileparts(which('ZS2020_demo_pipeline.m'));
addpath([curdir '/MFCC_extract/']);
addpath([curdir '/aux/']);
addpath([curdir '/misc/']);

% Paths to change:

audio_location = '/Users/rasaneno/speechdb/zerospeech2020/2017/'; % Where are ZS2017 audio located?
submission_path = '/Users/rasaneno/rundata/ZS2020/';     % Where to store submission files?
result_location = '/Users/rasaneno/rundata/ZS2020/eval/'; % Where to write results?


language = 'french';

submission_name = 'test_submission'; % Name your submission here

%% Start processing

ZS = loadZSData2017(language,audio_location); % Load Mandarin data

% Get MFCCs for test data with 10s chunks

ZS.(language).test.features_10 = getMFCCs(ZS.(language).test.filename_10,1,'white',1000,0.025,0.01,1,0);

% Create timestamps for the features with the same frame shift

ZS = synthesizeTimeStamps(ZS,0.01);

% Create template submission structure
createSubmissionTemplateZS2017(submission_name,submission_path)

% Add calculated features to the template
addTrack1FeaturesToSubmission(ZS,submission_name,submission_path);

% Evaluate 
if(~exist(result_location,'dir'))
    mkdir(result_location);
end

% Run ZS2020 evaluation toolkit
system(sprintf('find %s -name ".DS_Store" -delete',submission_path)); % Remove DS_Store files on OSX platform
ss = sprintf('source ~/.bash_profile;conda activate zerospeech2020;zerospeech2020-evaluate  -j 10 %s/%s/ %s 2017 -l %s track1 -dr 10s %s/ABXTasks/',submission_path,submission_name,result_location,lower(language),audio_location);
system(ss);

% Print results
type(sprintf('%s/mandarin_10s_abx.txt',result_location));





