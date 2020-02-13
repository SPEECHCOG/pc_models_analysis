
curdir = fileparts(which('ZS2020_demo_pipeline.m'));
addpath([curdir '/MFCC_extract/']);
addpath([curdir '/aux/']);
addpath([curdir '/misc/']);

% Demo pipeline

ZS = loadZSData2017('mandarin','/Users/rasaneno/speechdb/zerospeech2020/2017/'); % Load Mandarin data

% Get MFCCs for test data with 10s chunks

ZS.mandarin.test.features_10 = getMFCCs(ZS.mandarin.test.filename_10,1,'white',1000,0.025,0.01,1,1);

% Create timestamps for the features with the same frame shift

ZS = synthesizeTimeStamps(ZS,0.01);

% Create template submission structure
createSubmissionTemplateZS2017('test_submission','/Users/rasaneno/rundata/ZS2020/')

% Add calculated features to the template
addTrack1FeaturesToSubmission(ZS,'test_submission','/Users/rasaneno/rundata/ZS2020/');







