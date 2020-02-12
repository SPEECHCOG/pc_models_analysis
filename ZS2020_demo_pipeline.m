% Demo pipeline

ZS = loadZSData2017('mandarin','/Users/rasaneno/speechdb/zerospeech2020/2017/'); % Load English data

% Get MFCCs for test data

ZS.mandarin.test.features_10 = getMFCCs(ZS.mandarin.test.filename_10,1,'white',1000,0.025,0.01,1,1);

% Create timestamps for the features (if do not exist already)

ZS = synthesizeTimeStamps(ZS,0.01);

% Create template structure
createSubmissionTemplateZS2017('test_submission','/Users/rasaneno/rundata/ZS2020/')

% Add calculated features to the package
addTrack1FeaturesToSubmission(ZS,'test_submission','/Users/rasaneno/rundata/ZS2020/');







