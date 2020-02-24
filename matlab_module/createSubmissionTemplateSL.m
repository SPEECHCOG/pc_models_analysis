function createSubmissionTemplateSL()
%%%
% This creates the directory structure and metadata files for the 
% submission.
% SL stands for Symbolic Links, as they will be used for placing the 
% predictions of the model in the submission folder.
%
% predictions/model<n>/features_type/ -> track1 
% To evaluate the predictions, predictions/model<n>/features_type/ folder
% should have at least one language with at least one duration. Then,
% the evaluation command should be performed only on those existing.
%%%

% read configuration
configuration = readConfiguration();
config_submission = configuration.submission;

authors = {'Maria Andrea Cruz Blandon','Okko Rasanen'};
affiliation = 'Tampere University';
code_repository = 'https://github.com/SPEECHCOG/ZS2020';

ZSpath = fullfile(config_submission.path, config_submission.name);


if ~exist(ZSpath,'dir')
    mkdir(ZSpath);
    mkdir(fullfile(ZSpath, '2017'));
    mkdir(fullfile(ZSpath, '2017/code'));
else
    fprintf(['Proposed submission template/folder structure already ' ...
            'exists at %s\nOverwriting will destroy all existing ' ...
            'data.\n'],ZSpath);
    resp = input('Continue? (y/n)','s');
    if(strcmp(resp,'y'))
        fprintf('overwriting...\n');
        
        rmdir(ZSpath,'s');
        mkdir(ZSpath);
        mkdir(fullfile(ZSpath, '2017'));
        mkdir(fullfile(ZSpath, '2017/code'));
        % link to code repository for uploaded model
        fid = fopen([ZSpath '/2017/code/code.md'],'w');
        fprintf(fid,code_repository);
        fclose(fid);
        if exist('./*.DS_Store', 'file')
            system(['find /Users/rasaneno/rundata/ZS2020/ -name ' ...
                   '".DS_Store" -delete']);
        end
    else
        fprintf('aborting ...\n');
        return;
    end
end

% Create general metadata file for the submission
fid = fopen(fullfile(ZSpath, 'metadata.yaml'),'w');
fprintf(fid,'author: ');
fprintf(fid,'\t');
for j = 1:length(authors)-1
    fprintf(fid,[authors{j} ', ']);
end
fprintf(fid,[authors{end}]);        
fprintf(fid,'\n');
fprintf(fid,'affiliation: ');
fprintf(fid,['\t' affiliation '\n']);
fprintf(fid,'open source: ');
fprintf(fid,'\ttrue\n');
fclose(fid);

% Create ZS2017 challenge metadata file
fid = fopen(fullfile(ZSpath, '2017/metadata.yaml'),'w');
fprintf(fid,'system description: ');
fprintf(fid,['  a brief description of your system, pointing to a ' ... 
             'paper where possible\n']);
fprintf(fid,'hyperparameters: ');
fprintf(fid,'  values of all the hyperparameters\n');
fprintf(fid,'track1 supervised: ');
fprintf(fid,'  false\n');
fprintf(fid,'track2 supervised: ');
fprintf(fid,'  false\n');
fclose(fid);
