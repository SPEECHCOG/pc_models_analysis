function createSubmissionTemplateZS2017(submission_name,path_in)

authors = {'Maria Andrea Cruz Blandon','Okko Rasanen'};
affiliation = 'Tampere University';

if nargin <2
    ZSpath = sprintf('/Users/rasaneno/rundata/ZS2020/%s/',submission_name);
else
    ZSpath = [path_in '/' submission_name '/'];
end

if ~exist(ZSpath,'dir')
    mkdir(ZSpath);
    mkdir([ZSpath '/2017/']);
    mkdir([ZSpath '/2017/code/']);
    mkdir([ZSpath '/2017/track1/']);
    mkdir([ZSpath '/2017/track2/']);
else
    fprintf('Proposed submission template/folder structure already exists at %s\nOverwriting will destroy all existing data.\n',ZSpath);
    resp = input('Continue? (y/n)','s');
    if(strcmp(resp,'y'))
        fprintf('overwriting...\n');
        rmdir(ZSpath,'s');
        mkdir(ZSpath);
        mkdir([ZSpath '/2017/']);
    mkdir([ZSpath '/2017/code/']);
    mkdir([ZSpath '/2017/track1/']);
    
    languages = {'english','french','mandarin','LANG1','LANG2'};
    durations = {'1s','10s','120s'};
    for k = 1:length(languages)
        for j = 1:length(durations)
            mkdir([ZSpath sprintf('/2017/track1/%s/%s/',languages{k},durations{j})]);             
        end
    end    
    mkdir([ZSpath '/2017/track2/']);
    else
        fprintf('aborting ...\n');
        return;
    end
end

% Create general metadata file for the submission
fid = fopen([ZSpath '/metadata.yaml'],'w');
fprintf(fid,'author:\n');
fprintf(fid,'\t');
for j = 1:length(authors)-1
    fprintf(fid,[authors{j} ', ']);
end
fprintf(fid,[authors{end}]);        
fprintf(fid,'\n');
fprintf(fid,'affiliation:\n');
fprintf(fid,['\t' affiliation '\n']);
fprintf(fid,'open source:\n');
fprintf(fid,'\ttrue\n');
fclose(fid);

% Create ZS2017 challenge metadata file
fid = fopen([ZSpath '/2017/metadata.yaml'],'w');
fprintf(fid,'system description:\n');
fprintf(fid,'  a brief description of your system, pointing to a paper where possible\n');
fprintf(fid,'hyperparameters:\n');
fprintf(fid,'  values of all the hyperparameters\n');
fprintf(fid,'track1 supervised:\n');
fprintf(fid,'  false\n');
fprintf(fid,'track2 supervised:\n');
fprintf(fid,'  false\n');
fclose(fid);

