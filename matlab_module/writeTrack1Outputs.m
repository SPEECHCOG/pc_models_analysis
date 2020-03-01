function writeTrack1Outputs(filenames,features,timestamps,filepath)

% Version 1 (slow)

% for k = 1:length(filenames)
%     [~,fil,~] = fileparts(filenames{k});
%     fid = fopen([filepath '/' fil '.txt'],'w');
%     for j = 1:size(features{k},1)
%         s = [sprintf('%0.4f\t',timestamps{k}(j)) num2str(features{k}(j,:)) '\n'];
%         fprintf(fid,s);
%     end
%     fclose(fid);       
% end

% Version 2 (a bit faster)
for k = 1:length(filenames)
    [~,fil,~] = fileparts(filenames{k});
    fname = [filepath '/' fil '.txt'];
    
    % Transpose timestaps if needed
    tt = timestamps{k};
    if(size(tt,2) > size(tt,1))
        tt = tt';
    end
    
    % Combine timestamps with features
    M = [tt features{k}];
    
    %csvwrite(fname,M);
    dlmwrite(fname,M,'delimiter',' ');
    
end



