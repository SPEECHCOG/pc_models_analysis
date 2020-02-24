function writeResultOutput(filename,signal_names,times,classes)


fid = fopen(filename,'w+');

tot_classes = max(classes);

for curclass = 1:tot_classes
    
    a = find(classes == curclass);
    if(~isempty(a))
        fprintf(fid,sprintf('Class %d\n',curclass));
        for j = 1:length(a)
            if(length(signal_names{a(j)}) > 7)
                fprintf(fid,sprintf('%s %0.3f %0.3f\n',signal_names{a(j)},times(a(j),1),times(a(j),2)));
            else
                fprintf(fid,sprintf('%s %0.3f %0.3f\n',signal_names{a(j)},times(a(j),1),times(a(j),2)));
            end
        end
        fprintf(fid,'\n');
    end   
    
end
fprintf('Wrote total %d unique classes with %d tokens.\n',length(unique(classes)),length(classes));
fclose(fid);






