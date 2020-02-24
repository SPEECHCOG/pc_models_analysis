
language = 'english';

datadir = sprintf('/Users/rasaneno/speechdb/zerospeech2017/zerospeech2017/data/train/%s/',language);

a = dir([datadir '*.wav']);


filenames = cell(length(a),1);

for k = 1:length(a)
    filenames{k} = [a(k).folder '/' a(k).name];
    
end

F = haeMelPiirteet(filenames,0.025,0.01,16000,0);


save(sprintf('/Users/rasaneno/rundata/ZS2020_tmp/logmel_%s.mat',language),'F','filenames');

% Concatenate data into one large matrix

totlen = sum(cellfun(@length,F));

F_all = zeros(totlen,size(F{1},2));
wloc = 1;

for k = 1:length(F)
    F_all(wloc:wloc+size(F{k},1)-1,:) = F{k};
    wloc = wloc+size(F{k},1);
end

maxlen = 2e9/(24*8)-1;

F_all = F_all(1:maxlen-1,:);



F_all(isnan(F_all)) = 0;
F_all(isinf(F_all)) = 0;


meme = nanmean(F_all);
devi = nanstd(F_all);

F_all = F_all-repmat(meme,size(F_all,1),1);
F_all = F_all./repmat(devi,size(F_all,1),1);



shift1 = 5;
shift2 = 15;
shift3 = 40;

F_all_out1 = circshift(F_all,-shift1);
F_all_out2 = circshift(F_all,-shift2);
F_all_out3 = circshift(F_all,-shift3);

% Split into samples

seqlen = 200;

X_in = zeros(round(size(F_all,1)/seqlen)-1,seqlen,size(F{1},2));

wloc = 1;
for k = 1:size(X_in)
   X_in(k,:,:) = F_all(wloc:wloc+seqlen-1,:);
   wloc = wloc+seqlen;
end
   
X_out1 = zeros(round(size(F_all,1)/seqlen)-1,seqlen,size(F{1},2));

wloc = 1;
for k = 1:size(X_out1)
   X_out1(k,:,:) = F_all_out1(wloc:wloc+seqlen-1,:);
   wloc = wloc+seqlen;
end

X_out2 = zeros(round(size(F_all,1)/seqlen)-1,seqlen,size(F{1},2));

wloc = 1;
for k = 1:size(X_out2)
   X_out2(k,:,:) = F_all_out2(wloc:wloc+seqlen-1,:);
   wloc = wloc+seqlen;
end


X_out3 = zeros(round(size(F_all,1)/seqlen)-1,seqlen,size(F{1},2));

wloc = 1;
for k = 1:size(X_out3)
   X_out3(k,:,:) = F_all_out3(wloc:wloc+seqlen-1,:);
   wloc = wloc+seqlen;
end




save(sprintf('/Users/rasaneno/rundata/ZS2020_tmp/logmel_concat_%s_in.mat',language),'X_in');
save(sprintf('/Users/rasaneno/rundata/ZS2020_tmp/logmel_concat_%s_out1.mat',language),'X_out1');
save(sprintf('/Users/rasaneno/rundata/ZS2020_tmp/logmel_concat_%s_out2.mat',language),'X_out2');
save(sprintf('/Users/rasaneno/rundata/ZS2020_tmp/logmel_concat_%s_out3.mat',language),'X_out3');