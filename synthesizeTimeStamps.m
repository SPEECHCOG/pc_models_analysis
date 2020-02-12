function ZS = synthesizeTimeStamps(ZS,ws)
% function ZS = synthesizeTimeStamps(ZS,ws)
% 
% Creates equal spacing timestamps for all extracted features with frame
% shift ws.

languages = fields(ZS);

for langiter = 1:length(languages)
   if(isfield(ZS.(languages{1}).train,'features'))
       ff = ZS.(languages{1}).train.features;
       tt = createTimeStamps(ff,ws);
       ZS.(languages{1}).train.features_t = tt;
   end
   
   if(isfield(ZS.(languages{1}).test,'features_1'))
       ff = ZS.(languages{1}).test.features_1;
       tt = createTimeStamps(ff,ws);
       ZS.(languages{1}).test.features_1_t = tt;
   end
   
   if(isfield(ZS.(languages{1}).test,'features_10'))
       ff = ZS.(languages{1}).test.features_10;
       tt = createTimeStamps(ff,ws);
       ZS.(languages{1}).test.features_10_t = tt;
   end
   
   if(isfield(ZS.(languages{1}).test,'features_120'))
       ff = ZS.(languages{1}).test.features_120;
       tt = createTimeStamps(ff,ws);
       ZS.(languages{1}).test.features_120_t = tt;
   end
   
    
    
end


function tt = createTimeStamps(ff,ws)

tt = cell(length(ff),1);
for k = 1:length(ff)
    len = size(ff{k},1);
    
    t = ws/2:ws:len*ws-ws/2;
    
    tt{k} = t;
    
end