function ZS = synthesizeTimeStamps(ZS,ws,offset)
% function ZS = synthesizeTimeStamps(ZS,ws,offset)
% 
% Creates equal spacing timestamps for all extracted features with frame
% shift ws.

if nargin <3
    offset = 0;
end

languages = fields(ZS);

for langiter = 1:length(languages)
   if(isfield(ZS.(languages{1}).train,'features'))
       ff = ZS.(languages{1}).train.features;
       tt = createTimeStamps(ff,ws,offset);
       
       for k = 1:length(tt)
           tmp = tt{k} < 0;
           ZS.(languages{1}).train.features{k}(tmp,:) = [];
           tt{k}(tmp) = [];           
       end
       
       ZS.(languages{1}).train.features_t = tt;
   end
   
   if(isfield(ZS.(languages{1}).test,'features_1'))
       ff = ZS.(languages{1}).test.features_1;
       tt = createTimeStamps(ff,ws,offset);
       
           
       for k = 1:length(tt)
           tmp = tt{k} < 0;
           tmp = tmp+tt{k} > 1;                      
           ZS.(languages{1}).test.features_1{k}(tmp,:) = [];
           tt{k}(tmp) = [];           
       end
       
       ZS.(languages{1}).test.features_1_t = tt;
       
       
   end
   
   if(isfield(ZS.(languages{1}).test,'features_10'))
       ff = ZS.(languages{1}).test.features_10;
       tt = createTimeStamps(ff,ws,offset);
              
       for k = 1:length(tt)
           tmp = tt{k} < 0;          
           tmp = tmp+tt{k} > 10;                      
           ZS.(languages{1}).test.features_10{k}(tmp,:) = [];
           tt{k}(tmp) = [];           
       end
       
       ZS.(languages{1}).test.features_10_t = tt;
       
   end
   
   if(isfield(ZS.(languages{1}).test,'features_120'))
       ff = ZS.(languages{1}).test.features_120;
       tt = createTimeStamps(ff,ws,offset);       
       
       for k = 1:length(tt)
           
           
           tmp = tt{k} < 0;
           tmp = tmp+tt{k} > 120;                      
           ZS.(languages{1}).test.features_120{k}(tmp,:) = [];
           tt{k}(tmp) = [];           
       end
       
       ZS.(languages{1}).test.features_120_t = tt;       
       
   end
   
    
    
end


function tt = createTimeStamps(ff,ws,offset)

tt = cell(length(ff),1);
for k = 1:length(ff)
    len = size(ff{k},1);
    
    t = ws/2:ws:len*ws-ws/2;
    
    tt{k} = t-offset;
    
end