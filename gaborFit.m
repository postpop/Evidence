function [fitness, out] = gaborFit(param, p)

[popSize len] = size(param);
out = [];

tmpParam = reshape(param,popSize,len/p.nFilt,[]);
features = zeros(p.nFilt, p.bee.stis, popSize);
chunkIdx{1} = 1:popSize;
if ~isfield(p,'chunk') || p.chunk==1
   nChunks = floor(popSize/100);
   chunkSize = ceil(popSize/nChunks);
   chunks = reshape(1:(nChunks-1)*chunkSize,chunkSize,[]);
   chunkIdx = cell(chunks,1);
   for n = 1:nChunks-1
      chunkIdx{n} = chunks(:,n);
   end
   chunkIdx{n+1} = max(chunks(:))+1:popSize;
end

dur = 256;
T = (0:dur-1);%ms

for n = 1:length(chunkIdx)
   for f = 1:p.nFilt
      %% build gabor filter
      sig = 3+50*(1+tmpParam(chunkIdx{n},1,f));% ms - duration of Gabor env, [3 103]
      frq = 2+50*(1+tmpParam(chunkIdx{n},2,f));% Hz - frequency of Gabor "carrier", [2 102]]
      pha = 2*pi*(1+tmpParam(chunkIdx{n},3,f));% phase of Gabor "carrier", [0, 2pi]
      off = tmpParam(chunkIdx{n},4,f)/2;% offset of the filter[-.5, .5]
      nlThres  = 16*(1+tmpParam(chunkIdx{n},5,f));%[0 32]
      nlSat    = 6*(1+tmpParam(chunkIdx{n},6,f)+eps);%[0 12]
      featThres   = 16*(1+tmpParam(chunkIdx{n},7,f));%[0 32]
      featSat     = 6*(1+tmpParam(chunkIdx{n},8,f)+eps);%[0 12]
      for idx = 1:length(chunkIdx{n})
         ind = chunkIdx{n}(idx);
         env = gausswin(dur,sig(ind));
         car = sin(2*pi*frq(ind)*T/1000+pha(ind))';
         filt(idx,:) = off(ind)+env.*car;
      end
%          %          tmp = buffer(p.bee.stim{sti}, dur(idx)+1, dur(idx));
%          %          filterOut(sti,1:length(tmp)) = tmp(1:end-1,:)'*filt;
%          filterOut = p.bee.SSraw(:,1:length(filt))*filt;
%          nlOut = filterOut - nlThres(ind);
%          nlOut(nlOut(ind)<0) = 0;
%          nlOut(nlOut>nlSat(ind)) = nlSat(ind);
%          nlOut= reshape(nlOut,p.bee.maxStimLen,p.bee.stis);
%          feat = nansum(nlOut)./p.bee.stimLen;
%          feat = feat - featThres(ind);
%          feat(feat<0) = 0;
%          feat(feat>featSat(ind)) = featSat(ind);
%          features(f,:,chunkIdx{n}(ind)) = feat;
%       end
      tmp = p.bee.SSraw*filt';
      tmp = bsxfun(@minus, tmp, nlSat');
      tmp = bsxfun(@times, tmp, nlThres');
      tmp = 1./(1+exp(tmp));   
      tmp = bsxfun(@times,tmp, p.bee.nanMask);
      tmp = reshape(tmp,p.bee.maxStimLen,p.bee.stis, length(chunkIdx{n}));   
      tmpFeat = reshape(bsxfun(@times, sum(tmp),1./p.bee.stimLen), p.bee.stis, length(chunkIdx{n}));   
      tmpFeat = bsxfun(@minus, tmpFeat, featSat');
      tmpFeat = bsxfun(@times, tmpFeat, featThres');
      tmpFeat = 1./(1+exp(tmpFeat));   
      features(f,:,chunkIdx{n}) = tmpFeat;
      
   end
end
X = [ones(1,p.bee.stis, popSize);p.givenFeatures;features];
%X = x2fx([p.givenFeatures; features]', 'purequadratic');
betahat = zeros(popSize, size(X,1));
y = zeros(popSize, p.bee.stis);
for ind = 1:popSize
   betahat(ind,:) = X(:,:,ind)'\p.bee.resp';
   y(ind,:) = X(:,:,ind)'*betahat(ind,:)';
end
fitness = nanmean(bsxfun(@minus, y, p.bee.resp).^2,2);
fitness = 1-fitness'./(nanvar(p.bee.resp)+eps);
fitness(isnan(fitness)) = 0;