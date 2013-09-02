function [fitness, out] = filterAndSumModelFASTGPU(param, p)

[popSize len] = size(param);
out = [];

tmpParamG = gpuArray(reshape(param,popSize,len/p.nFilt,[]));
featuresG = gpuArray(zeros(p.nFilt, p.bee.stis, p.popSize));
chunkIdx{1} = 1:p.popSize;
% if ~isfield(p,'chunk') || p.chunk==1
%    nChunks = floor(popSize/100);
%    chunkSize = ceil(popSize/nChunks);
%    chunks = reshape(1:(nChunks-1)*chunkSize,chunkSize,[]);
%    chunkIdx = cell(chunks,1);
%    for n = 1:nChunks-1
%       chunkIdx{n} = chunks(:,n);
%    end
%    chunkIdx{n+1} = max(chunks(:))+1:popSize;
% end

for n = 1:length(chunkIdx)
   for f = 1:p.nFilt
      tmpG = p.bee.SSrawG*tmpParamG(chunkIdx{n},1:p.nComp,f)';
      tmpG = bsxfun(@minus, tmpG, 6*(1+tmpParamG(chunkIdx{n},p.nComp + 2,f))'+eps);
      tmpG = bsxfun(@times, tmpG, -16*(1+tmpParamG(chunkIdx{n},p.nComp + 1,f))');
      tmpG = bsxfun(@times,1./(1+exp(tmpG)), p.bee.nanMaskG);
      featuresG(f,:,chunkIdx{n}) = reshape(bsxfun(@times, sum(reshape(tmpG,p.bee.maxStimLen,p.bee.stis, length(chunkIdx{n}))),1./p.bee.stimLenG), p.bee.stis, length(chunkIdx{n}));
   end
end
features = gather(featuresG);
X = [ones(1,p.bee.stis, p.popSize);p.givenFeatures;features];
%X = x2fx([p.givenFeatures; features]', 'purequadratic');
betahat = zeros(p.popSize, size(X,1));
y = zeros(p.popSize, p.bee.stis);
for ind = 1:p.popSize
   betahat(ind,:) = X(:,:,ind)'\p.bee.resp';
   y(ind,:) = X(:,:,ind)'*betahat(ind,:)';
end
fitness = nanmean(bsxfun(@minus, y, p.bee.resp).^2,2);
fitness = 1-fitness'./(nanvar(p.bee.resp)+eps);
fitness(isnan(fitness)) = 0;