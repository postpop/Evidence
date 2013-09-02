function [fitness, out] = filterAndSumModelFAST2(param, p)

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

for n = 1:length(chunkIdx)
   for f = 1:p.nFilt
      %% pedestrian's version
      tmp = p.bee.SSraw*tmpParam(chunkIdx{n},1:p.nComp,f)';
      tmp = bsxfun(@minus, tmp, 2*6*(1+tmpParam(chunkIdx{n},p.nComp + 2,f))'+eps);
      tmp = bsxfun(@times, tmp, 2*-16*(1+tmpParam(chunkIdx{n},p.nComp + 1,f))');
      tmp = 1./(1+exp(tmp));
      tmp = bsxfun(@times,tmp, p.bee.nanMask);
      tmp = reshape(tmp,p.bee.maxStimLen,p.bee.stis, length(chunkIdx{n}));
      features(f,:,chunkIdx{n}) = reshape(bsxfun(@times, sum(tmp),1./p.bee.stimLen), p.bee.stis, length(chunkIdx{n}));
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
