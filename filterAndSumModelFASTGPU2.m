function [fitness, out] = filterAndSumModelFASTGPU2(param, p)

[popSize len] = size(param);
out = [];

tmpParamG = reshape(gpuArray(param),popSize,len/p.nFilt,[]);
featuresG = parallel.gpu.GPUArray.zeros(p.nFilt, p.bee.stis, p.popSize);
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
      %tmp = bxfun(@minus, tmp, 6*(1+tmpParam(chunkIdx{n},p.nComp + 2,f))'+eps);
      %tmpG = tmpG - repmat(6*(1+tmpParamG(chunkIdx{n},p.nComp + 2,f))', size(tmpG,1), 1);
      %tmp = bsxfun(@times, tmp, -16*(1+tmpParam(chunkIdx{n},p.nComp + 1,f))');
      %tmpG = tmpG.*repmat(-16*(1+tmpParamG(chunkIdx{n},p.nComp + 1,f))', size(tmpG,1), 1);
      %tmpG = 1./(1+exp(tmpG));
      %tmp = bsxfun(@times,tmp, p.bee.nanMaskG);
      %tmpG = tmpG.*repmat(p.bee.nanMaskG,1,size(tmpG,2));
      tmpG = arrayfun(@nonlin, tmpG, ...
         repmat(6*(1+tmpParamG(chunkIdx{n},p.nComp + 2,f))', size(tmpG,1), 1),...
         repmat(-16*(1+tmpParamG(chunkIdx{n},p.nComp + 1,f))', size(tmpG,1), 1),...
         repmat(p.bee.nanMaskG,1,size(tmpG,2)));
      %tmp = reshape(tmp,p.bee.maxStimLen,p.bee.stis, length(chunkIdx{n}));
      %tmpG = squeeze(sum(reshape(tmpG,p.bee.maxStimLen,p.bee.stis, length(chunkIdx{n}))));
      %features(f,:,chunkIdx{n}) = reshape(bsxfun(@times, sum(tmp),1./p.bee.stimLenG), p.bee.stis, length(chunkIdx{n}));
      %featuresG(f,:,chunkIdx{n}) = (reshape(tmpG.*repmat(1./p.bee.stimLenG',1,size(tmpG,2)), p.bee.stis, length(chunkIdx{n})));   
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

function x = nonlin(x,p1,p2,nanMask) 
x = nanMask./(1 + exp(-16.*p2.*(x - 6.*p1)));
