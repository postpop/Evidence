function [mse, features, filterOut, y, X, betahat, nlOut] = getClassifier(filters, nlParam, p)


features = zeros(p.nFilt, p.bee.stis);
filterOut = zeros(p.nFilt, p.bee.stis, p.bee.maxStimLen);
nlOut = filterOut;

for fil = 1:p.nFilt
   % linear filter stage
   if p.orthogonalizeStim%orthogonalize stim w.r.t. the current filter
      p.SSraw = p.SSraw-(p.bee.SSraw*filters(fil,:)')*filters(fil,:)';% NOT TESTED YET!!!!!
   end
   
   tmpFilt = p.bee.SSraw*filters(fil,:)';
   tmpFilt = tmpFilt.*p.bee.nanMask;
   filterOut(fil,:,:) = reshape(tmpFilt,p.bee.maxStimLen,p.bee.stis)';
   
   % sigmoidal nonlinearity
   tmpFilt = 1./(1 + exp(-nlParam(fil,1)*(tmpFilt  - nlParam(fil,2))));
   tmpFilt = tmpFilt.*p.bee.nanMask;
   nlOut(fil,:,:) = reshape(tmpFilt,p.bee.maxStimLen,p.bee.stis)';
   % integrator
   if ~isfield(p,'sum') || p.sum==1
      features(fil,:) = nansum(nlOut(fil,:,:),3)./p.bee.stimLen;
   end
end
%%
if ~isfield(p,'sum') || p.sum==1
   X = [ones(1, size(features,2)); p.givenFeatures; features]';
   %X = x2fx([p.givenFeatures; features]', 'purequadratic');
   betahat = X\p.bee.resp';
   y = X*betahat;
   mse = nanmean((y' - p.bee.resp).^2);
else
   X = [ones(1, length(features)); squeeze(nlOut)']';
   y = [];
   betahat = [];
   for t = 1:size(X,2)-1
      betahat(t,:) = X(:,[1 t+1])\p.bee.resp';
      y(:,t) = X(:,[1 t+1])*betahat(t,:)';
      mse(t) = nanmean((y(:,t)' - p.bee.resp).^2);
   end
   [mse idx] = min(mse);
   y = y(:,idx);
end

% %if size(X,1)~=size(X,2), X = X';end
% betahat = X\p.bee.resp';
% y = X*betahat;
%
% % sAll = 2.^(-10:2:10);
% % cAll = 2.^(-10:2:10);
% % [accuracy, predlabel,H,fitScape,decisionValues, modelOpt] = svmGridReg(X, p.bee.resp', sAll, cAll);
% % betahat = modelOpt;
% % y = predlabel;%X*betahat;
%
% mse = nanmean((y' - p.bee.resp).^2);
%
%
