function [fitness, out] = filterAndSumModel(param, p)

[popSize, len] = size(param);
fitness = zeros(popSize, 1);
out = [];

if ~isfield(p,'nlParamScale1') 
   p.nlParamScale1 = 16;
   p.nlParamScale2 = 8;
end
for ind = 1:popSize
   % fix modelParameters
   idx = reshape(1:len, [],p.nFilt)';
   for fil = 1:p.nFilt
      filters(fil,:) = param(ind,idx(fil,1:end-2));%./sum(abs(param(ind,idx(fil,1:end-2))));% normalize filter
      %filters(fil,:) = param(ind,idx(fil,1:end-2))./sum((param(ind,idx(fil,1:end-2))));% normalize filter
      %filters(fil,:) = param(ind,idx(fil,1:end-2))./norm(param(ind,idx(fil,1:end-2)));% normalize filter
      % exp nonlinearity
%      nlParam(fil,1) = 16*(1+param(ind,idx(fil,end-1)));%8*(1+(param(ind,idx(fil,end-1))));%20+40*(param(ind,idx(fil,end-1)));
%      nlParam(fil,2) = 6*(1+param(ind,idx(fil,end))+eps);%4*(1+param(ind,idx(fil,end))+eps);%4+8*param(ind,idx(fil,end))+eps;
      nlParam(fil,1) = p.nlParamScale1*(1+param(ind,idx(fil,end-1)));%8*(1+(param(ind,idx(fil,end-1))));%20+40*(param(ind,idx(fil,end-1)));
      nlParam(fil,2) = p.nlParamScale2*(1+param(ind,idx(fil,end))+eps);%4*(1+param(ind,idx(fil,end))+eps);%4+8*param(ind,idx(fil,end))+eps;
   end
   
   if p.nFilt>1 && p.orthogonalizeFilter%orthogonalize filters
      filters = orth(filters');
   end
   if p.GPU && popSize>1 % on GPU
      fitness(ind) = getClassifierGPU(filters, nlParam, p);
   elseif popSize>1 % does not compute intermediate values
      fitness(ind) = getClassifierMnml(filters, nlParam, p);
   elseif popSize==1 % classifier with full output. slow!
      [fitness(ind), out.features, out.filterOut, out.y, out.X, out.betahat, out.nlOut] = getClassifier(filters, nlParam, p);
      %[fitness(ind), out.features, out.filterOut, out.y, out.X, out.betahat, out.nlOut] = getClassifierLoop(filters, nlParam, p);
      out.mse = fitness(ind);
      if length(out.y)==1
         out.rsq = 1;
      else
         cc = corrcoef(out.y, p.bee.resp);
         out.rsq = cc(2)^2;
      end
      out.filters = filters;
      out.nlParam = nlParam;
   end
end
fitness = 1-fitness'./(nanvar(p.bee.resp)+eps);
fitness(isnan(fitness)) = 0;
