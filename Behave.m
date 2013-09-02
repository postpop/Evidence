classdef Behave < handle
   
   properties
      stim, stis, stimLen, maxStimLen, minStimLen, SSraw, nanMask % stimulus stuff
      resp        % response vector
      stimIsCyclic = false,
      filtLen,    %length of the filter to be learned in the original stimulus space
      basisSize,  % number of basis vectors
      basis,      % holds the basis vectors (filtLen x basisSize)
      SSrawG, stimLenG, nanMaskG % names for variables on the GPU
   end
   
   methods (Access='public')
      function self = Behave(stim, resp, filtLenORbasis, stimIsCyclic)
         %ARGS:
         %  stim - cell array N x T of stimuli
         %  resp - array N x 1 of responses (response prob, phonotaxis
         %         scores etc.)
         %  filtLenORbasis -
         %  stimIsCyclic - determines whether stim is preriodic and can
         %                 therefore be wrapped
         self.stim = stim;
         self.stis = length(self.stim);
         self.resp = resp;
         self.stimIsCyclic = stimIsCyclic;
         
         if length(filtLenORbasis(:))==1
            self.filtLen = filtLenORbasis;
            self.basis = [];
         else
            self.basis = filtLenORbasis;
            [self.filtLen, self.basisSize] = size(self.basis);
         end
         self.basisSize = size(self.basis,2);
         %% generate stimulus blocks for faster convolution
         
         % determine original stimulus length
         for sti = 1:self.stis
            self.stimLen(sti) = length(self.stim{sti});
         end
         self.minStimLen = min(self.stimLen);
         self.maxStimLen = max(self.stimLen);
         
         %
         if self.stimIsCyclic % need to concatenate stimulus and hence update length
            if  self.filtLen>self.minStimLen % need to concatenate stimulus and hence update length
               self.maxStimLen = ...
                  max(self.stimLen(self.stimLen<self.filtLen))...
                  *ceil(self.filtLen/max(self.stimLen(self.stimLen<self.filtLen)));
               self.maxStimLen = max(max(self.stimLen), self.maxStimLen);
               
            end
            self.maxStimLen = self.maxStimLen + self.filtLen;
         else
            self.maxStimLen = max(self.maxStimLen,self.filtLen);
         end
         % pre-allocate matrix
         if isempty(self.basis)
            self.SSraw = nan(self.maxStimLen*self.stis,self.filtLen);
         else
            self.SSraw = nan(self.maxStimLen*self.stis,self.basisSize);
         end
         
         for sti = 1:self.stis
            if self.stimLen(sti) < self.filtLen % stim shorter than filter
               if self.stimIsCyclic % for periodic stimulus: concatenate
                  repFactor = ceil(self.filtLen/self.stimLen(sti));
                  self.stim{sti} = repmat(self.stim{sti},1,repFactor);
                  self.stimLen(sti) = length(self.stim{sti});%need to update stimLen
               else % otherwise: zero pad CAUTION!!
                  disp('CAUTION!! Stimulus shorter than filter for non-periodic stim. ZERO PADDING. Might yield wrong results!!!')
                  self.stim{sti} = padarray(self.stim{sti}, self.filtLen - self.stimLen(sti), 'post');
               end
            end
            if size(self.stim{sti},1)>=1,self.stim{sti} = self.stim{sti}';end
            
            if self.stimIsCyclic % add one filter length to the end
               try
                  self.stim{sti} = [self.stim{sti}(1:end)'; self.stim{sti}(1:self.filtLen)'];
               catch
                  self.stim{sti} = [self.stim{sti}(1:end); self.stim{sti}(1:self.filtLen)];
               end
            end
            if size(self.stim{sti},1)>=1,self.stim{sti} = self.stim{sti}';end
            
            %self.stimLen(sti) = length(self.stim{sti});%update stimLen in any case for safety
            try
               SSrawTmp = makeStimRows(self.stim{sti},self.filtLen,1);
            catch
               SSrawTmp = makeStimRows(self.stim{sti}',self.filtLen,1);
            end
            if ~isempty(self.basis)% project onto basis
               SSrawTmp = SSrawTmp * self.basis;
            end
            self.SSraw((sti-1)*self.maxStimLen + (1:size(SSrawTmp,1)),:) = SSrawTmp;
         end
         
         self.nanMask = isnan(self.SSraw(:,1));
         self.SSraw(self.nanMask,:)=0;% why?
         self.nanMask = ~self.nanMask;
         self.SSraw = single(self.SSraw);
      end
      
      function initGPU(self)
         try
            %GPUstart
            self.nanMaskG = gpuArray(single(self.nanMask));
            self.SSrawG = gpuArray(single(self.SSraw));
            self.stimLenG = gpuArray(single(self.stimLen));
         catch ME, disp(ME), end
      end
      
      function [bTrain, bTest, trainIdx, testIdx] = getXVsets(self, holdout, type)
         if nargin==2
            type = 'holdout';
         end
         cv = cvpartition(self.stis,type,holdout);
         trainIdx = find(cv.training(1));
         testIdx  = find(cv.test(1));
         filtLenORbasis = self.filtLen;
         if ~isempty(self.basis), filtLenORbasis = self.basis;end
         bTrain = Behave(self.stim(trainIdx), self.resp(trainIdx), filtLenORbasis, self.stimIsCyclic);
         bTest  = Behave(self.stim(testIdx),  self.resp(testIdx),  filtLenORbasis, self.stimIsCyclic);
      end
      
            
   end
   
end
