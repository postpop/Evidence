function p = GA(popSize, maxGens, ngenes, bits, objFun, objFunParam)
% p = GA(popSize, maxGens, ngenes, bits, objFun, objFunParam)

% modified from based on:
% TurboGA is an experimental genetic algorithm based on SpeedyGA
% Copyright (C) 2007, 2008, 2009  Keki Burjorjee
% Created and tested under Matlab 7 (R14).

len = bits*ngenes;                    
% The length of the genomes

clampingFlag=1;            
% 1 => Use a mechanism called clamping (see
%       http://www.cs.brandeis.edu/~kekib/GAWorkings.pdf for details)
% 0 => Do not use clamping

probCrossover=.96;           
% The probability of crossing over.
probMutation=0.003;        
% The mutation probability (per bit).
% If clampingFlag=1, probMutation should not be
% dependent on len, the length of the genomes.
%
sigmaScalingFlag=1;       
% Sigma Scaling is described on pg 168 of M. Mitchell's
% GA book. It often improves GA performance.
sigmaScalingCoeff=1;       
% Higher values => less fitness pressure

SUSFlag=1;                
% 1 => Use Stochastic Universal Sampling (pg 168 of
%      M. Mitchell's GA book)
% 0 => Do not use Stochastic Universal Sampling
%      Stochastic Universal Sampling almost always
%      improves performance

crossoverType=2;           
% 0 => no crossover
% 1 => 1pt crossover
% 2 => uniform crossover
% If clampingFlag=1, crossoverType should be 2

visualizationFlag=0;      
% 0 => don't visualize bit frequencies
% 1 => visualize bit frequencies

verboseFlag=1;             
% 1 => display details of each generation
% 0 => run quietly

useMaskRepositoriesFlag=0; 
% 1 => draw uniform crossover and mutation masks from
%      a pregenerated repository of randomly generated bits.
%      Significantly improves the speed of the code with
%      no apparent changes in the behavior of
%      the SGA
% 0 => generate uniform crossover and mutation
%      masks on the fly. Slower.

elitePoolFlag = 1;
% 1 - never loose the best solution by always keeping 
%     (and never mutating) the N best solutions
elitePoolSizeSize = ceil(round(popSize/50)/2)*2; % number of best individuals to clone to the next generation


% clamping parameters
flagFreq=0.01;
unflagFreq=0.1;
flagPeriod=200;

flaggedGens=-ones(1,len);


% crossover masks to use if crossoverType==0
mutationOnlycrossmasks=false(popSize-elitePoolSize,len);

% DOESN'T HELP FOR REAL-LIFE PROBLEMS - delete??
% pre-generate two repositoriesof random binary digits from which the
% the masks used in mutation and uniform crossover will be picked.
% maskReposFactor determines the size of these repositories.
maskReposFactor=50;
uniformCrossmaskRepos=rand((popSize-elitePoolSize)/2,(len+1)*maskReposFactor)<0.5;
mutmaskRepos=rand(popSize-elitePoolSize,(len+1)*maskReposFactor)<probMutation;

% preallocate vectors for recording the average and maximum fitness in each
% generation
avgFitnessHist=zeros(1,maxGens+1);
maxFitnessHist=zeros(1,maxGens+1);

eliteIndiv=[];
eliteFitness=-realmax;


% the population is a popSize by len matrix of randomly generated boolean
% values
pop=rand(popSize,len)<.5;

for gen=0:maxGens
   if verboseFlag || visualizationFlag, tic, end
   bitFreqs=sum(pop)/popSize;
   if clampingFlag
      lociToFlag=abs(0.5-bitFreqs)>(0.5-flagFreq) & flaggedGens<0;
      flaggedGens(lociToFlag)=0;
      lociToUnflag=abs(0.5-bitFreqs)<0.5-unflagFreq ;
      flaggedGens(lociToUnflag)=-1;
      flaggedLoci=flaggedGens>=0;
      flaggedGens(flaggedLoci)=flaggedGens(flaggedLoci)+1;
      mutateLocus=flaggedGens<=flagPeriod;
      if verboseFlag         
         display([' FlaggedLoci = ' num2str(sum(flaggedGens>0))...
            ', maxFlaggedGens = ' num2str(max(flaggedGens)) ', unMutatedLoci = ' num2str(sum(~mutateLocus))]);
      end
   end
   % evaluate the fitness of the population. The vector of fitness values
   % returned  must be of dimensions 1 x popSize.
   param = getParamFromGenome(pop,bits);
   fitnessVals = feval(objFun, param, objFunParam);
   fitnessVals(isnan(fitnessVals)) = min(fitnessVals)-1;
   % stop if almost perfect
   [val, idx] = sort(fitnessVals);
   maxFitnessHist(1,gen+1) = val(end);
   maxIndex = idx(end);
   %[maxFitnessHist(1,gen+1),maxIndex]=max(fitnessVals);
   avgFitnessHist(1,gen+1)=mean(fitnessVals);
   if eliteFitness<maxFitnessHist(gen+1)
      eliteFitness=maxFitnessHist(gen+1);
      eliteIndivGenome=pop(maxIndex,:);
      eliteIndivParam=param(maxIndex,:);
   end
   
   
   p.pop = pop;
   p.fitnessVals = fitnessVals;
   p.maxFitnessHist = maxFitnessHist;
   p.avgFitnessHist = avgFitnessHist;
   p.maxGens = maxGens;
   p.gen = gen;
   p.eliteIndivGenome = eliteIndivGenome;
   p.eliteIndivParam = eliteIndivParam;
   if mean(fitnessVals)/max(abs(fitnessVals))>0.999 || max(fitnessVals)>0.99, break;end
   
   % Conditionally perform bit-frequency visualization
   if visualizationFlag
      subplot(122)
      set (gcf, 'color', 'w');
      hold off
      plot(1:popSize,sort(fitnessVals), '.-');
      title(['gen = ' num2str(gen) ', avg fit = ' sprintf('%0.3f', avgFitnessHist(1,gen+1))...
         ', max fit = ' sprintf('%0.3f', max(fitnessVals)) ', took ' num2str(toc,4) 's']);
      ylabel('sorted fitness');
      xlabel('Locus');
      drawnow;
   end
   
   % Conditionally perform sigma scaling
   if sigmaScalingFlag
      sigma=std(fitnessVals);
      if sigma~=0;
         fitnessVals=1+(fitnessVals-mean(fitnessVals))/...
            (sigmaScalingCoeff*sigma);
         fitnessVals(fitnessVals<=0)=0;
      else
         fitnessVals=ones(1,popSize);
      end
   end
   
   
   % Normalize the fitness values and then create an array with the
   % cumulative normalized fitness values (the last value in this array
   % will be 1)
   cumNormFitnessVals=cumsum(fitnessVals/sum(fitnessVals));
   
   % Use fitness proportional selection with Stochastic Universal or Roulette
   % Wheel Sampling to determine the indices of the parents
   % of all crossover operations
   if SUSFlag
      markers=rand(1,1)+(1:popSize)/popSize;
      markers(markers>1)=markers(markers>1)-1;
   else
      markers = rand(1,popSize);
   end
   [~, parentIndices] = histc(markers,[0 cumNormFitnessVals]);
   parentIndices=parentIndices(randperm(popSize));
   
   eliteParents = pop(idx(end-elitePoolSize+1:end),:);
   % determine the first parents of each mating pair
   firstParents=pop(parentIndices(1:(popSize-elitePoolSize)/2),:);
   % determine the second parents of each mating pair
   secondParents=pop(parentIndices((popSize+elitePoolSize)/2+1:end),:);
   
   % create crossover masks
   if crossoverType==0
      masks=mutationOnlycrossmasks;
   elseif crossoverType==1
      masks=false((popSize-elitePoolSize)/2, len);
      temp=ceil(rand((popSize-elitePoolSize)/2,1)*(len-1));
      for i=1:(popSize-elitePoolSize)/2
         masks(i,1:temp(i))=true;
      end
   else
      if useMaskRepositoriesFlag
         temp=floor(rand*len*(maskReposFactor-1));
         masks=uniformCrossmaskRepos(:,temp+1:temp+len);
      else
         masks=rand((popSize-elitePoolSize)/2, len)<.5;
      end
   end
   
   % determine which parent pairs to leave uncrossed
   reprodIndices=rand((popSize-elitePoolSize)/2,1)<1-probCrossover;
   masks(reprodIndices,:)=false;
   
   % implement crossover
   firstKids=firstParents;
   firstKids(masks)=secondParents(masks);
   secondKids=secondParents;
   secondKids(masks)=firstParents(masks);
   pop=[firstKids; secondKids];
   
   % implement mutation
   if useMaskRepositoriesFlag
      temp=floor(rand*len*(maskReposFactor-1));
      masks=mutmaskRepos(1:end,temp+1:temp+len);
   else
      masks=rand(popSize-elitePoolSize, len)<probMutation;
   end
   if clampingFlag
      masks(:,~mutateLocus)=false;
   end
   pop=[eliteParents;xor(pop,masks)];
   
   % display the generation number, the average Fitness of the population,
   % and the maximum fitness of any individual in the population
   if verboseFlag
      display(['gen=' num2str(gen,'%.3d') '   avgFitness=' ...
         num2str(avgFitnessHist(1,gen+1),'%3.3f') '   maxFitness=' ...
         num2str(maxFitnessHist(1,gen+1),'%3.3f') ...
         '   took ' num2str(toc,'%3.2f') 's']);
   end
   
end
fprintf('\n')
