function param = getParamFromGenome(genomes, bits)

[popSize, nBases] = size(genomes);
nGenes = nBases/bits;
param = zeros(popSize, nGenes);
base = repmat(2.^(0:bits-1),popSize,nGenes);

tmp = genomes.*base;
for ind = 1:popSize
   param(ind,:) = sum(reshape(tmp(ind,:),[],nGenes));
end
param = (param - 2^(bits-1))/2^(bits-1);
