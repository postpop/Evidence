# Evidence
Fits a multi-dimensional LNP-Integrator model to behavioral data.

## Model structure:
1. Feature extraction: time-varying stimulus is processed by LN models
2. Integration: output of each LN ('firing rate') is integrated to yield a feature value ('spike count')
3. Weighing: Feature values of multiple LN models are linearly combined to yield behavioral response value

Some tweaks to optimize performance:
- filter is represented in a raised-cosine basis
- nonlinearity is parameterized (sigmoidal)
- GPU implementation of the model for faster evaluation during fitting (using Matlab's GPU capabilities).

## Demo code
```matlab
load('demo.mat')
stim = ;
resp = ;
p.bee = Behave(stim, resp, ..);
pGa = GA(p);
```

This should yield the following Figure:

![demo figure](demo.png)


## Code base used in:
__Jan Clemens__, Bernhard Ronacher  
Feature extraction and combination underlying decision making during courtship in grasshoppers   
[_2013_, Journal of Neuroscience, 33(29):12136-12145](http://www.jneurosci.org/content/33/29/12136.abstract) | [pdf](http://www.princeton.edu/~janc/pdf/clemens_2013_feature.pdf)

__Jan Clemens__, Matthias Hennig  
Computational principles underlying the recognition of acoustic signals in insects   
[_2013_, Journal of Computational Neuroscience, 35(1):75-85](http://link.springer.com/article/10.1007/s10827-013-0441-0) | [pdf](http://www.princeton.edu/~janc/pdf/clemens_2013_computational.pdf)
