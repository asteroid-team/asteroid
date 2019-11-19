# Asteroid : Audio Source Separation on steroids

## Why Asteroid ? 
Audio source separation and speech enhancement are fast evolving fields with 
a growing number of papers submitted to conferences each year. While datasets 
such as [wsj0-{2, 3}mix](http://www.merl.com/demos/deep-clustering),
[WHAM](http://wham.whisper.ai/) or 
[MS-SNSD](https://github.com/microsoft/MS-SNSD) are being shared, there has 
been little effort to create common codebases for development and evaluation 
of source separation and speech enhancement algorithms. Hence AsSteroid !

The intent of Asteroid is to be a __community-based project__, to go beyond 
sharing datasets. 
We share tools to create new algorithms as well as recipes 
to reproduce published papers.

## Guiding principles
* __User friendliness.__
* __Modularity.__
* __Extensibility.__
* __Reproducibility.__

## Recipes 
* [ ] Deep clustering ([Hershey et al.](https://arxiv.org/abs/1508.04306) and [Isik et al.](https://arxiv.org/abs/1607.02173))
* [ ] Tasnet ([Luo et al.](https://arxiv.org/abs/1711.00541))
* [ ] ConvTasnet ([Luo et al.](https://arxiv.org/abs/1809.07454))
* [ ] Chimera ++ (for ) ([Luo et al.](https://arxiv.org/abs/1611.06265) and [Wang et al.](https://ieeexplore.ieee.org/document/8462507))
* [ ] FurcaNeXt ([Shi et al.](https://arxiv.org/abs/1902.04891))
* [ ] DualPathRNN ([Luo et al.](https://arxiv.org/abs/1910.06379))
* [ ] Two step learning ([Tzinis et al.](https://arxiv.org/abs/1910.09804))

## Filterbanks
Analysis-synthesis
* [x] STFT (See unit tests)
* [x] Free e.g fully learned ([Luo et al.](https://arxiv.org/abs/1711.00541))
* [x] Analytic free e.g fully learned under analycity constraint ([Pariente et al.](https://128.84.21.199/abs/1910.10400))
* [x] Parametrized Sinc ([Pariente et al.](https://128.84.21.199/abs/1910.10400))

Analysis only (can be extended)
* [ ] Fixed Multi-Phase Gammatones ([Ditter et al.](https://arxiv.org/abs/1910.11615))
* [ ] Parametrized modulated Gaussian windows ([Openreview](https://openreview.net/forum?id=HyewT1BKvr))
* [ ] Parametrized Gammatone ([Openreview](https://openreview.net/forum?id=HyewT1BKvr))
* [ ] Parametrized Gammachirp ([Openreview](https://openreview.net/forum?id=HyewT1BKvr))

## Running a recipe

## Writing your own recipe

## Contributing

## Reproducibility
Make table with all the results per dataset here.
