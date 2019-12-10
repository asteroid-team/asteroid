### Mixture model based source separation on SMS-WSJ

Configuration:                                               
  beamformer = 'mvdr_souden'                         
  mask_estimator = 'cacgmm'                                       
  postfilter = None                                             
  stft_shift = 128                                              
  stft_size = 512                                              
  stft_window = 'hann'                                          
  
evaluation metric   | cv_dev93      |  test_eval92 
:-------------------|--------------:|--------------:
PESQ			    | 2.068		    | 2.187
STOI				| 0.820			| 0.800
mir_eval SDR		| 12.34			| 12.11
invasive SDR		| 15.74			| 15.47

### References
```BibTex
@Article{SmsWsj19,
  author    = {Drude, Lukas and Heitkaemper, Jens and Boeddeker, Christoph and Haeb-Umbach, Reinhold},
  title     = {{SMS-WSJ}: Database, performance measures, and baseline recipe for multi-channel source separation and recognition},
  journal   = {arXiv preprint arXiv:1910.13934},
  year      = {2019},
}
```