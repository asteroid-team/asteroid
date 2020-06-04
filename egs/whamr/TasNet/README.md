### Results

Following table shows SI-SDR improvements of the original paper 
and Asteroid's recipe. In average, our system is about 2dB better
for the same number of parameters.

|   task          |Input (dB)|  Paper  |    Here  |
|:---------------:|:--------:|:-------:|:--------:|
| sep_clean       |  0.0     |   14.2  | **16.8** |
| sep_noisy       | -4.5     |   12.0  | **13.7** |
| sep_reverb      | -3.3     |   8.9   | **10.6** |  
| sep_reverb_noisy| -6.1     |   9.2   | **11.0** |

All the experiments were done with the same configuration (only the tasks changes):

```yaml
data:
  mode: min
  sample_rate: 8000
  task: sep_clean
filterbank:
  kernel_size: 40
  n_filters: 512
  stride: 20
masknet:
  dropout: 0.3
  n_layers: 4
  n_src: 2
  n_units: 600
optim:
  lr: 0.001
  optimizer: adam
  weight_decay: 1.0e-05
training:
  batch_size: 32
  early_stop: true
  epochs: 200
  half_lr: true
```

### All metrics 
The following are output of `final_metrics.json`.
##### Clean separation
```json
"si_sdr": 16.76808701529354,
"si_sdr_imp": 16.768125949735637,
"sdr": 17.189693158808677,
"sdr_imp": 17.037464558464684,
"sir": 27.281386306662466,
"sir_imp": 27.129157706318463,
"sar": 17.7691818584944,
"sar_imp": -131.3877807475557,
"stoi": 0.9642581421603139,
"stoi_imp": 0.22618525229628597
```
##### Noisy separation
```json
"si_sdr": 9.25277109949865,
"si_sdr_imp": 13.739940125395641,
"sdr": 9.901496581627462,
"sdr_imp": 14.130608088232679,
"sir": 23.814840532053537,
"sir_imp": 23.6679900856008,
"sar": 10.323204569060126,
"sar_imp": 8.928649466834461,
"stoi": 0.8782651045376821,
"stoi_imp": 0.2505054514470455
```
##### Reverberant separation
```json
"si_sdr": 7.341333914145555,
"si_sdr_imp": 10.631356216662889,
"sdr": 9.431018910042448,
"sdr_imp": 9.547844736186036,
"sir": 21.850962651547228,
"sir_imp": 21.694702659905484,
"sar": 9.901812210974635,
"sar_imp": -10.838234734968795,
"stoi": 0.899942067670028,
"stoi_imp": 0.2242012600717384
```

##### Noisy reverberant separation
```json
"si_sdr": 4.878757792806432,
"si_sdr_imp": 11.006090461420074,
"sdr": 6.692355035696564,
"sdr_imp": 10.178869336725988,
"sir": 20.067529432296666,
"sir_imp": 19.915254119873744,
"sar": 7.196904153627265,
"sar_imp": 4.430949537724975,
"stoi": 0.8279722173828477,
"stoi_imp": 0.22829729176536692
```
