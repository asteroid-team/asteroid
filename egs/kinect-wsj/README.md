### Kinect-WSJ dataset
Kinect-WSJ is a reverberated, noisy version of the WSJ0-2MIX dataset. 
Microphones are placed on a linear array with spacing between the devices 
resembling that of Microsoft Kinect ™, the device used to record the CHiME-5 dataset. 
This was done so that we could use the real ambient noise captured as part of CHiME-5 dataset. 
The room impulse responses (RIR) were simulated for a sampling rate of 16,000 Hz.

**Requirements**  
* wsj_path :  Path to precomputed wsj-2mix dataset. Should contain the folder 2speakers/wav16k/. 
If you don't have wsj_mix dataset, please create it using the scripts in egs/wsj0_mix
* chime_path : Path to chime-5 dataset. Should contain the folders train, dev and eval
* dihard_path : Path to dihard labels. Should contain ```*.lab``` files for the train and dev set

**References**  
[Original repo](https://github.com/sunits/Reverberated_WSJ_2MIX/)

```
@inproceedings{sivasankaran2020,  
  booktitle = {2020 28th {{European Signal Processing Conference}} ({{EUSIPCO}})},  
  title={Analyzing the impact of speaker localization errors on speech separation for automatic speech recognition},
  author={Sunit Sivasankaran and Emmanuel Vincent and Dominique Fohr},
  year={2021},  
  month = Jan,  
}
```

