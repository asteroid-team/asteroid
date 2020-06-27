## for debugging purposes only
import numpy as np
from lhotse.dataset.source_separation import PreMixedSourceSeparationDataset, DynamicallyMixedSourceSeparationDataset
from lhotse.cut import CutSet
import torch
from local.dataset_wrapper import LhotseDataset, OnTheFlyMixing

#train_set_static = PreMixedSourceSeparationDataset(sources_set=CutSet.from_yaml('data/cuts_sources.yml.gz'),
   #                                             mixtures_set=CutSet.from_yaml('data/cuts_mix.yml.gz'),
              #                                  root_dir=".")
#
#train_set_static[0]
#from torch.utils.data import DataLoader, Dataset



train_set = OnTheFlyMixing() #LhotseDataset(train_set_static, 300, 0)
train_set[0]
#for i in DataLoader(train_set, batch_size=1, shuffle=True):
 #   print(i)