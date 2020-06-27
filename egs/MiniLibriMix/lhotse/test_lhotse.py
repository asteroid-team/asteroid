## for debugging purposes only

from lhotse.dataset.source_separation import PreMixedSourceSeparationDataset, DynamicallyMixedSourceSeparationDataset
from lhotse.cut import CutSet
train_set = PreMixedSourceSeparationDataset(sources_set=CutSet.from_yaml('data/cuts_sources.yml.gz'),
                                                mixtures_set=CutSet.from_yaml('data/cuts_mix.yml.gz'),
                                                root_dir=".")

train_set[0]
train_set[1]