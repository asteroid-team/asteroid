from lhotse.utils import load_yaml
import lilcom
import numpy as np

sources_yaml = 'data/cuts_sources.yml.gz'
mixture_yaml = 'data/cuts_mix.yml.gz'
noise_yaml = 'data/cuts_noise.yml.gz'


def parse_yaml(y):
    y = load_yaml(y)

    rec_ids = {}
    for entry in y:
        key = entry["features"]["recording_id"]
        if key not in rec_ids.keys():
            rec_ids[key] = [entry["features"]["storage_path"]]
        else:
            rec_ids[key].append(entry["features"]["storage_path"])

    return rec_ids

sources = parse_yaml(sources_yaml)
mixtures = parse_yaml(mixture_yaml)
noises = parse_yaml(noise_yaml)

assert set([x for x in sources.keys()]) == set([x for x in mixtures.keys()]) == set([x for x in noises.keys()])


for k in sources.keys():

    c_sources = []
    for s in sources[k]:
        with open(s, 'rb') as f:
            c_sources.append(lilcom.decompress(f.read()))
    c_sources = np.stack(c_sources)

    with open(noises[k][0], 'rb') as f:
        c_noise = lilcom.decompress(f.read())

    with open(mixtures[k][0], 'rb') as f:
        c_mix = lilcom.decompress(f.read())

    onthefly = np.sum(np.exp(c_sources), 0) + np.exp(c_noise)

    np.testing.assert_array_almost_equal(onthefly, np.exp(c_mix))
