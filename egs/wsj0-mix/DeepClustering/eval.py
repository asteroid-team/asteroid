''' Evaluation of the Deep clustering model '''
import os
import argparse
import random
import sklearn
import torch
import tqdm
from torch.utils.data import DataLoader

from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr, pairwise_mse
from asteroid.data.wsj0_mix import WSJ2mixDataset 
#from asteroid.metrics import get_metrics
import asteroid.filterbanks as fb
from asteroid.filterbanks.transforms import take_mag
from model import load_best_model
from train import DcSystem

from ipdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', type=int, default=0,
                    help='Whether to use the GPU for model execution')
parser.add_argument('--exp_dir', default='exp/tmp',
                    help='Experiment root')
parser.add_argument('--n_save_ex', type=int, default=-1,
                    help='Number of audio examples to save, -1 means all')
compute_metrics = ['si_sdr', 'sdr', 'sir', 'sar', 'stoi']

kmeans = sklearn.cluster.KMeans(n_clusters=2)

def main(conf):
    set_trace()
    test_set = WSJ2mixDataset(conf['data']['tt_wav_len_list'],
                              conf['data']['wav_base_path']+'/tt',
                              sample_rate=conf['data']['sample_rate'])
    test_loader = DataLoader(test_set, shuffle=True,
                             batch_size=1,
                             num_workers=conf['data']['num_workers'],
                             drop_last=False)
    istft = fb.Decoder(fb.STFTFB(**conf['filterbank']))
    exp_dir = conf['main_args']['exp_dir']
    model_path = os.path.join(exp_dir, 'checkpoints/_ckpt_epoch_0.ckpt')
    model = load_best_model(conf, model_path)
    pit_loss = PITLossWrapper(pairwise_mse, mode='pairwise')

    system = DcSystem(model, None, None, None, config=conf)

    # Randomly choose the indexes of sentences to save.
    exp_dir = conf['main_args']['exp_dir']
    exp_save_dir = os.path.join(exp_dir, 'examples/')
    n_save = conf['main_args']['n_save_ex']
    if n_save == -1:
        n_save = len(test_set)
    save_idx = random.sample(range(len(test_set)), n_save)
    series_list = []
    torch.no_grad().__enter__()

    for batch in test_loader:
        batch = [ele.type(torch.float32) for ele in batch]
        inputs, targets, masks = system.unpack_data(batch)
        est_targets = system(inputs)
        mix_stft = system.enc(inputs.unsqueeze(1))
        min_loss, min_idx = pit_loss.best_perm_from_perm_avg_loss(\
                pairwise_mse, est_targets[1], masks)
        for sidx in min_idx:
            src_stft = mix_stft * est_targets[1][sidx]
            src_sig = istft(src_stft)


def cluster(self, net_embed, vad_mask):
    """

    source: github.com/funcwj/deep-clustering.git
    Arguments
        spectra:    log-magnitude spectrogram(real numbers)
        net_embed: Embedding output from network: Dimension TF x D
        vad_mask:   binary mask for non-silence bins(if non-sil: 1)
        return
            pca_embed: PCA embedding vector(dim 3)
            spk_masks: binary masks for each speaker
    """
    # filter silence embeddings: TF x D => N x D
    active_embed = net_embed[vad_mask.reshape(-1)]
    # classes: N x D
    # pca_mat: N x 3
    classes = self.kmeans.fit_predict(active_embed)

    def form_mask(classes, spkid, vad_mask):
        mask = ~vad_mask
        # mask = np.zeros_like(vad_mask)
        mask[vad_mask] = (classes == spkid)
        return mask

    return [
        form_mask(classes, spk, vad_mask) for spk in range(self.num_spks)
    ]




if __name__ == '__main__':
    import yaml
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    with open('local/conf.yml') as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    print(arg_dic)
    main(arg_dic)
