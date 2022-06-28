import json
import os
import torch
from torch import nn
from sklearn.cluster import KMeans

from asteroid import torch_utils
import asteroid_filterbanks as fb
from asteroid.engine.optimizers import make_optimizer
from asteroid_filterbanks.transforms import mag, apply_mag_mask
from asteroid.dsp.vad import ebased_vad
from asteroid.masknn.recurrent import SingleRNN
from asteroid.utils.torch_utils import pad_x_to_y


def make_model_and_optimizer(conf):
    """Function to define the model and optimizer for a config dictionary.
    Args:
        conf: Dictionary containing the output of hierachical argparse.
    Returns:
        model, optimizer.
    The main goal of this function is to make reloading for resuming
    and evaluation very simple.
    """
    enc, dec = fb.make_enc_dec("stft", **conf["filterbank"])
    masker = Chimera(enc.n_feats_out // 2, **conf["masknet"])
    model = Model(enc, masker, dec)
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    return model, optimizer


class Chimera(nn.Module):
    def __init__(
        self,
        in_chan,
        n_src,
        rnn_type="lstm",
        n_layers=2,
        hidden_size=600,
        bidirectional=True,
        dropout=0.3,
        embedding_dim=20,
        take_log=False,
        EPS=1e-8,
    ):
        super().__init__()
        self.input_dim = in_chan
        self.n_src = n_src
        self.take_log = take_log
        # RNN common
        self.embedding_dim = embedding_dim
        self.rnn = SingleRNN(
            rnn_type,
            in_chan,
            hidden_size,
            n_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        rnn_out_dim = hidden_size * 2 if bidirectional else hidden_size
        # Mask heads
        self.mask_layer = nn.Linear(rnn_out_dim, in_chan * self.n_src)
        self.mask_act = nn.Sigmoid()  # sigmoid or relu or softmax
        # DC head
        self.embedding_layer = nn.Linear(rnn_out_dim, in_chan * embedding_dim)
        self.embedding_act = nn.Tanh()  # sigmoid or tanh
        self.EPS = EPS

    def forward(self, input_data):
        batch, _, n_frames = input_data.shape
        if self.take_log:
            input_data = torch.log(input_data + self.EPS)
        # Common net
        out = self.rnn(input_data.permute(0, 2, 1))
        out = self.dropout(out)

        # DC head
        proj = self.embedding_layer(out)  # batch, time, freq * emb
        proj = self.embedding_act(proj)
        proj = proj.view(batch, n_frames, -1, self.embedding_dim).transpose(1, 2)
        # (batch, freq * frames, emb)
        proj = proj.reshape(batch, -1, self.embedding_dim)
        proj_norm = torch.norm(proj, p=2, dim=-1, keepdim=True)
        projection_final = proj / (proj_norm + self.EPS)

        # Mask head
        mask_out = self.mask_layer(out).view(batch, n_frames, self.n_src, self.input_dim)
        mask_out = mask_out.permute(0, 2, 3, 1)
        mask_out = self.mask_act(mask_out)
        return projection_final, mask_out


class Model(nn.Module):
    def __init__(self, encoder, masker, decoder):
        super().__init__()
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_rep = self.encoder(x)
        final_proj, mask_out = self.masker(mag(tf_rep))
        return final_proj, mask_out

    def separate(self, x):
        """Separate with mask-inference head, output waveforms"""
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_rep = self.encoder(x)
        proj, mask_out = self.masker(mag(tf_rep))
        masked = apply_mag_mask(tf_rep.unsqueeze(1), mask_out)
        wavs = torch_utils.pad_x_to_y(self.decoder(masked), x)
        dic_out = dict(tfrep=tf_rep, mask=mask_out, masked_tfrep=masked, proj=proj)
        return wavs, dic_out

    def dc_head_separate(self, x):
        """Cluster embeddings to produce binary masks, output waveforms"""
        kmeans = KMeans(n_clusters=self.masker.n_src)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_rep = self.encoder(x)
        mag_spec = mag(tf_rep)
        proj, mask_out = self.masker(mag_spec)
        active_bins = ebased_vad(mag_spec)
        active_proj = proj[active_bins.view(1, -1)]
        #
        bin_clusters = kmeans.fit_predict(active_proj.cpu().data.numpy())
        # Create binary masks
        est_mask_list = []
        for i in range(self.masker.n_src):
            # Add ones in all inactive bins in each mask.
            mask = ~active_bins
            mask[active_bins] = torch.from_numpy((bin_clusters == i)).to(mask.device)
            est_mask_list.append(mask.float())  # Need float, not bool
        # Go back to time domain
        est_masks = torch.stack(est_mask_list, dim=1)
        masked = apply_mag_mask(tf_rep, est_masks)
        wavs = pad_x_to_y(self.decoder(masked), x)
        dic_out = dict(tfrep=tf_rep, mask=mask_out, masked_tfrep=masked, proj=proj)
        return wavs, dic_out


def load_best_model(train_conf, exp_dir):
    """Load best model after training.

    Args:
        train_conf (dict): dictionary as expected by `make_model_and_optimizer`
        exp_dir(str): Experiment directory. Expects to find
            `'best_k_models.json'` of `checkpoints` directory in it.

    Returns:
        nn.Module the best (or last) pretrained model according to the val_loss.
    """
    # Create the model from recipe-local function
    model, _ = make_model_and_optimizer(train_conf)
    try:
        # Last best model summary
        with open(os.path.join(exp_dir, "best_k_models.json"), "r") as f:
            best_k = json.load(f)
        best_model_path = min(best_k, key=best_k.get)
    except FileNotFoundError:
        # Get last checkpoint
        all_ckpt = os.listdir(os.path.join(exp_dir, "checkpoints/"))
        all_ckpt = [
            (ckpt, int("".join(filter(str.isdigit, os.path.basename(ckpt)))))
            for ckpt in all_ckpt
            if ckpt.find("ckpt") >= 0
        ]
        all_ckpt.sort(key=lambda x: x[1])
        best_model_path = os.path.join(exp_dir, "checkpoints", all_ckpt[-1][0])
    # Load checkpoint
    checkpoint = torch.load(best_model_path, map_location="cpu")
    # Load state_dict into model.
    model = torch_utils.load_state_dict_in(checkpoint["state_dict"], model)
    model.eval()
    return model
