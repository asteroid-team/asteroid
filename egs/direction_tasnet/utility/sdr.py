import numpy as np
from itertools import permutations
from torch.autograd import Variable

import scipy,time,numpy
import itertools

import torch

def calc_sdr(estimation, origin, mask=None):
    """
    batch-wise SDR caculation for one audio file.
    estimation: (batch, nsample)
    origin: (batch, nsample)
    mask: optional, (batch, nsample), binary
    """
    
    if mask is not None:
        origin = origin * mask
        estimation = estimation * mask
    
    origin_power = np.sum(origin**2, 1, keepdims=True) + 1e-8  # (batch, 1)
    
    scale = np.sum(origin*estimation, 1, keepdims=True) / origin_power  # (batch, 1)
    
    est_true = scale * origin  # (batch, nsample)
    est_res = estimation - est_true  # (batch, nsample)
    
    true_power = np.sum(est_true**2, 1)
    res_power = np.sum(est_res**2, 1)
    
    return 10*np.log10(true_power) - 10*np.log10(res_power)  # (batch, 1)


def calc_sdr_torch(estimation, origin, mask=None):
    """
    batch-wise SDR caculation for one audio file on pytorch Variables.
    estimation: (batch, nsample)
    origin: (batch, nsample)
    mask: optional, (batch, nsample), binary
    """
    
    if mask is not None:
        origin = origin * mask
        estimation = estimation * mask
    
    origin_power = torch.pow(origin, 2).sum(1, keepdim=True) + 1e-8  # (batch, 1)
    
    scale = torch.sum(origin*estimation, 1, keepdim=True) / origin_power  # (batch, 1)
    
    est_true = scale * origin  # (batch, nsample)
    est_res = estimation - est_true  # (batch, nsample)
    
    true_power = torch.pow(est_true, 2).sum(1)
    res_power = torch.pow(est_res, 2).sum(1)
    
    return 10*torch.log10(true_power) - 10*torch.log10(res_power)  # (batch, 1)


def batch_SDR(estimation, origin, mask=None):
    """
    batch-wise SDR caculation for multiple audio files.
    estimation: (batch, nsource, nsample)
    origin: (batch, nsource, nsample)
    mask: optional, (batch, nsample), binary
    """
    
    batch_size_est, nsource_est, nsample_est = estimation.shape
    batch_size_ori, nsource_ori, nsample_ori = origin.shape
    
    assert batch_size_est == batch_size_ori, "Estimation and original sources should have same shape."
    assert nsource_est == nsource_ori, "Estimation and original sources should have same shape."
    assert nsample_est == nsample_ori, "Estimation and original sources should have same shape."
    
    assert nsource_est < nsample_est, "Axis 1 should be the number of sources, and axis 2 should be the signal."
    
    batch_size = batch_size_est
    nsource = nsource_est
    nsample = nsample_est
    
    # zero mean signals
    estimation = estimation - np.mean(estimation, 2, keepdims=True)
    origin = origin - np.mean(origin, 2, keepdims=True)
    
    # possible permutations
    perm = list(set(permutations(np.arange(nsource))))
    
    # pair-wise SDR
    SDR = np.zeros((batch_size, nsource, nsource))
    for i in range(nsource):
        for j in range(nsource):
            SDR[:,i,j] = calc_sdr(estimation[:,i], origin[:,j], mask)
    
    # choose the best permutation
    SDR_max = []
    for i in range(batch_size):
        SDR_perm = []
        for permute in perm:
            sdr = 0.
            for idx in range(len(permute)):
                sdr += SDR[i][idx][permute[idx]]
            SDR_perm.append(sdr)
        SDR_max.append(np.max(SDR_perm) / nsource)
    
    return np.asarray(SDR_max)


def batch_SDR_torch(estimation, origin, mask=None):
    """
    batch-wise SDR caculation for multiple audio files.
    estimation: (batch, nsource, nsample)
    origin: (batch, nsource, nsample)
    mask: optional, (batch, nsample), binary
    """
    
    batch_size_est, nsource_est, nsample_est = estimation.size()
    batch_size_ori, nsource_ori, nsample_ori = origin.size()
    
    assert batch_size_est == batch_size_ori, "Estimation and original sources should have same shape."
    assert nsource_est == nsource_ori, "Estimation and original sources should have same shape."
    assert nsample_est == nsample_ori, "Estimation and original sources should have same shape."
    
    assert nsource_est < nsample_est, "Axis 1 should be the number of sources, and axis 2 should be the signal."
    
    batch_size = batch_size_est
    nsource = nsource_est
    nsample = nsample_est
    
    # zero mean signals
    estimation = estimation - torch.mean(estimation, 2, keepdim=True).expand_as(estimation)
    origin = origin - torch.mean(origin, 2, keepdim=True).expand_as(estimation)
    
    # possible permutations
    perm = list(set(permutations(np.arange(nsource))))
    
    # pair-wise SDR
    SDR = torch.zeros((batch_size, nsource, nsource)).type(estimation.type())
    for i in range(nsource):
        for j in range(nsource):
            SDR[:,i,j] = calc_sdr_torch(estimation[:,i], origin[:,j], mask)
    
    # choose the best permutation
    SDR_max = []
    SDR_perm = []
    for permute in perm:
        sdr = []
        for idx in range(len(permute)):
            sdr.append(SDR[:,idx,permute[idx]].view(batch_size,-1))
        sdr = torch.sum(torch.cat(sdr, 1), 1)
        SDR_perm.append(sdr.view(batch_size, 1))
    SDR_perm = torch.cat(SDR_perm, 1)
    SDR_max, _ = torch.max(SDR_perm, dim=1)
    
    return SDR_max / nsource


def compute_measures(se,s,j):
    Rss=s.transpose().dot(s)
    this_s=s[:,j]

    a=this_s.transpose().dot(se)/Rss[j,j]
    e_true=a*this_s
    e_res=se-a*this_s
    Sss=np.sum((e_true)**2)
    Snn=np.sum((e_res)**2)

    SDR=10*np.log10(Sss/Snn)

    Rsr= s.transpose().dot(e_res)
    b=np.linalg.inv(Rss).dot(Rsr)

    e_interf = s.dot(b)
    e_artif= e_res-e_interf

    SIR=10*np.log10(Sss/np.sum((e_interf)**2))
    SAR=10*np.log10(Sss/np.sum((e_artif)**2))
    return SDR, SIR, SAR

def GetSDR(se,s):
    se=se-np.mean(se,axis=0)
    s=s-np.mean(s,axis=0)
    nsampl,nsrc=se.shape
    nsampl2,nsrc2=s.shape
    assert(nsrc2==nsrc)
    assert(nsampl2==nsampl)

    SDR=np.zeros((nsrc,nsrc))
    SIR=SDR.copy()
    SAR=SDR.copy()

    for jest in range(nsrc):
        for jtrue in range(nsrc):
            SDR[jest,jtrue],SIR[jest,jtrue],SAR[jest,jtrue]=compute_measures(se[:,jest],s,jtrue)


    perm=list(itertools.permutations(np.arange(nsrc)))
    nperm=len(perm)
    meanSIR=np.zeros((nperm,))
    for p in range(nperm):
        tp=SIR.transpose().reshape(nsrc*nsrc)
        idx=np.arange(nsrc)*nsrc+list(perm[p])
        meanSIR[p]=np.mean(tp[idx])
    popt=np.argmax(meanSIR)
    per=list(perm[popt])
    idx=np.arange(nsrc)*nsrc+per
    SDR=SDR.transpose().reshape(nsrc*nsrc)[idx]
    SIR=SIR.transpose().reshape(nsrc*nsrc)[idx]
    SAR=SAR.transpose().reshape(nsrc*nsrc)[idx]
    return SDR, SIR, SAR, per
