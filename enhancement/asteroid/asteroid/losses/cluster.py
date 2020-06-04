import torch

def batch_matrix_norm(matrix, norm_order=2):
    """ normalization of the matrix
    Args:
        matrix: torch.Tensor. Expected shape [batch, *]
        norm_order: int. Order of normalization. 

    Returns:
        normed matrix: torch.Tensor. 
    """
    return torch.norm(matrix, p=norm_order, dim=[1,2])

def deep_clustering_loss(embedding, tgt_index, spk_cnt=None):
    """ Compute the deep clustering loss defined in  
    Args:
        embedding: torch.Tensor. Expected shape  [batch, frame x frequency, embeddingDim]
        tgt_index: torch.LongTensor. Dominating source in each time frequency bin. Expected shape:
            [batch, frequency, frame]
        spk_cnt: int. Number of speakers. Estimated from tgt_index if not given.

    Returns:
         `torch.Tensor`. Deep clustering loss for every sample

    Examples:

        >>> import torch
        >>> from asteroid.losses.cluster import deep_clustering_loss
        >>> spk_cnt = 3
        >>> embedding = torch.randn(10, 5*400, 20)
        >>> targets = torch.LongTensor(10, 400, 5).random_(0, spk_cnt)
        >>> loss = deep_clustering_loss(embedding, targets, spk_cnt)
        >>> print(loss.size)
        torch.size([10])

    Reference:
            Zhong-Qiu Wang, Jonathan Le Roux, John R. Hershey 
            "ALTERNATIVE OBJECTIVE FUNCTIONS FOR DEEP CLUSTERING"
    """
    if spk_cnt is None:
        spk_cnt = len(tgt_index.unique())

    batch, bins, frame = tgt_index.shape
    tgt_embedding = torch.zeros(batch, frame*bins, spk_cnt)
    tgt_embedding.scatter_(2, tgt_index.view(batch, frame*bins,1),1)
    est_proj = torch.einsum('ijk,ijl->ikl', embedding, embedding)
    true_proj = torch.einsum('ijk,ijl->ikl', tgt_embedding, tgt_embedding)
    true_est_proj = torch.einsum('ijk,ijl->ikl', embedding, tgt_embedding)
    return batch_matrix_norm(est_proj) + batch_matrix_norm(true_proj) - \
            2*batch_matrix_norm(true_est_proj)


