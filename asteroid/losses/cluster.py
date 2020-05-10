import torch


def deep_clustering_loss(embedding, tgt_index, binary_mask=None):
    """ Compute the deep clustering loss defined in [1].

    Args:
        embedding (torch.Tensor): Estimated embeddings.
            Expected shape  (batch, frequency x frame, embedding_dim)
        tgt_index (torch.Tensor): Dominating source index in each TF bin.
            Expected shape: [batch, frequency, frame]
        binary_mask (torch.Tensor): VAD in TF plane. Bool or Float.
            See asteroid.filterbanks.transforms.ebased_vad.

    Returns:
         `torch.Tensor`. Deep clustering loss for every batch sample.

    Examples:
        >>> import torch
        >>> from asteroid.losses.cluster import deep_clustering_loss
        >>> spk_cnt = 3
        >>> embedding = torch.randn(10, 5*400, 20)
        >>> targets = torch.LongTensor([10, 400, 5]).random_(0, spk_cnt)
        >>> loss = deep_clustering_loss(embedding, targets)

    Reference:
        [1] Zhong-Qiu Wang, Jonathan Le Roux, John R. Hershey
            "ALTERNATIVE OBJECTIVE FUNCTIONS FOR DEEP CLUSTERING"

    Notes:
        Be careful in viewing the embedding tensors. The target indices
        `tgt_index` are of shape (batch, freq, frames). Even if the embedding
        is of shape (batch, freq*frames, emb), the underlying view should be
        (batch, freq, frames, emb) and not (batch, frames, freq, emb).
    """
    spk_cnt = len(tgt_index.unique())

    batch, bins, frames = tgt_index.shape
    if binary_mask is None:
        binary_mask = torch.ones(batch, bins * frames, 1)
    binary_mask = binary_mask.float()
    if len(binary_mask.shape) == 3:
        binary_mask = binary_mask.view(batch, bins * frames, 1)
    # If boolean mask, make it float.
    binary_mask = binary_mask.to(tgt_index.device)

    # Fill in one-hot vector for each TF bin
    tgt_embedding = torch.zeros(batch, bins * frames, spk_cnt,
                                device=tgt_index.device)
    tgt_embedding.scatter_(2, tgt_index.view(batch, bins * frames, 1), 1)

    # Compute VAD-weighted DC loss
    tgt_embedding = tgt_embedding * binary_mask
    embedding = embedding * binary_mask
    est_proj = torch.einsum('ijk,ijl->ikl', embedding, embedding)
    true_proj = torch.einsum('ijk,ijl->ikl', tgt_embedding, tgt_embedding)
    true_est_proj = torch.einsum('ijk,ijl->ikl', embedding, tgt_embedding)
    # Equation (1) in [1]
    cost = batch_matrix_norm(est_proj) + batch_matrix_norm(true_proj)
    cost = cost - 2 * batch_matrix_norm(true_est_proj)
    # Divide by number of active bins, for each element in batch
    return cost / torch.sum(binary_mask, dim=[1, 2])


def batch_matrix_norm(matrix, norm_order=2):
    """ Normalize a matrix according to `norm_order`

    Args:
        matrix (torch.Tensor): Expected shape [batch, *]
        norm_order (int): Norm order.

    Returns:
        torch.Tensor, normed matrix of shape [batch]
    """
    keep_batch = list(range(1, matrix.ndim))
    return torch.norm(matrix, p=norm_order, dim=keep_batch) ** norm_order
