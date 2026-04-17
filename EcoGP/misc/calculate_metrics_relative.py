import torch


def calculate_metrics_relative(y_true, y_pred, k):
    res = {
        "Precision": precision_at_k(y_true, y_pred, k),
        "NDCG": ndcg_at_k(y_true, y_pred, k),
        "CORR": spearman_corr(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
    }

    return res


def ndcg_at_k(y_true: torch.Tensor, y_score: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute NDCG@k for predictions.

    Args:
        y_true: relevance labels (batch_size, n_items)
        y_score: predicted scores (batch_size, n_items)
        k: cutoff

    Returns:
        ndcg: tensor of shape (batch_size,)
    """
    # sort by predicted score
    _, indices = torch.topk(y_score, k, dim=1)

    # get true relevance at top-k
    gains = torch.gather(y_true, 1, indices)

    # discounted cumulative gain (DCG)
    discounts = torch.log2(torch.arange(k, device=y_true.device).float() + 2.0)
    dcg = (gains / discounts).sum(dim=1)

    # ideal DCG (IDCG)
    _, ideal_indices = torch.topk(y_true, k, dim=1)
    ideal_gains = torch.gather(y_true, 1, ideal_indices)
    idcg = (ideal_gains / discounts).sum(dim=1)

    # avoid division by zero
    ndcg = dcg / torch.clamp(idcg, min=1e-8)
    return ndcg


def precision_at_k(y_true, y_pred, k=5):
    """
    Compute Precision@k for each row of (y_true, y_pred) tensors.
    Works fully in PyTorch, supports GPU.

    Args:
        y_true (torch.Tensor): Binary ground truth, shape (n_samples, n_labels)
        y_pred (torch.Tensor): Prediction scores, shape (n_samples, n_labels)
        k (int): The number of top elements to consider

    Returns:
        torch.Tensor: scalar tensor with mean Precision@k across rows
    """
    y_true = y_true.bool().float()
    y_pred = y_pred.float()

    # Get indices of top-k predictions per row
    topk_idx = torch.topk(y_pred, k, dim=1).indices

    # Gather corresponding true labels
    topk_true = torch.gather(y_true, 1, topk_idx)

    # Compute precision per row: number of true positives in top-k / k
    precision_per_row = topk_true.sum(dim=1) / k

    # Return mean precision@k (ignores NaNs if any row has no positives)
    return precision_per_row.nanmean()


def spearman_corr(x, y, dim=1):
    """
    Compute Spearman rank correlation between two tensors along a given dimension.

    Args:
        x, y: Tensors of the same shape
        dim: Dimension along which to compute correlation

    Returns:
        Spearman correlation tensor
    """
    # Rank along the given dimension
    x_rank = torch.argsort(torch.argsort(x, dim=dim), dim=dim).float()
    y_rank = torch.argsort(torch.argsort(y, dim=dim), dim=dim).float()

    # Mean centering
    x_rank = x_rank - x_rank.mean(dim=dim, keepdim=True)
    y_rank = y_rank - y_rank.mean(dim=dim, keepdim=True)

    # Compute Pearson correlation on ranks
    cov = (x_rank * y_rank).sum(dim=dim)
    x_std = torch.sqrt((x_rank ** 2).sum(dim=dim))
    y_std = torch.sqrt((y_rank ** 2).sum(dim=dim))

    return cov / (x_std * y_std)


def rmse(pred, target, dim=1):
    """
    Compute Root Mean Squared Error (RMSE) between pred and target.

    Args:
        pred: Predicted tensor
        target: Ground truth tensor
        dim: Dimension(s) to reduce over, if None reduces all elements

    Returns:
        RMSE value or tensor along specified dimension
    """
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=dim))
