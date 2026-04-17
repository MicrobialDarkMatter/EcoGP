import torch


def calculate_metrics(y_true, y_pred):
    res = {
        "AUC": calculate_roc_auc(y_true, y_pred),
        "NLL": calculate_nll(y_true, y_pred),
        "MAE": calculate_mae(y_true, y_pred),
        "PR_AUC": calculate_pr_auc(y_true, y_pred),
    }

    return res


def calculate_roc_auc(y_true, y_pred):
    """
    Compute AUC for each column of 2D tensors (y_true, y_pred) using ranks.
    Works fully in PyTorch, supports GPU.
    """
    y_true = y_true.float()
    y_pred = y_pred.float()

    # Sort predictions and reorder labels
    sorted_idx = torch.argsort(y_pred, dim=0)
    sorted_true = torch.gather(y_true, 0, sorted_idx)

    # Count positives and negatives
    pos = torch.sum(sorted_true, dim=0)
    neg = sorted_true.shape[0] - pos

    # Compute rank sums
    rank = torch.arange(1, sorted_true.shape[0] + 1, device=y_true.device).unsqueeze(1)
    rank_sum = torch.sum(rank * sorted_true, dim=0)

    auc = (rank_sum - pos * (pos + 1) / 2) / (pos * neg)
    return auc.mean().item()


def calculate_pr_auc(y_true, y_pred, eps=1e-8):
    """
    Compute PR AUC (area under precision–recall curve) for each column of 2D tensors.
    Fully in PyTorch, GPU-compatible, no loops.

    Args:
        y_true: (N, C) binary labels
        y_pred: (N, C) predicted scores
        eps: numerical stability

    Returns:
        pr_auc: (C,) tensor of PR AUCs
    """
    y_true = y_true.float()
    y_pred = y_pred.float()

    # Sort predictions descending
    sorted_idx = torch.argsort(y_pred, dim=0, descending=True)
    sorted_true = torch.gather(y_true, 0, sorted_idx)

    # Cumulative sums of TP and FP
    tp = torch.cumsum(sorted_true, dim=0)
    fp = torch.cumsum(1 - sorted_true, dim=0)

    # Compute precision and recall
    precision = tp / (tp + fp + eps)
    recall = tp / (torch.sum(y_true, dim=0, keepdim=True) + eps)

    # Add (0,1) at start for integration
    precision = torch.cat([torch.ones(1, y_true.shape[1], device=y_true.device), precision], dim=0)
    recall = torch.cat([torch.zeros(1, y_true.shape[1], device=y_true.device), recall], dim=0)

    # Compute area under PR curve using trapezoidal rule
    # Note: recall is monotonic increasing
    d_recall = recall[1:] - recall[:-1]
    pr_auc = torch.sum(precision[1:] * d_recall, dim=0)

    return pr_auc.mean().item()


def calculate_nll(y_true, y_pred, eps=1e-4):
    """
    Compute mean binary negative log-likelihood (NLL) per column.

    Args:
        y_true: (N, C) binary labels
        y_pred: (N, C) predicted probabilities (after sigmoid)
        eps: small constant for numerical stability

    Returns:
        nll: (C,) tensor of NLL values (lower = better)
    """
    y_true = y_true.float()
    y_pred = torch.clamp(y_pred, eps, 1 - eps)

    nll = -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    return nll.item()


def calculate_mae(y_true, y_pred):
    mae = torch.mean(torch.abs(y_true - y_pred))
    return mae.item()
