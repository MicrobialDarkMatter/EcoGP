from sklearn import metrics
import torch


def calculate_metrics(y_true, y_pred):
    res = {}

    # AUC
    auc_per_species = [
        metrics.roc_auc_score(y_true[:, i], y_pred[:, i]) if not all(
            y_true[:, i] == 0) else float("nan") for i in
        range(y_true.shape[1])
    ]
    auc = torch.tensor(auc_per_species)
    auc = (auc[~torch.isnan(auc)]).mean().item()
    res["AUC"] = auc

    # NLL
    nll_per_species = [
        metrics.log_loss(y_true[:, i], y_pred[:, i], labels=[0.0, 1.0]) if not all(
            y_true[:, i] == 0) else float("nan") for i in
        range(y_true.shape[1])
    ]
    nll = torch.tensor(nll_per_species)
    nll = (nll[~torch.isnan(nll)]).mean().item()
    res["NLL"] = nll

    # MAE
    mae_per_species = [
        metrics.mean_absolute_error(y_true[:, i], y_pred[:, i]) if not all(
            y_true[:, i] == 0) else float("nan") for i in
        range(y_true.shape[1])
    ]
    mae = torch.tensor(mae_per_species)
    mae = (mae[~torch.isnan(mae)]).mean().item()
    res["MAE"] = mae

    # PR AUC
    pr_auc_per_species = [
        metrics.average_precision_score(y_true[:, i], y_pred[:, i]) if not all(
            y_true[:, i] == 0) else float("nan") for i in
        range(y_true.shape[1])
    ]
    pr_auc = torch.tensor(pr_auc_per_species)
    pr_auc = (pr_auc[~torch.isnan(pr_auc)]).mean().item()
    res["PR_AUC"] = pr_auc

    return res


def precision_at_k(labels, preds, k, probability_threshold=0.5):
    """

    :param labels:
    :param preds:
    :param k:
    :param probability_threshold:
    :return:
    """
    topk_indices = torch.topk(preds, k=k, dim=0).indices

    # p_k_pred = (torch.gather(input=preds, dim=0, index=topk_indices) >= probability_threshold).int()
    p_k_label = torch.gather(input=labels, dim=0, index=topk_indices)

    # p_k_species = (p_k_pred == p_k_label).sum(dim=0) / k
    p_k_species = p_k_label.mean(dim=0)
    p_k_avg = p_k_species.mean()

    return p_k_avg, p_k_species

