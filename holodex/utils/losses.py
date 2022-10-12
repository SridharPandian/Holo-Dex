import torch
import torch.nn as nn
import torch.nn.functional as F

# VICReg losses
def compute_std_loss(rep, epsilon = 1e-04):
    rep = rep - rep.mean(dim = 0)
    rep_std = torch.sqrt(rep.var(dim = 0) + epsilon)
    return torch.mean(F.relu(1 - rep_std)) / 2.0

def off_diagonal(rep_cov):
    n, _ = rep_cov.shape
    return rep_cov.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def compute_cov_loss(rep, feature_size):
    rep_cov = (rep.T @ rep) / (rep.shape[0] - 1)
    return off_diagonal(rep_cov).pow_(2).sum().div(feature_size)


# Standard VICReg Loss
def vicreg_loss(input_rep, output_rep, feature_size, sim_coef, std_coef, cov_coef):
    sim_loss = F.mse_loss(input_rep, output_rep)
    std_loss = compute_std_loss(input_rep) + compute_std_loss(output_rep)
    cov_loss = compute_cov_loss(input_rep, feature_size) + compute_cov_loss(output_rep, feature_size)

    final_loss = (sim_coef * sim_loss) + (std_coef * std_loss) + (cov_coef * cov_loss)
    loss_dict = {
        'train_loss': final_loss.item(),
        'sim_loss': sim_loss.item(),
        'std_loss': std_loss.item(),
        'cov_loss': cov_loss.item()
    }

    return final_loss, loss_dict

# From sthalles SimCLR implementation: https://github.com/sthalles/SimCLR
def nt_xent_loss(queries, keys, temperature = 0.1):
    b, device = queries.shape[0], queries.device

    n = b * 2
    projs = torch.cat((queries, keys))
    logits = projs @ projs.t()

    mask = torch.eye(n, device=device).bool()
    logits = logits[~mask].reshape(n, n - 1)
    logits /= temperature

    labels = torch.cat(((torch.arange(b, device=device) + b - 1), torch.arange(b, device=device)), dim=0)
    loss = F.cross_entropy(logits, labels, reduction='sum')
    loss /= n
    return loss 