"""
graphgps/loss/supcon_loss.py

Supervised Contrastive Loss for graph embeddings.
Encourages embeddings of the same class to be close, different classes to be apart.

Based on: "Supervised Contrastive Learning for Pre-trained Graph Neural Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for graph classification.

    Args:
        temperature: Controls how hard the contrastive penalty is.
                    Higher = softer assignment, lower = harder assignment.
                    Typical values: 0.07 to 0.5
        base_temperature: Initial temperature, defaults to 0.07
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [B, D] - L2-normalized embeddings
            labels: [B] - class labels

        Returns:
            scalar loss
        """
        device = embeddings.device
        B = embeddings.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()  # [B, B] - same class = 1

        # Compute cosine similarity
        # embeddings are already L2-normalized, so dot product = cosine similarity
        logits = torch.matmul(embeddings, embeddings.T) / self.temperature  # [B, B]

        # For numerical stability, subtract max per row
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Mask out self-comparisons (diagonal)
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        # Count how many positive pairs we have per anchor
        exp_logits = torch.exp(logits) * logits_mask  # [B, B]
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Compute mean of log-likelihood over positive
        mask_sum = mask.sum(1)  # [B]
        mask_sum[mask_sum == 0] = 1  # avoid division by zero

        pos_log_prob = (mask * log_prob).sum(1) / mask_sum
        loss = pos_log_prob

        # Scale by temperature
        loss = loss * self.temperature / self.base_temperature
        loss = loss.mean()

        return loss


@register_loss('supcon')
def supcon_loss(pred, true):
    """
    Wrapper for SupConLoss to match GraphGPS loss interface.

    Args:
        pred: tuple of (embeddings, labels) from prototype head
              OR (embeddings, None) if labels come from batch.y
        true: batch.y labels

    Returns:
        loss, pred_scores (for logging)
    """
    if cfg.model.loss_fun == 'supcon':
        embeddings, labels = pred

        if labels is None:
            # Labels should come from batch.y - they need to be passed differently
            # In prototype head, we return (embeddings, labels) to this loss
            # If we get None here, it means we're being called incorrectly
            raise ValueError(
                "SupCon loss requires labels to be passed as second element of pred tuple. "
                "Ensure PrototypeHead returns (embeddings, labels) and train_epoch passes this."
            )

        crit = SupConLoss(
            temperature=getattr(cfg, 'loss.supcon_temperature', 0.07),
            base_temperature=getattr(cfg, 'loss.supcon_base_temperature', 0.07)
        )
        loss = crit(embeddings, labels)

        # Return dummy pred_score for logging (CE-style expects [N, C] but we return [N])
        pred_score = embeddings  # Use embeddings as proxy for logging
        return loss, pred_score