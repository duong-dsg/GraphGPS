"""
graphgps/head/prototype_head.py

Prototype-based head for JSLibs library detection.
Uses supervised contrastive loss for training and cosine-similarity-to-prototypes
for inference.

Training:
  - Computes graph embeddings from encoder
  - Applies SupCon loss to pull same-class embeddings together

Inference:
  - Compares embeddings to class prototypes (mean of train embeddings per class)
  - Returns top-k libraries by cosine similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.register import register_head


@register_head('prototype')
class PrototypeHead(nn.Module):
    """
    Head that produces embeddings for SupCon loss and stores prototypes.

    Args:
        dim_in: Input feature dimension (after GNN encoder)
        dim_out: Output embedding dimension
        num_classes: Number of library classes
        normalize: L2-normalize output embeddings (required for SupCon + cosine sim)
        prototype_update_freq: How often to update prototypes (every N epochs)
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        num_classes,
        normalize=True,
        prototype_update_freq=1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.normalize = normalize
        self.prototype_update_freq = prototype_update_freq

        # Projection head: dim_in → dim_out
        self.projection = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, dim_out),
        )

        # Storage for class prototypes
        self.register_buffer('prototypes', torch.zeros(num_classes, dim_out))
        self.prototypes_updated = False

        # Track support set for prototype computation
        self._support_embeddings = [[] for _ in range(num_classes)]
        self._support_labels = []

    def forward(self, batch):
        """
        Args:
            batch: PyG batch with batch.x, batch.edge_index, batch.batch

        Returns:
            (embeddings, labels) tuple for SupCon loss
        """
        # Get embeddings from the model's graph pooling
        # The GPS model should return pooled graph embeddings in batch.x after MP
        if hasattr(batch, 'x') and batch.x is not None:
            embeddings = batch.x
        else:
            raise ValueError("Batch missing x attribute")

        # Apply projection
        embeddings = self.projection(embeddings)

        # L2 normalize for SupCon and cosine similarity
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Get labels
        labels = batch.y.view(-1) if batch.y is not None else None

        return embeddings, labels

    def update_prototypes(self, embeddings, labels):
        """
        Update class prototypes with new embeddings.

        Args:
            embeddings: [N, D] embeddings from current batch
            labels: [N] class labels
        """
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                class_embeddings = embeddings[mask]
                if len(self._support_embeddings[c]) == 0:
                    self._support_embeddings[c] = class_embeddings.detach().cpu()
                else:
                    self._support_embeddings[c] = torch.cat([
                        self._support_embeddings[c],
                        class_embeddings.detach().cpu()
                    ], dim=0)

    def compute_prototypes(self):
        """
        Compute final prototypes from collected support embeddings.

        Returns:
            prototypes: [num_classes, D] tensor of class prototypes
        """
        for c in range(self.num_classes):
            if len(self._support_embeddings[c]) > 0:
                stacked = self._support_embeddings[c]
                prototype = stacked.mean(dim=0)
                if self.normalize:
                    prototype = F.normalize(prototype, p=2, dim=0)
                self.prototypes[c] = prototype
            else:
                # No support examples for this class - leave as zeros
                pass

        self.prototypes_updated = True
        return self.prototypes

    def predict(self, embeddings, topk=5, threshold=0.0):
        """
        Predict library labels using cosine similarity to prototypes.

        Args:
            embeddings: [N, D] embeddings
            topk: Number of top predictions to return
            threshold: Minimum cosine similarity to consider a match

        Returns:
            dict with 'predictions': list of (lib_idx, score) sorted by score
        """
        if not self.prototypes_updated:
            self.compute_prototypes()

        # Compute cosine similarity: [N, num_classes]
        # embeddings and prototypes are L2-normalized, so dot product = cosine
        similarity = torch.mm(embeddings, self.prototypes.to(embeddings.device))

        # Get topk predictions
        top_scores, top_indices = torch.topk(similarity, k=min(topk, self.num_classes), dim=1)

        results = []
        for i in range(embeddings.shape[0]):
            row_scores = top_scores[i].cpu().numpy()
            row_indices = top_indices[i].cpu().numpy()

            preds = []
            for score, lib_idx in zip(row_scores, row_indices):
                if score >= threshold:
                    preds.append((int(lib_idx), float(score)))
                else:
                    preds.append((int(lib_idx), float(score)))  # still return but mark low conf

            results.append({
                'topk': preds,
                'best_lib': preds[0][0] if preds else -1,
                'best_score': preds[0][1] if preds else 0.0,
            })

        return results

    def reset_prototypes(self):
        """Clear stored support embeddings and prototypes."""
        self._support_embeddings = [[] for _ in range(self.num_classes)]
        self.prototypes.zero_()
        self.prototypes_updated = False

    def get_prototypes(self):
        """Return current prototypes tensor."""
        return self.prototypes


@register_head('prototype_linear')
class PrototypeLinearHead(nn.Module):
    """
    Simpler prototype head that directly maps to class embeddings.
    No projection MLP, just L2-normalized linear layer.

    Suitable when the encoder already produces good embeddings.
    """

    def __init__(self, dim_in, dim_out, num_classes, normalize=True):
        super().__init__()
        self.num_classes = num_classes
        self.normalize = normalize

        # Direct linear mapping to embeddings
        self.linear = nn.Linear(dim_in, dim_out, bias=False)

        # Initialize weights for uniform angular distribution
        nn.init.xavier_uniform_(self.linear.weight, gain=1.0)

        # Class prototypes
        self.register_buffer('prototypes', torch.zeros(num_classes, dim_out))

    def forward(self, batch):
        embeddings = self.linear(batch.x)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        labels = batch.y.view(-1) if batch.y is not None else None
        return embeddings, labels

    def update_prototypes(self, embeddings, labels):
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                class_emb = embeddings[mask]
                stacked = torch.cat([self.prototypes[c:c+1].expand(len(class_emb), -1), class_emb], dim=0)
                self.prototypes[c] = stacked.mean(dim=0)
                if self.normalize:
                    self.prototypes[c] = F.normalize(self.prototypes[c], p=2, dim=0)

    def predict(self, embeddings, topk=5, threshold=0.0):
        similarity = torch.mm(embeddings, self.prototypes.to(embeddings.device))
        top_scores, top_indices = torch.topk(similarity, k=min(topk, self.num_classes), dim=1)

        results = []
        for i in range(embeddings.shape[0]):
            row_scores = top_scores[i].cpu().numpy()
            row_indices = top_indices[i].cpu().numpy()
            preds = [(int(lib_idx), float(score)) for score, lib_idx in zip(row_scores, row_indices)]
            results.append({
                'topk': preds,
                'best_lib': preds[0][0] if preds else -1,
                'best_score': preds[0][1] if preds else 0.0,
            })
        return results

    def reset_prototypes(self):
        self.prototypes.zero_()

    def get_prototypes(self):
        return self.prototypes