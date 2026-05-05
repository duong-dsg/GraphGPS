from typing import Dict

import torch.nn as nn
from torch_geometric.graphgym.register import (
    register_edge_encoder,
    register_node_encoder,
)
 
# Constants
EDGE_GROUPS: Dict[str, int] = {
    # Syntax
    "AST":            0,
    "CONTAINS":       0,
    # Control flow
    "CFG":            1,
    "DOMINATE":       1,
    "POST_DOMINATE":  1,
    # Data flow
    "REACHING_DEF":   2,
    # Control dependence
    "CDG":            3,
    # Call / argument
    "CALL":           4,
    "ARGUMENT":       4,
    "PARAMETER_LINK": 4,
    # Reference
    "REF":            5,
}
NUM_EDGE_GROUPS  = len(set(EDGE_GROUPS.values()))   # 6
NODE_FEATURE_DIM = 128
 
 
# =============================================================================
# Node encoder
# =============================================================================

@register_node_encoder("CPGNode")
class CPGNodeEncoder(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.proj = nn.Linear(NODE_FEATURE_DIM, emb_dim)
 
    def forward(self, batch):
        if batch.x is None:
            raise ValueError("CPGNodeEncoder: batch.x is None")
        batch.x = self.proj(batch.x.float())
        return batch


# =============================================================================
# Edge encoder
# =============================================================================

@register_edge_encoder("CPGEdge")
class CPGEdgeEncoder(nn.Module):
    """
    edge_attr [:, 0] = group id  ∈ {0…5}
    edge_attr [:, 1] = direction ∈ {0=forward, 1=reverse}
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        self.edge_emb = nn.Embedding(NUM_EDGE_GROUPS, emb_dim)
        self.dir_emb  = nn.Embedding(2,               emb_dim)
        nn.init.xavier_uniform_(self.edge_emb.weight)
        nn.init.xavier_uniform_(self.dir_emb.weight)
 
    def forward(self, batch):
        if batch.edge_attr is None:
            raise ValueError("CPGEdgeEncoder: batch.edge_attr is None")
        batch.edge_attr = (self.edge_emb(batch.edge_attr[:, 0])
                         + self.dir_emb (batch.edge_attr[:, 1]))
        return batch
