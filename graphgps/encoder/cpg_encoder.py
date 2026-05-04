import torch.nn as nn
from torch_geometric.graphgym.register import (
    register_edge_encoder,
    register_node_encoder,
)
 
# Edge groups must match training exactly
EDGE_GROUPS = {
    "AST": 0,       "CONTAINS": 0,
    "CFG": 1,       "DOMINATE": 1,  "POST_DOMINATE": 1,
    "REACHING_DEF": 2,
    "CDG": 3,
    "CALL": 4,      "ARGUMENT": 4,  "PARAMETER_LINK": 4,
    "REF": 5,
}
NUM_EDGE_TYPES = len(set(EDGE_GROUPS.values()))   # 6
 
 
@register_node_encoder("CPGNode")
class CPGNodeEncoder(nn.Module):
    """Projects 128-dim hash node features → model embedding dim."""
 
    def __init__(self, emb_dim: int):
        super().__init__()
        self.in_dim = 128
        self.encoder = nn.Linear(self.in_dim, emb_dim)
 
    def forward(self, batch):
        if batch.x is None:
            raise ValueError("CPGNodeEncoder: batch.x is None")
        batch.x = self.encoder(batch.x.float())
        return batch
 
 
@register_edge_encoder("CPGEdge")
class CPGEdgeEncoder(nn.Module):
    """
    Encodes (edge_type, direction) pairs → edge embedding.
 
    batch.edge_attr shape: [E, 2]
        col 0 = edge group index  ∈ {0..5}
        col 1 = direction         ∈ {0=forward, 1=reverse}
    """
 
    def __init__(self, emb_dim: int):
        super().__init__()
        self.edge_emb = nn.Embedding(NUM_EDGE_TYPES, emb_dim)
        self.dir_emb  = nn.Embedding(2, emb_dim)
        nn.init.xavier_uniform_(self.edge_emb.weight)
        nn.init.xavier_uniform_(self.dir_emb.weight)
 
    def forward(self, batch):
        if batch.edge_attr is None:
            raise ValueError("CPGEdgeEncoder: batch.edge_attr is None")
 
        edge_type = batch.edge_attr[:, 0].long()
        direction = batch.edge_attr[:, 1].long()
 
        # Additive combination (matches training intent)
        batch.edge_attr = self.edge_emb(edge_type) + self.dir_emb(direction)
        return batch

# ===========
# import torch
# from torch_geometric.graphgym.register import (
#     register_node_encoder,
#     register_edge_encoder,
# )
 
# from graphgps.loader.dataset.jslibs import EDGE_GROUPS
 
# # =========================
# # Node Encoder for CPG
# # =========================
# @register_node_encoder('CPGNode')
# class CPGNodeEncoder(torch.nn.Module):
#     """
#     Node encoder for CPG graphs.
 
#     Assumes:
#         batch.x shape = [num_nodes, input_dim]
#         (e.g., 128-dim hash feature)
 
#     Projects node features → embedding space
#     """
 
#     def __init__(self, emb_dim):
#         super().__init__()
 
#         # ⚠️ MUST match your dataset
#         self.in_dim = 128
 
#         self.encoder = torch.nn.Linear(self.in_dim, emb_dim)
 
#     def forward(self, batch):
#         if batch.x is None:
#             raise ValueError("CPGNodeEncoder requires node features (batch.x)")
 
#         batch.x = self.encoder(batch.x.float())
#         return batch
 
 
# # =========================
# # Edge Encoder for CPG
# # =========================
# @register_edge_encoder('CPGEdge')
# class CPGEdgeEncoder(torch.nn.Module):
#     """
#     Edge encoder for CPG graphs.
 
#     Assumes:
#         batch.edge_attr shape = [num_edges, 1]
#         where value ∈ {0,1,2,3} (AST, CFG, PDG, CDG)
 
#     Converts edge type → embedding
#     """
 
#     def __init__(self, emb_dim):
#         super().__init__()
 
       
#         # self.encoder = torch.nn.Embedding(4, emb_dim) # AST, CFG, REACHING_DEF, CDG
#         # torch.nn.init.xavier_uniform_(self.encoder.weight.data)
 
#         num_edge_types = len(set(EDGE_GROUPS.values()))
#         self.edge_emb = torch.nn.Embedding(num_edge_types, emb_dim)
#         self.dir_emb = torch.nn.Embedding(2, emb_dim)    # forward/backward
 
#     def forward(self, batch):
#         edge_type = batch.edge_attr[:, 0]
#         direction = batch.edge_attr[:, 1]
 
#         if batch.edge_attr is None:
#             raise ValueError("CPGEdgeEncoder requires edge_attr")
 
#         batch.edge_attr = (
#             self.edge_emb(edge_type) +
#             self.dir_emb(direction)
#         )
 
#         batch.edge_attr = self.encoder(edge_type)
#         return batch
 