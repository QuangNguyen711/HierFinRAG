import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv

class TTGNN(nn.Module):
    """
    Table-Text Graph Neural Network (TTGNN).
    
    Architecture:
    - Maintains input embedding dimension (1024) throughout
    - Node Type Embeddings: Distinct embeddings for P, S, T, C
    - Edge Type Embeddings: Incorporating edge relations (sem, struct, ref)
    - Graph Attention Layers: Relational attention mechanism
    - Output: Same dimension as input for direct similarity computation
    """
    def __init__(self, input_dim=1024, hidden_dim=1024, num_layers=2, num_heads=8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Keep embeddings at input dimension for semantic alignment
        assert hidden_dim == input_dim, "hidden_dim must equal input_dim to maintain semantic space"
        
        # 1. Node Type Embeddings (additive, not replacement)
        # 5 types: Paragraph, Section, Table, Cell, Global(optional)
        self.node_type_emb = nn.Embedding(5, hidden_dim)
        
        # 2. Edge Type Embeddings
        # 3 types: Semantic, Structural, Reference
        self.edge_type_emb = nn.Embedding(3, hidden_dim)
        
        # 3. GNN Layers (GATv2 with edge features)
        # Each layer outputs hidden_dim to maintain dimension
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=hidden_dim,
                    add_self_loops=True,
                    concat=True  # Concatenate heads: (hidden_dim/heads) * heads = hidden_dim
                )
            )
            
        # 4. Output Projection - refine but keep dimension
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index, edge_attr, node_types):
        """
        Args:
            x: Node features [N, 1024] - Vietnamese embeddings
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge type indices [E]
            node_types: Node type indices [N]
            
        Returns:
            Enhanced node embeddings [N, 1024] in same semantic space
        """
        # A. Initial Embedding Fusion (additive to preserve semantics)
        h = x + self.node_type_emb(node_types)
        
        # B. Edge Embedding Lookup
        edge_embeddings = self.edge_type_emb(edge_attr)
        
        # C. Message Passing with Residual Connections
        for layer in self.layers:
            h_in = h
            
            # GATv2 Layer
            h = layer(h, edge_index, edge_attr=edge_embeddings)
            
            # Activation & Dropout
            h = F.relu(h)
            h = self.dropout(h)
            
            # Residual (maintains dimension)
            h = h + h_in
            
        # D. Final Refinement (stays at 1024-dim)
        h = self.output_proj(h)
        
        return h
