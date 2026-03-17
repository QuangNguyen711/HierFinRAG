import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class TTGNN(nn.Module):
    """
    Table-Text Graph Neural Network (TTGNN).
    
    Architecture:
    - Maintains input embedding dimension (1024) throughout
    - Node Type Embeddings: Distinct embeddings for P, S, T, C, ColHeader, RowHeader (6 types)
    - Edge Type Embeddings: Incorporating edge relations (sem, struct-down, struct-up, etc.)
    - Graph Attention Layers: Relational attention mechanism with LayerNorm
    - Output: Residual connection with learnable gate to preserve Baseline performance
    """
    def __init__(self, input_dim=1024, hidden_dim=1024, num_layers=2, num_heads=8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Keep embeddings at input dimension for semantic alignment
        assert hidden_dim == input_dim, "hidden_dim must equal input_dim to maintain semantic space"
        
        # 1. Node Type Embeddings (additive)
        # 6 types: Paragraph(0), Section(1), Table(2), Cell(3), ColHeader(4), RowHeader(5)
        self.node_type_emb = nn.Embedding(6, hidden_dim)
        
        # 2. Edge Type Embeddings
        # 5 types: Semantic(0), Struct-Down(1), Struct-Up(2), Temporal(3), Accounting(4)
        self.edge_type_emb = nn.Embedding(5, hidden_dim)
        
        # 3. GNN Layers & Normalization
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() # [SỬA ĐỔI 1]: Thêm LayerNorm để chống kẹt Loss
        
        for _ in range(num_layers):
            self.layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=hidden_dim,
                    add_self_loops=True,
                    concat=True
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            
        # 4. Output Projection & Gate
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        # [SỬA ĐỔI 2]: Khởi tạo Gate = 0.0. Ở Epoch 1, GNN sẽ bị khóa hoàn toàn
        # giúp MRR bắt đầu chính xác bằng với MRR của Baseline BGE-M3.
        self.gate = nn.Parameter(torch.tensor([0.0])) 
        
    def forward(self, x, edge_index, edge_attr, node_types):
        """
        Args:
            x: Node features [N, 1024] - Vietnamese embeddings
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge type indices [E]
            node_types: Node type indices [N]
            
        Returns:
            Enhanced node embeddings [N, 1024]
        """
        # [SỬA ĐỔI 3]: Lưu lại vector gốc của BGE-M3
        x_base = x 
        
        # A. Fusion ban đầu: Text Embedding + Node Type Info
        h = x + self.node_type_emb(node_types)
        
        # B. Edge Embedding
        edge_embeddings = self.edge_type_emb(edge_attr)
        
        # C. Message Passing
        for i, layer in enumerate(self.layers):
            h_residual = h
            h = layer(h, edge_index, edge_attr=edge_embeddings)
            h = F.elu(h) # [SỬA ĐỔI 4]: Đổi thành ELU chuẩn theo paper
            h = self.dropout(h)
            
            # [SỬA ĐỔI 5]: Cộng Residual và đưa qua LayerNorm để ổn định phân phối
            h = self.norms[i](h + h_residual) 
            
        h = self.output_proj(h)
        
        # [SỬA ĐỔI 6]: Công thức lai (Hybrid) - Trả về Vector Gốc + Vector GNN học được
        return x_base + self.gate * h