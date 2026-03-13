import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple
import numpy as np
from ..parsing.base import Document

class GraphBuilder:
    """
    Constructs a PyG graph from a parsed Document object.
    
    Node Types:
    0: Paragraph (P)
    1: Section (S)
    2: Table (T)
    3: Cell (C)
    
    Edge Types:
    0: Semantic (sem) - calculated via threshold
    1: Structural (struct) - explicit hierarchy
    2: Cross-Reference (ref) - explicit mentions
    """
    
    def __init__(self, embedding_dim=1024, use_real_embeddings=False):
        self.embedding_dim = embedding_dim
        self.use_real_embeddings = use_real_embeddings
        self.encoder_model = None
        
        if self.use_real_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                print("Loading Vietnamese embedding model: AITeamVN/Vietnamese_Embedding...")
                self.encoder_model = SentenceTransformer('AITeamVN/Vietnamese_Embedding')
                self.encoder_model.max_seq_length = 2048
                
                # Update embedding_dim to match model output
                self.embedding_dim = self.encoder_model.get_sentence_embedding_dimension()
                print(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            except Exception as e:
                print(f"Warning: Could not load embedding model: {e}")
                print("Falling back to random embeddings")
                self.use_real_embeddings = False

    def _encode(self, text: str) -> torch.Tensor:
        if self.use_real_embeddings and self.encoder_model is not None:
            # Use real Vietnamese embedding model
            embedding = self.encoder_model.encode(text, convert_to_tensor=True)
            return embedding
        else:
            # Fallback: random vector for demo
            return torch.randn(self.embedding_dim)

    def build_graph(self, doc: Document) -> Data:
        node_features = []
        node_types = []
        node_ids_map = {} # Map internal ID (e.g. "p_1") to node index (0, 1, ...)
        current_idx = 0
        
        # 1. Create Nodes
        
        # Sections
        for sec in doc.sections:
            node_features.append(self._encode(sec.title))
            node_types.append(1) # Section
            node_ids_map[sec.id] = current_idx
            current_idx += 1
            
        # Paragraphs
        for p in doc.paragraphs:
            node_features.append(self._encode(p.text))
            node_types.append(0) # Paragraph
            node_ids_map[p.id] = current_idx
            current_idx += 1
            
        # Tables
        for table in doc.tables:
            # Table node embedding: caption + headers
            table_text = f"{table.caption} {' '.join(table.col_headers)}"
            node_features.append(self._encode(table_text))
            node_types.append(2) # Table
            node_ids_map[table.id] = current_idx
            current_idx += 1
            
            # Cell nodes
            for cell in table.cells:
                # Cell ID convention: tableId_rX_cY
                cell_id = f"{table.id}_r{cell.row_idx}_c{cell.col_idx}"
                cell_text = str(cell.value)

                node_features.append(self._encode(cell_text))
                node_types.append(3) # Cell
                node_ids_map[cell_id] = current_idx
                current_idx += 1
                
        x = torch.stack(node_features)
        node_types_tensor = torch.tensor(node_types, dtype=torch.long)
        
        # 2. Create Edges
        edge_indices = [[], []]
        edge_attrs = []
        
        # A. Phân biệt chiều Structural Edges (Top-down: 1, Bottom-up: 2)
        # Section <-> Content
        for sec in doc.sections:
            s_idx = node_ids_map[sec.id]
            for c_id in sec.content_ids:
                if c_id in node_ids_map:
                    c_idx = node_ids_map[c_id]
                    # Top-down (Index 1)
                    edge_indices[0].append(s_idx); edge_indices[1].append(c_idx); edge_attrs.append(1)
                    # Bottom-up (Index 2)
                    edge_indices[0].append(c_idx); edge_indices[1].append(s_idx); edge_attrs.append(2)

        # Table <-> Cell
        for table in doc.tables:
            t_idx = node_ids_map[table.id]
            for cell in table.cells:
                c_id = f"{table.id}_r{cell.row_idx}_c{cell.col_idx}"
                if c_id in node_ids_map:
                    c_idx = node_ids_map[c_id]
                    # Top-down (Index 1)
                    edge_indices[0].append(t_idx); edge_indices[1].append(c_idx); edge_attrs.append(1)
                    # Bottom-up (Index 2)
                    edge_indices[0].append(c_idx); edge_indices[1].append(t_idx); edge_attrs.append(2)

        # B. Cạnh Ngữ nghĩa (Semantic Edges - Index 0)
        # Nối Paragraphs (0) với Cells (3) nếu tương đồng cao
        p_indices = (node_types_tensor == 0).nonzero(as_tuple=True)[0]
        c_indices = (node_types_tensor == 3).nonzero(as_tuple=True)[0]
        
        if len(p_indices) > 0 and len(c_indices) > 0:
            p_embs = F.normalize(x[p_indices], p=2, dim=1)
            c_embs = F.normalize(x[c_indices], p=2, dim=1)
            sim_matrix = torch.matmul(p_embs, c_embs.T)
            
            # Threshold 0.8 như paper đề xuất
            rel_pairs = (sim_matrix > 0.8).nonzero()
            for rel in rel_pairs:
                idx_p = p_indices[rel[0]]
                idx_c = c_indices[rel[1]]
                # Semantic nối 2 chiều (Index 0)
                edge_indices[0].extend([idx_p.item(), idx_c.item()])
                edge_indices[1].extend([idx_c.item(), idx_p.item()])
                edge_attrs.extend([0, 0])

        return Data(x=x, edge_index=torch.tensor(edge_indices, dtype=torch.long), 
                    edge_attr=torch.tensor(edge_attrs, dtype=torch.long), 
                    node_types=node_types_tensor)
