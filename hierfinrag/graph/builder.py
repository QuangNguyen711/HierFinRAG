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
                
        # Stack features
        if node_features:
            x = torch.stack(node_features)
            node_types_tensor = torch.tensor(node_types, dtype=torch.long)
        else:
             x = torch.empty((0, self.embedding_dim))
             node_types_tensor = torch.empty((0,), dtype=torch.long)
        
        # 2. Create Edges
        edge_indices = [[], []]
        edge_attrs = []
        
        # Structural Edges (Section -> Content)
        for sec in doc.sections:
            if sec.id in node_ids_map:
                sec_idx = node_ids_map[sec.id]
                for content_id in sec.content_ids:
                    if content_id in node_ids_map:
                        content_idx = node_ids_map[content_id]
                        
                        # Parent -> Child
                        edge_indices[0].append(sec_idx)
                        edge_indices[1].append(content_idx)
                        edge_attrs.append(1) # struct
                        
                        # Child -> Parent
                        edge_indices[0].append(content_idx)
                        edge_indices[1].append(sec_idx)
                        edge_attrs.append(1)

        # Structure: Table -> Cell
        for table in doc.tables:
            if table.id in node_ids_map:
                table_idx = node_ids_map[table.id]
                for cell in table.cells:
                    cell_id = f"{table.id}_r{cell.row_idx}_c{cell.col_idx}"
                    if cell_id in node_ids_map:
                        cell_idx = node_ids_map[cell_id]
                        
                        edge_indices[0].append(table_idx)
                        edge_indices[1].append(cell_idx)
                        edge_attrs.append(1) # struct
                        
                        edge_indices[0].append(cell_idx)
                        edge_indices[1].append(table_idx)
                        edge_attrs.append(1)

        # Convert to tensor
        if edge_indices[0]:
            edge_index = torch.tensor(edge_indices, dtype=torch.long)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.long)
            
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, node_types=node_types_tensor)
