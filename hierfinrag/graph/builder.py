import torch
import torch.nn.functional as F
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
        node_ids_map = {}
        current_idx = 0
        
        # Lưu lại text để xử lý Semantic Edges ở cuối
        para_texts = {}
        cell_values = {}
        
        # --- 1. TẠO NODES ---
        # Sections (Type 1)
        for sec in doc.sections:
            node_features.append(self._encode(sec.title))
            node_types.append(1); node_ids_map[sec.id] = current_idx; current_idx += 1
            
        # Paragraphs (Type 0)
        for p in doc.paragraphs:
            node_features.append(self._encode(p.text))
            node_types.append(0); node_ids_map[p.id] = current_idx
            para_texts[current_idx] = p.text # Lưu lại để match
            current_idx += 1
            
        # Bóc tách Table thành Grid Graph
        for table in doc.tables:
            # Table Root Node (Type 2)
            t_idx = current_idx
            caption = table.caption if hasattr(table, 'caption') and table.caption else "Bảng"
            node_features.append(self._encode(caption))
            node_types.append(2); node_ids_map[table.id] = current_idx; current_idx += 1
            
            # Node Tiêu đề Cột (Type 4)
            col_headers_exist = hasattr(table, 'col_headers') and table.col_headers
            if col_headers_exist:
                for c_idx, col_head in enumerate(table.col_headers):
                    ch_id = f"{table.id}_ch_{c_idx}"
                    node_features.append(self._encode(str(col_head)))
                    node_types.append(4); node_ids_map[ch_id] = current_idx; current_idx += 1
                    
            # Node Tiêu đề Hàng (Type 5)
            row_headers_exist = hasattr(table, 'row_headers') and table.row_headers
            if row_headers_exist:
                for r_idx, row_head in enumerate(table.row_headers):
                    rh_id = f"{table.id}_rh_{r_idx}"
                    node_features.append(self._encode(str(row_head)))
                    node_types.append(5); node_ids_map[rh_id] = current_idx; current_idx += 1
            
            # Node Cell (Type 3) - CHỈ CHỨA VALUE THUẦN
            for cell in table.cells:
                cell_id = f"{table.id}_r{cell.row_idx}_c{cell.col_idx}"
                cell_val_str = str(cell.value)
                node_features.append(self._encode(cell_val_str))
                node_types.append(3); node_ids_map[cell_id] = current_idx
                cell_values[current_idx] = cell_val_str # Lưu lại để match
                current_idx += 1
                
        x = torch.stack(node_features)
        node_types_tensor = torch.tensor(node_types, dtype=torch.long)

        # --- 2. TẠO EDGES ---
        edge_indices = [[], []]
        edge_attrs = []
        
        def add_edge(src, dst, e_type):
            edge_indices[0].append(src); edge_indices[1].append(dst); edge_attrs.append(e_type)

        # A. Cạnh phân cấp Section <-> Content (1: Down, 2: Up)
        for sec in doc.sections:
            s_idx = node_ids_map[sec.id]
            for c_id in sec.content_ids:
                if c_id in node_ids_map:
                    c_idx = node_ids_map[c_id]
                    add_edge(s_idx, c_idx, 1); add_edge(c_idx, s_idx, 2)

        # B. Cạnh Grid Graph cho Table
        for table in doc.tables:
            t_idx = node_ids_map[table.id]
            
            # Nối Table Root với các Header
            if hasattr(table, 'col_headers') and table.col_headers:
                for c_idx in range(len(table.col_headers)):
                    ch_idx = node_ids_map[f"{table.id}_ch_{c_idx}"]
                    add_edge(t_idx, ch_idx, 1); add_edge(ch_idx, t_idx, 2)
                    
            if hasattr(table, 'row_headers') and table.row_headers:
                for r_idx in range(len(table.row_headers)):
                    rh_idx = node_ids_map[f"{table.id}_rh_{r_idx}"]
                    add_edge(t_idx, rh_idx, 1); add_edge(rh_idx, t_idx, 2)

            # Nối Header với Cell
            for cell in table.cells:
                c_idx = node_ids_map[f"{table.id}_r{cell.row_idx}_c{cell.col_idx}"]
                has_any_header = False
                
                # Col Header trỏ xuống Cell
                ch_id = f"{table.id}_ch_{cell.col_idx}"
                if ch_id in node_ids_map:
                    add_edge(node_ids_map[ch_id], c_idx, 1); add_edge(c_idx, node_ids_map[ch_id], 2)
                    has_any_header = True
                    
                # Row Header trỏ ngang sang Cell
                rh_id = f"{table.id}_rh_{cell.row_idx}"
                if rh_id in node_ids_map:
                    add_edge(node_ids_map[rh_id], c_idx, 1); add_edge(c_idx, node_ids_map[rh_id], 2)
                    has_any_header = True
                    
                # Nếu bảng không có header nào, nối thẳng Cell vào Table Root
                if not has_any_header:
                    add_edge(t_idx, c_idx, 1); add_edge(c_idx, t_idx, 2)

        # C. Cạnh Ngữ nghĩa (Semantic Edges - Index 0) THEO CHUẨN PAPER
        # Dùng Exact Matching để tìm Cell được nhắc đến trong Paragraph
        for p_idx, p_text in para_texts.items():
            for c_idx, c_val in cell_values.items():
                # Điều kiện: Value phải đủ dài (tránh match số 0, 1) và nằm trong đoạn văn
                if len(c_val) >= 3 and c_val in p_text:
                    add_edge(p_idx, c_idx, 0)
                    add_edge(c_idx, p_idx, 0)

        return Data(x=x, edge_index=torch.tensor(edge_indices, dtype=torch.long), 
                    edge_attr=torch.tensor(edge_attrs, dtype=torch.long), 
                    node_types=node_types_tensor)
