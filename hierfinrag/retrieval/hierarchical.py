import torch
import torch.nn.functional as F
import numpy as np
import re
from typing import List, Any, Dict, Tuple, Optional, Set
from torch_geometric.data import Data


class HierarchicalRetriever:
    """
    Two-stage hierarchical retrieval as described in the HierFinRAG paper:
    1. Stage 1: Retrieve top-K relevant sections using embedding similarity
    2. Stage 2: Extract subgraph from selected sections, run GNN, retrieve leaf nodes
    """
    
    def __init__(self, encoder_model=None, gnn_model=None):
        """
        Args:
            encoder_model: The embedding model (e.g., Vietnamese BERT)
            gnn_model: The TTGNN model for graph reasoning
        """
        self.encoder_model = encoder_model
        self.gnn_model = gnn_model
        
    def encode_query(self, query: str) -> torch.Tensor:
        """Encode query string to embedding vector."""
        if self.encoder_model is not None:
            return self.encoder_model.encode(query, convert_to_tensor=True)
        else:
            raise ValueError("Encoder model not initialized")
    
    def stage1_retrieve_sections(
        self,
        query: str,
        section_embeddings: torch.Tensor,
        section_metadata: List[Dict],
        top_k: int = 3
    ) -> List[Tuple[int, float, Dict]]:
        """
        Stage 1: Retrieve top-K relevant sections using embedding similarity.
        
        Args:
            query: Query string
            section_embeddings: Embeddings of section nodes [num_sections, D]
            section_metadata: Metadata for each section
            top_k: Number of sections to retrieve
            
        Returns:
            List of (section_idx, score, metadata) tuples
        """
        # Encode query
        query_emb = self.encode_query(query)
        
        # Normalize for cosine similarity
        query_emb_norm = F.normalize(query_emb.unsqueeze(0), p=2, dim=1)
        section_emb_norm = F.normalize(section_embeddings, p=2, dim=1)
        
        # Compute similarity
        similarities = torch.mm(query_emb_norm, section_emb_norm.t()).squeeze(0)
        
        # Get top-k
        top_k_actual = min(top_k, len(similarities))
        top_scores, top_indices = torch.topk(similarities, k=top_k_actual)
        
        results = []
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            if idx < len(section_metadata):
                results.append((idx, score, section_metadata[idx]))
        
        return results
    
    def extract_subgraph(
        self,
        selected_section_indices: List[int],
        graph: Data,
        node_metadata: List[Dict]
    ) -> Tuple[Data, List[int], Dict[int, int]]:
        """
        Extract a subgraph containing selected sections and their children.
        
        Args:
            selected_section_indices: Indices of sections to include
            graph: Full graph
            node_metadata: Metadata for all nodes
            
        Returns:
            subgraph: Extracted subgraph
            subgraph_node_indices: Original indices of nodes in subgraph
            old_to_new_idx: Mapping from original to subgraph indices
        """
        # Find all nodes that belong to selected sections
        subgraph_nodes = set(selected_section_indices)
        
        # Add all children of selected sections by traversing edges
        edge_index = graph.edge_index
        for sec_idx in selected_section_indices:
            # Find all outgoing edges from this section
            neighbors = edge_index[1][edge_index[0] == sec_idx].tolist()
            subgraph_nodes.update(neighbors)
            
            # For table nodes, also add their cells (grandchildren)
            for neighbor_idx in neighbors:
                if neighbor_idx < len(node_metadata):
                    if node_metadata[neighbor_idx]['type'] == 'Table':
                        # Add all cells of this table
                        table_neighbors = edge_index[1][edge_index[0] == neighbor_idx].tolist()
                        subgraph_nodes.update(table_neighbors)
        
        subgraph_node_indices = sorted(list(subgraph_nodes))
        old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(subgraph_node_indices)}
        
        # Extract node features
        subgraph_x = graph.x[subgraph_node_indices]
        subgraph_node_types = graph.node_types[subgraph_node_indices]
        
        # Extract edges within subgraph
        mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in subgraph_nodes and dst in subgraph_nodes:
                mask[i] = True
        
        subgraph_edge_index = edge_index[:, mask]
        subgraph_edge_attr = graph.edge_attr[mask]
        
        # Remap edge indices to new numbering
        for i in range(subgraph_edge_index.shape[1]):
            subgraph_edge_index[0, i] = old_to_new_idx[subgraph_edge_index[0, i].item()]
            subgraph_edge_index[1, i] = old_to_new_idx[subgraph_edge_index[1, i].item()]
        
        subgraph = Data(
            x=subgraph_x,
            edge_index=subgraph_edge_index,
            edge_attr=subgraph_edge_attr,
            node_types=subgraph_node_types
        )
        
        return subgraph, subgraph_node_indices, old_to_new_idx
    
    def stage2_retrieve_leafs(
        self,
        query: str,
        subgraph: Data,
        subgraph_node_indices: List[int],
        node_metadata: List[Dict],
        top_k: int = 5
    ) -> List[Tuple[int, float, Dict]]:
        """
        Stage 2: Run GNN on subgraph and retrieve top-K leaf nodes.
        
        Args:
            query: Query string
            subgraph: Extracted subgraph
            subgraph_node_indices: Original indices of nodes in subgraph
            node_metadata: Full node metadata list
            top_k: Number of leaf nodes to retrieve
            
        Returns:
            List of (original_node_idx, score, metadata) tuples
        """
        # Run GNN on subgraph if available
        if self.gnn_model is not None:
            with torch.no_grad():
                gnn_embeddings = self.gnn_model(
                    subgraph.x,
                    subgraph.edge_index,
                    subgraph.edge_attr,
                    subgraph.node_types
                )
        else:
            # Fallback to original embeddings
            gnn_embeddings = subgraph.x
        
        # Encode query
        query_emb = self.encode_query(query)
        
        # Both query and GNN embeddings are now 1024-dim (no projection needed)
        
        # Filter to only leaf nodes (Paragraphs and Cells)
        leaf_indices = []
        leaf_embeddings = []
        leaf_types = {'Paragraph': 0, 'Cell': 0}
        
        for i, orig_idx in enumerate(subgraph_node_indices):
            if orig_idx < len(node_metadata):
                node_type = node_metadata[orig_idx]['type']
                if node_type in ['Paragraph', 'Cell']:
                    leaf_indices.append(orig_idx)
                    leaf_embeddings.append(gnn_embeddings[i])
                    leaf_types[node_type] += 1
        
        if not leaf_embeddings:
            return []
        
        # Debug: print leaf composition
        print(f"    → Leaf nodes in subgraph: {leaf_types['Paragraph']} Paragraphs, {leaf_types['Cell']} Cells")
        
        leaf_embeddings = torch.stack(leaf_embeddings)
        
        # Compute similarity
        query_emb_norm = F.normalize(query_emb.unsqueeze(0), p=2, dim=1)
        leaf_emb_norm = F.normalize(leaf_embeddings, p=2, dim=1)
        
        similarities = torch.mm(query_emb_norm, leaf_emb_norm.t()).squeeze(0)
        
        # Get top-k
        top_k_actual = min(top_k, len(similarities))
        top_scores, top_indices = torch.topk(similarities, k=top_k_actual)
        
        results = []
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            orig_idx = leaf_indices[idx]
            results.append((orig_idx, score, node_metadata[orig_idx]))
        
        return results, leaf_types  # Return type distribution for debugging
    
    def     retrieve(
        self,
        query: str,
        graph: Data,
        node_metadata: List[Dict],
        top_k_sections: int = 3,
        top_k_leafs: int = 5
    ) -> Dict[str, List[Tuple[int, float, Dict]]]:
        """
        Full two-stage hierarchical retrieval.
        
        Args:
            query: Query string
            graph: Full document graph
            node_metadata: Metadata for all nodes
            top_k_sections: Number of sections to retrieve in stage 1
            top_k_leafs: Number of leaf nodes to retrieve in stage 2
            
        Returns:
            Dictionary with 'sections' and 'leafs' results
        """
        # Separate section nodes from full graph
        section_indices = []
        section_embeddings = []
        section_metadata = []
        
        for i, meta in enumerate(node_metadata):
            if meta['type'] == 'Section':
                section_indices.append(i)
                section_embeddings.append(graph.x[i])
                section_metadata.append(meta)
        
        if not section_embeddings:
            return {'sections': [], 'leafs': []}
        
        section_embeddings = torch.stack(section_embeddings)
        
        # Stage 1: Retrieve sections
        section_results = self.stage1_retrieve_sections(
            query, section_embeddings, section_metadata, top_k_sections
        )
        
        # Get actual section node indices in the full graph
        selected_section_indices = [section_indices[res[0]] for res in section_results]
        
        # Extract subgraph
        subgraph, subgraph_node_indices, old_to_new = self.extract_subgraph(
            selected_section_indices, graph, node_metadata
        )
        
        # Stage 2: Retrieve leaf nodes from subgraph
        leaf_results, leaf_type_dist = self.stage2_retrieve_leafs(
            query, subgraph, subgraph_node_indices, node_metadata, top_k_leafs
        )
        
        return {
            'sections': section_results,
            'leafs': leaf_results,
            'subgraph_size': len(subgraph_node_indices),
            'leaf_types': leaf_type_dist  # Include type distribution
        }


# Legacy functions kept for backward compatibility
def cosine_similarity(a, b):
    # Mock cosine similarity
    # a: [n, d], b: [m, d]
    if isinstance(a, torch.Tensor):
        a = a.detach().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().numpy()
    # Simplified
    return np.dot(a, b.T)

def reciprocal_rank_fusion(scores_list, k=60):
    # Simplified RRF for 2 lists of scores (assuming they are ranks or we convert to ranks)
    # The snippet implies we have `bm25_scores` and `dense_scores` which are raw scores.
    # We should convert them to ranks first.
    # Placeholder implementation
    return np.array(scores_list[0]) + np.array(scores_list[1]) # Very naive

def level1_retrieval(query_emb, section_embeddings, k=5):
    """
    Args:
        query_emb: Query embedding [1, d]
        section_embeddings: All section embeddings [num_sections, d]
        k: Number of sections to retrieve
    
    Returns:
        top_k_sections: Indices of most relevant sections
    """
    # Compute cosine similarity
    if isinstance(query_emb, np.ndarray):
        query_emb = torch.tensor(query_emb)
    if isinstance(section_embeddings, np.ndarray):
        section_embeddings = torch.tensor(section_embeddings)

    # Manual cosine sim if not normalized
    # Assuming normalized for dot product
    similarities = torch.mm(query_emb, section_embeddings.t())
    
    # Get top-k sections
    top_k_indices = torch.topk(similarities, k=k).indices
    
    return top_k_indices

def level2_retrieval(query, section_nodes, bm25_retriever, k=10):
    """
    Args:
        query: Query text
        section_nodes: List of [Paragraph, Table] nodes in selected sections
        k: Number of elements to retrieve
    
    Returns:
        top_k_elements: Most relevant paragraphs/tables
    """
    # Sparse retrieval (BM25)
    # bm25_scores = bm25_retriever.score(query, section_nodes)
    bm25_scores = np.random.rand(len(section_nodes)) # Placeholder
    
    # Dense retrieval (embedding similarity)
    # query_emb = embed_model.encode(query)
    # dense_scores = cosine_similarity(query_emb, [n.embedding for n in section_nodes])
    dense_scores = np.random.rand(len(section_nodes)) # Placeholder
    
    # Reciprocal Rank Fusion
    # combined_scores = reciprocal_rank_fusion(bm25_scores, dense_scores)
    combined_scores = bm25_scores + dense_scores # Placeholder
    
    # Get top-k
    top_k_indices = np.argsort(combined_scores)[-k:][::-1]
    
    return [section_nodes[i] for i in top_k_indices]

def select_relevant_rows(query, table, top_k=5):
    """
    Args:
        query: Query text
        table: Table object with rows
        top_k: Number of rows to select
    
    Returns:
        relevant_rows: Top-k rows by relevance
    """
    row_representations = []
    for row in table.rows:
        # Concatenate row header + all cell values
        # assuming row.cells is list of objects with .value
        # and row.header is string.
        cell_values = [getattr(c, 'value', str(c)) for c in row.cells]
        row_text = str(row.header) + " " + " ".join(cell_values)
        row_representations.append(row_text)
    
    # Embed and compute similarity
    query_emb = embed_model.encode(query)
    row_embs = embed_model.encode(row_representations)
    similarities = cosine_similarity(query_emb, row_embs)
    
    top_k_indices = np.argsort(similarities.flatten())[-top_k:][::-1]
    return [table.rows[i] for i in top_k_indices]

def select_relevant_columns(query, table, intent):
    """
    Args:
        query: Query text
        table: Table object
        intent: Financial intent (helps identify column type)
    
    Returns:
        relevant_columns: Columns likely to contain answer
    """
    if intent == "Numerical":
        # For specific year mentions, select that year's column
        year_match = re.search(r'(20\d{2})', query)
        if year_match:
            year = year_match.group(1)
            # Assuming table.columns is list of objects with .header
            matching_cols = [col for col in table.columns if year in col.header]
            if matching_cols:
                return matching_cols
        
        # Otherwise, select most recent period columns
        return table.get_latest_period_columns()
    
    elif intent == "Comparison":
        # Select multiple period columns
        return table.get_time_series_columns()
    
    else:
        # Default: all columns
        return table.columns

def extract_answer_cells(relevant_rows, relevant_columns, table):
    """
    Returns:
        cells: Intersection of relevant rows and columns
    """
    cells = []
    for row in relevant_rows:
        for col in relevant_columns:
            cell = table.get_cell(row.index, col.index)
            cells.append(cell)
    
    return cells
