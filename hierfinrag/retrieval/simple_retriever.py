import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from ..parsing.base import Document

class SimpleRetriever:
    """
    Simple retrieval based on cosine similarity between query and node embeddings.
    Works with the graph embeddings from GraphBuilder.
    """
    
    def __init__(self, encoder_model=None):
        """
        Args:
            encoder_model: The same embedding model used in GraphBuilder
        """
        self.encoder_model = encoder_model
        
    def encode_query(self, query: str) -> torch.Tensor:
        """Encode a query string into an embedding vector."""
        if self.encoder_model is not None:
            return self.encoder_model.encode(query, convert_to_tensor=True)
        else:
            raise ValueError("Encoder model not initialized")
    
    def retrieve(
        self, 
        query: str, 
        node_embeddings: torch.Tensor, 
        node_metadata: List[Dict],
        top_k: int = 5
    ) -> List[Tuple[int, float, Dict]]:
        """
        Retrieve top-k most relevant nodes for a query.
        
        Args:
            query: Query string
            node_embeddings: Tensor of shape [N, D] containing node embeddings
            node_metadata: List of dicts with node information (id, type, text, etc.)
            top_k: Number of top results to return
            
        Returns:
            List of tuples (node_idx, similarity_score, metadata)
        """
        # Encode query
        query_emb = self.encode_query(query)
        
        # Normalize embeddings for cosine similarity
        query_emb_norm = F.normalize(query_emb.unsqueeze(0), p=2, dim=1)
        node_emb_norm = F.normalize(node_embeddings, p=2, dim=1)
        
        # Compute cosine similarity
        similarities = torch.mm(query_emb_norm, node_emb_norm.t()).squeeze(0)
        
        # Get top-k indices
        top_k = min(top_k, len(similarities))
        top_scores, top_indices = torch.topk(similarities, k=top_k)
        
        # Prepare results
        results = []
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            if idx < len(node_metadata):
                results.append((idx, score, node_metadata[idx]))
        
        return results
    
    def retrieve_with_gnn_embeddings(
        self,
        query: str,
        gnn_embeddings: torch.Tensor,
        node_metadata: List[Dict],
        top_k: int = 5
    ) -> List[Tuple[int, float, Dict]]:
        """
        Retrieve using GNN-enhanced embeddings.
        Projects query embedding to match GNN output dimension if needed.
        """
        # Encode query
        query_emb = self.encode_query(query)
        
        # Check if dimensions match
        query_dim = query_emb.shape[0]
        gnn_dim = gnn_embeddings.shape[1]
        
        if query_dim != gnn_dim:
            # Simple linear projection using mean pooling for dimension matching
            # In practice, you'd want a learned projection layer
            if query_dim > gnn_dim:
                # Reduce dimension by reshaping and averaging
                query_emb = query_emb[:gnn_dim]
            else:
                # Increase dimension by padding with zeros
                padding = torch.zeros(gnn_dim - query_dim, device=query_emb.device)
                query_emb = torch.cat([query_emb, padding])
        
        # Normalize embeddings for cosine similarity
        query_emb_norm = F.normalize(query_emb.unsqueeze(0), p=2, dim=1)
        node_emb_norm = F.normalize(gnn_embeddings, p=2, dim=1)
        
        # Compute cosine similarity
        similarities = torch.mm(query_emb_norm, node_emb_norm.t()).squeeze(0)
        
        # Get top-k indices
        top_k_actual = min(top_k, len(similarities))
        top_scores, top_indices = torch.topk(similarities, k=top_k_actual)
        
        # Prepare results
        results = []
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            if idx < len(node_metadata):
                results.append((idx, score, node_metadata[idx]))
        
        return results
