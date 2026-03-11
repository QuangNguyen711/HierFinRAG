"""
Contrastive Loss for TTGNN Training (Equation 4 from paper).

Loss function:
    Lcon = -log(exp(sim(hi, hp)/τ) / Σ exp(sim(hi, hn)/τ))

where:
    - hi: anchor node embedding
    - hp: positive node embedding (aligned text-table pair)
    - hn: negative node embeddings
    - τ: temperature parameter
    - sim: cosine similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for graph node embeddings.
    
    Based on Equation 4 in the paper:
    Lcon = -log(exp(sim(hi, hp)/τ) / Σvn∈N exp(sim(hi, hn)/τ))
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter τ for scaling similarities
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self, 
        anchor_embeddings: torch.Tensor,
        positive_indices: List[List[int]],
        negative_indices: List[List[int]],
        all_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            anchor_embeddings: Query embeddings [batch_size, embedding_dim]
            positive_indices: List of positive node indices for each query
            negative_indices: List of negative node indices for each query
            all_embeddings: All node embeddings [num_nodes, embedding_dim]
            
        Returns:
            Scalar loss value
        """
        batch_size = anchor_embeddings.shape[0]
        
        # Collect losses from all positive pairs
        losses = []
        
        # Normalize embeddings
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        all_embeddings_norm = F.normalize(all_embeddings, p=2, dim=1)
        
        for i in range(batch_size):
            anchor = anchor_embeddings[i]  # [embedding_dim]
            pos_idx = positive_indices[i]
            neg_idx = negative_indices[i]
            
            # Skip if empty (shouldn't happen after pre-validation, but be safe)
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue
            
            # Get positive and negative embeddings
            pos_embs = all_embeddings_norm[pos_idx]  # [num_pos, embedding_dim]
            neg_embs = all_embeddings_norm[neg_idx]  # [num_neg, embedding_dim]
            
            # Compute similarities
            pos_sim = torch.matmul(pos_embs, anchor)  # [num_pos]
            neg_sim = torch.matmul(neg_embs, anchor)  # [num_neg]
            
            # Apply temperature scaling
            pos_sim = pos_sim / self.temperature
            neg_sim = neg_sim / self.temperature
            
            # For each positive, compute contrastive loss using log-sum-exp trick
            for pos_s in pos_sim:
                # Combine positive with negatives for numerical stability
                logits = torch.cat([pos_s.unsqueeze(0), neg_sim])  # [1 + num_neg]
                
                # Loss = -log(exp(s_pos) / sum(exp(all)))
                #      = -s_pos + log_sum_exp(all)
                loss_i = -pos_s + torch.logsumexp(logits, dim=0)
                losses.append(loss_i)
        
        # Average over all positive pairs
        # After pre-validation, we should always have valid samples
        return torch.stack(losses).mean()


class InfoNCELoss(nn.Module):
    """
    Alternative: InfoNCE Loss (popular for contrastive learning).
    
    Similar to supervised contrastive but normalized differently.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            query_embeddings: [batch_size, embedding_dim]
            positive_embeddings: [batch_size, embedding_dim]
            negative_embeddings: [batch_size, num_negatives, embedding_dim]
            
        Returns:
            Scalar loss
        """
        # Normalize
        query = F.normalize(query_embeddings, p=2, dim=1)
        positive = F.normalize(positive_embeddings, p=2, dim=1)
        negative = F.normalize(negative_embeddings, p=2, dim=2)
        
        # Positive similarity
        pos_sim = torch.sum(query * positive, dim=1) / self.temperature  # [batch_size]
        
        # Negative similarities
        neg_sim = torch.matmul(negative, query.unsqueeze(2)).squeeze(2) / self.temperature  # [batch_size, num_negatives]
        
        # LogSumExp for numerical stability
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch_size, 1 + num_negatives]
        
        # InfoNCE: -log(exp(pos) / (exp(pos) + Σexp(neg)))
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss


class TripletMarginLoss(nn.Module):
    """
    Alternative: Triplet Margin Loss for ranking.
    
    Loss = max(0, margin + sim(anchor, negative) - sim(anchor, positive))
    """
    
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            anchor_embeddings: [batch_size, embedding_dim]
            positive_embeddings: [batch_size, embedding_dim]
            negative_embeddings: [batch_size, embedding_dim]
        """
        # Cosine similarity
        pos_sim = F.cosine_similarity(anchor_embeddings, positive_embeddings)
        neg_sim = F.cosine_similarity(anchor_embeddings, negative_embeddings)
        
        # Triplet loss
        loss = torch.clamp(self.margin + neg_sim - pos_sim, min=0.0)
        
        return loss.mean()
