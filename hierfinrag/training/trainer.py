"""
TTGNN Training Script with Supervised Contrastive Learning.

Trains the Table-Text Graph Neural Network using:
1. Query encoder (Vietnamese embedding model)
2. TTGNN on document graph
3. Supervised contrastive loss (Equation 4 from paper)
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import numpy as np

from ..graph.ttgnn import TTGNN
from ..graph.builder import GraphBuilder
from ..parsing.json_parser import JSONParser
from .contrastive_loss import SupervisedContrastiveLoss


class TTGNNTrainingDataset(Dataset):
    """Dataset for TTGNN training."""
    
    def __init__(
        self, 
        training_samples: List[Dict[str, Any]],
        graph_data: Any,
        node_metadata: List[Dict[str, Any]],
        query_encoder: Any,
        doc_id_to_node_range: Dict[str, Tuple[int, int]] = None
    ):
        """
        Args:
            training_samples: List of training samples from data generator
            graph_data: PyG graph Data object
            node_metadata: Metadata for all nodes
            query_encoder: Embedding model for encoding queries
            doc_id_to_node_range: Mapping of document_id -> (start_idx, end_idx) in combined graph
        """
        self.samples = training_samples
        self.graph = graph_data
        self.node_metadata = node_metadata
        self.query_encoder = query_encoder
        self.doc_id_to_node_range = doc_id_to_node_range or {}
        
        # Create node_id to index mapping
        self.node_id_to_idx = {meta['id']: i for i, meta in enumerate(node_metadata)}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Encode query (access dataclass attributes, not dict keys)
        query_embedding = self.query_encoder.encode(
            sample.query, 
            convert_to_tensor=True
        )
        
        # Map node IDs to indices
        positive_indices = [
            self.node_id_to_idx[nid] 
            for nid in sample.positive_nodes 
            if nid in self.node_id_to_idx
        ]
        
        negative_indices = [
            self.node_id_to_idx[nid] 
            for nid in sample.negative_nodes 
            if nid in self.node_id_to_idx
        ]
        
        return {
            'query_embedding': query_embedding,
            'positive_indices': positive_indices,
            'negative_indices': negative_indices,
            'sample_id': sample.id,
            'document_id': sample.document_id
        }


def collate_fn(batch):
    """Custom collate function for batching."""
    return {
        'query_embeddings': torch.stack([item['query_embedding'] for item in batch]),
        'positive_indices': [item['positive_indices'] for item in batch],
        'negative_indices': [item['negative_indices'] for item in batch],
        'sample_ids': [item['sample_id'] for item in batch],
        'document_ids': [item['document_id'] for item in batch]
    }


class TTGNNTrainer:
    """Trainer for TTGNN model."""
    
    def __init__(
        self,
        model: TTGNN,
        graph: Any,
        node_metadata: List[Dict[str, Any]],
        query_encoder: Any,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        doc_id_to_node_range: Dict[str, Tuple[int, int]] = None
    ):
        """
        Args:
            model: TTGNN model
            graph: Document graph
            node_metadata: Node metadata
            query_encoder: Query embedding model
            device: Training device
            doc_id_to_node_range: Mapping of document_id -> (start_idx, end_idx) in combined graph
        """
        self.model = model.to(device)
        self.graph = graph
        self.node_metadata = node_metadata
        self.query_encoder = query_encoder
        self.device = device
        self.doc_id_to_node_range = doc_id_to_node_range or {}
        
        # Move graph to device
        self.graph.x = self.graph.x.to(device)
        self.graph.edge_index = self.graph.edge_index.to(device)
        self.graph.edge_attr = self.graph.edge_attr.to(device)
        self.graph.node_types = self.graph.node_types.to(device)
        
        print(f"✓ Trainer initialized on device: {device}")
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            query_embeddings = batch['query_embeddings'].to(self.device)
            positive_indices = batch['positive_indices']
            negative_indices = batch['negative_indices']
            document_ids = batch['document_ids']
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute node embeddings (forward pass through GNN)
            node_embeddings = self.model(
                self.graph.x,
                self.graph.edge_index,
                self.graph.edge_attr,
                self.graph.node_types
            )
            
            # Filter indices to only include nodes from the respective documents
            # This ensures each sample only sees nodes from its own document
            if self.doc_id_to_node_range:
                filtered_positive_indices = []
                filtered_negative_indices = []
                
                for i, doc_id in enumerate(document_ids):
                    if doc_id in self.doc_id_to_node_range:
                        start_idx, end_idx = self.doc_id_to_node_range[doc_id]
                        
                        # Filter positive indices to be within document range
                        valid_pos = [idx for idx in positive_indices[i] 
                                   if start_idx <= idx < end_idx]
                        filtered_positive_indices.append(valid_pos)
                        
                        # Filter negative indices to be within document range
                        valid_neg = [idx for idx in negative_indices[i] 
                                   if start_idx <= idx < end_idx]
                        filtered_negative_indices.append(valid_neg)
                    else:
                        # No filtering if document not in mapping (fallback)
                        filtered_positive_indices.append(positive_indices[i])
                        filtered_negative_indices.append(negative_indices[i])
                
                positive_indices = filtered_positive_indices
                negative_indices = filtered_negative_indices
            
            # Compute loss (queries and nodes are both 768-dim now)
            loss = loss_fn(
                anchor_embeddings=query_embeddings,
                positive_indices=positive_indices,
                negative_indices=negative_indices,
                all_embeddings=node_embeddings
            )
            
            # Skip batch if loss is invalid (e.g., no valid pairs after filtering)
            if not isinstance(loss, torch.Tensor) or torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss in batch, skipping...")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'loss': avg_loss
        }
    
    def train(
        self,
        train_dataset: TTGNNTrainingDataset,
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        save_dir: str = "models/ttgnn"
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_dataset: Training dataset
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            temperature: Temperature for contrastive loss
            save_dir: Directory to save model checkpoints
            
        Returns:
            Training history
        """
        print(f"\n{'='*80}")
        print(f"Starting TTGNN Training")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Temperature: {temperature}")
        print(f"{'='*80}\n")
        
        # Create dataloader
        dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        # Initialize optimizer and loss
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        loss_fn = SupervisedContrastiveLoss(temperature=temperature)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs
        )
        
        # Training history
        history = {
            'train_loss': []
        }
        
        best_loss = float('inf')
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            metrics = self.train_epoch(dataloader, optimizer, loss_fn, epoch)
            
            history['train_loss'].append(metrics['loss'])
            
            print(f"Epoch {epoch}/{num_epochs} - Loss: {metrics['loss']:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                    },
                    save_path / 'best_model.pt'
                )
                print(f"  → Saved best model (loss: {best_loss:.4f})")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': metrics['loss'],
                    },
                    save_path / f'checkpoint_epoch_{epoch}.pt'
                )
            
            scheduler.step()
        
        print(f"\n{'='*80}")
        print(f"✓ Training completed!")
        print(f"Best loss: {best_loss:.4f}")
        print(f"Model saved to: {save_path}")
        print(f"{'='*80}\n")
        
        return history


def load_model(
    model: TTGNN,
    checkpoint_path: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> TTGNN:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    return model
