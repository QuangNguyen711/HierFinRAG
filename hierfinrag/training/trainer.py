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
    """Dataset for TTGNN training with per-document graphs."""
    
    def __init__(
        self, 
        training_samples: List[Dict[str, Any]],
        document_graphs: Dict[str, Any],
        document_metadata: Dict[str, List[Dict[str, Any]]],
        query_encoder: Any
    ):
        """
        Args:
            training_samples: List of training samples from data generator
            document_graphs: Dict mapping document_id -> PyG graph Data object
            document_metadata: Dict mapping document_id -> node metadata list
            query_encoder: Embedding model for encoding queries
        """
        self.document_graphs = document_graphs
        self.document_metadata = document_metadata
        self.query_encoder = query_encoder
        
        # Create per-document node_id to index mappings
        self.doc_node_id_to_idx = {}
        for doc_id, metadata in document_metadata.items():
            self.doc_node_id_to_idx[doc_id] = {
                meta['id']: i for i, meta in enumerate(metadata)
            }
        
        # Filter out invalid samples (documents or nodes not in graphs)
        print(f"Validating {len(training_samples)} training samples...")
        valid_samples = []
        invalid_count = 0
        missing_docs = set()
        
        for sample in training_samples:
            doc_id = sample.document_id
            
            # Check if document exists
            if doc_id not in self.document_graphs:
                missing_docs.add(doc_id)
                invalid_count += 1
                continue
            
            # Get node mapping for this document
            node_id_to_idx = self.doc_node_id_to_idx[doc_id]
            
            # Map node IDs to indices
            positive_indices = [
                node_id_to_idx[nid]
                for nid in sample.positive_nodes
                if nid in node_id_to_idx
            ]
            
            negative_indices = [
                node_id_to_idx[nid]
                for nid in sample.negative_nodes
                if nid in node_id_to_idx
            ]
            
            # Only keep samples with valid positive and negative nodes
            if len(positive_indices) > 0 and len(negative_indices) > 0:
                valid_samples.append(sample)
            else:
                invalid_count += 1
        
        self.samples = valid_samples

        print(f"Pre-computing embeddings for {len(self.samples)} queries...")
        all_queries = [s.query for s in self.samples]
        
        # Encode batch lớn (ví dụ 128) để tận dụng tối đa sức mạnh GPU H100
        self.query_embeddings = query_encoder.encode(
            all_queries, 
            batch_size=128, 
            show_progress_bar=True, 
            convert_to_tensor=True
        ).cpu() # Lưu vào CPU RAM để tránh tràn VRAM nếu data cực lớn
        
        if missing_docs:
            print(f"  ⚠ Missing documents: {sorted(missing_docs)}")
        if invalid_count > 0:
            print(f"  ⚠ Removed {invalid_count} invalid samples (missing document or nodes)")
        print(f"  ✓ {len(self.samples)} valid samples ready for training")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        doc_id = sample.document_id
        
        # Encode query
        query_embedding = self.query_embeddings[idx]
        
        # Get node mapping for this document
        node_id_to_idx = self.doc_node_id_to_idx[doc_id]
        
        # Map node IDs to indices (no prefixing needed - each graph is isolated)
        positive_indices = [
            node_id_to_idx[nid]
            for nid in sample.positive_nodes
            if nid in node_id_to_idx
        ]
        
        negative_indices = [
            node_id_to_idx[nid]
            for nid in sample.negative_nodes
            if nid in node_id_to_idx
        ]
        
        return {
            'query_embedding': query_embedding,
            'positive_indices': positive_indices,
            'negative_indices': negative_indices,
            'sample_id': sample.id,
            'document_id': doc_id
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
    """Trainer for TTGNN model with per-document graphs."""
    
    def __init__(
        self,
        model: TTGNN,
        document_graphs: Dict[str, Any],
        document_metadata: Dict[str, List[Dict[str, Any]]],
        query_encoder: Any,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: TTGNN model
            document_graphs: Dict mapping document_id -> graph
            document_metadata: Dict mapping document_id -> node metadata
            query_encoder: Query embedding model
            device: Training device
        """
        self.model = model.to(device)
        self.document_graphs = document_graphs
        self.document_metadata = document_metadata
        self.query_encoder = query_encoder
        self.device = device
        
        # Move all graphs to device
        for doc_id, graph in self.document_graphs.items():
            graph.x = graph.x.to(device)
            graph.edge_index = graph.edge_index.to(device)
            graph.edge_attr = graph.edge_attr.to(device)
            graph.node_types = graph.node_types.to(device)
        
        # Cache to store GNN outputs per document (for efficiency within an epoch)
        self._node_embeddings_cache = {}
        
        print(f"✓ Trainer initialized on device: {device}")
        print(f"✓ Managing {len(document_graphs)} separate document graphs")
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch with per-document graph processing."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Clear cache at start of epoch
        self._node_embeddings_cache = {}
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            query_embeddings = batch['query_embeddings'].to(self.device)
            positive_indices = batch['positive_indices']
            negative_indices = batch['negative_indices']
            document_ids = batch['document_ids']
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Process each unique document in the batch
            # For each document, run GNN once and collect node embeddings
            unique_docs = set(document_ids)
            doc_node_embeddings = {}
            
            for doc_id in unique_docs:
                graph = self.document_graphs[doc_id]
                
                # Run GNN forward pass for this document
                node_embeddings = self.model(
                    graph.x,
                    graph.edge_index,
                    graph.edge_attr,
                    graph.node_types
                )
                doc_node_embeddings[doc_id] = node_embeddings
            
            # Now compute loss for each sample in batch
            # Each sample uses only its document's node embeddings
            batch_losses = []
            
            for i in range(len(document_ids)):
                doc_id = document_ids[i]
                query_emb = query_embeddings[i:i+1]  # [1, 1024]
                pos_idx = positive_indices[i]
                neg_idx = negative_indices[i]
                node_embs = doc_node_embeddings[doc_id]

                # ====================================================
                # BẮT ĐẦU CHÈN LỌC NODE LÁ CHO TRAINING
                # ====================================================
                doc_node_types = self.document_graphs[doc_id].node_types
                leaf_mask = (doc_node_types == 0) | (doc_node_types == 3)
                
                # Cập nhật lại list Negative Indices chỉ chứa các node lá hợp lệ
                valid_neg_idx = [idx for idx in neg_idx if leaf_mask[idx]]
                
                if not valid_neg_idx:
                    # Nếu vì lý do nào đó sample này không có valid negative, bỏ qua
                    continue
                # ====================================================
                # KẾT THÚC CHÈN LỌC
                # ====================================================
                
                # Compute loss for this sample
                # Bỏ cái sim_scores_override đi, chỉ dùng valid_neg_idx
                sample_loss = loss_fn(
                    anchor_embeddings=query_emb,
                    positive_indices=[pos_idx],
                    negative_indices=[valid_neg_idx], 
                    all_embeddings=node_embs
                )
                batch_losses.append(sample_loss)
            
            # Average loss over batch
            loss = torch.stack(batch_losses).mean()
            
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
        val_dataset: TTGNNTrainingDataset = None, # Thêm tham số này
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

        val_dataloader = None
        if val_dataset:
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
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
        history = {'train_loss': [], 'val_loss': []} 
        best_mrr = 0.0 # MRR càng cao càng tốt (Max là 1.0)
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            # 1. Train 1 epoch
            train_metrics = self.train_epoch(dataloader, optimizer, loss_fn, epoch)
            history['train_loss'].append(train_metrics['loss'])
            
            # 2. Đánh giá trên tập Validation (nếu có)
            val_loss_val = train_metrics['loss'] # Mặc định dùng train loss nếu ko có val
            if val_dataloader:
                val_metrics = self.evaluate(val_dataloader, loss_fn)
                val_loss_val = val_metrics['val_loss']
                history['val_loss'].append(val_loss_val)
                
                print(f"Epoch {epoch:03d}/{num_epochs} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_loss_val:.4f}")
                print(f"  → TTGNN    | MRR: {val_metrics['mrr']:.4f} | R@5: {val_metrics['recall@5']:.4f} | R@10: {val_metrics['recall@10']:.4f}")
                print(f"  → Baseline | MRR: {val_metrics['base_mrr']:.4f} | R@5: {val_metrics['base_recall@5']:.4f} | R@10: {val_metrics['base_recall@10']:.4f}")
            else:
                print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_metrics['loss']:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # 3. Save best model dựa trên Val Loss
            current_mrr = val_metrics['mrr'] if val_dataloader else 0.0
            
            if val_dataloader and current_mrr > best_mrr:
                best_mrr = current_mrr
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss_val,
                    'mrr': best_mrr, # Lưu thêm mrr để sau này tiện xem
                    'train_loss': train_metrics['loss']
                }, save_path / 'best_model.pt')
                print(f"  → Saved best model (MRR: {best_mrr:.4f})")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss_val,
                    },
                    save_path / f'checkpoint_epoch_{epoch}.pt'
                )
            
            scheduler.step()
        
        print(f"\n{'='*80}")
        print(f"✓ Training completed!")
        print(f"Best validation MRR: {best_mrr:.4f}")
        print(f"Model saved to: {save_path}")
        print(f"{'='*80}\n")
        
        return history

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        loss_fn: nn.Module
    ) -> Dict[str, float]:
        """Evaluate the model on validation set using IR metrics (TTGNN vs Baseline RAG)."""
        self.model.eval()
        total_loss = 0.0
        
        # Tracking IR Metrics cho TTGNN
        total_mrr = 0.0
        total_recall_at_5 = 0.0
        total_recall_at_10 = 0.0
        
        # Tracking IR Metrics cho Baseline (Thuần Embedding)
        base_mrr = 0.0
        base_recall_at_5 = 0.0
        base_recall_at_10 = 0.0
        
        num_samples = 0
        
        for batch in dataloader:
            query_embeddings = batch['query_embeddings'].to(self.device)
            positive_indices = batch['positive_indices']
            negative_indices = batch['negative_indices']
            document_ids = batch['document_ids']
            
            unique_docs = set(document_ids)
            doc_node_embeddings = {}
            
            # Forward pass lấy Graph Embeddings
            for doc_id in unique_docs:
                graph = self.document_graphs[doc_id]
                node_embeddings = self.model(
                    graph.x, graph.edge_index, graph.edge_attr, graph.node_types
                )
                doc_node_embeddings[doc_id] = node_embeddings
            
            batch_losses = []
            
            for i in range(len(document_ids)):
                doc_id = document_ids[i]
                query_emb = query_embeddings[i:i+1] # [1, 1024]
                pos_idx = positive_indices[i]
                neg_idx = negative_indices[i]
                
                # Embeddings từ TTGNN
                node_embs = doc_node_embeddings[doc_id]
                # Embeddings gốc chưa qua GNN (Baseline RAG)
                base_embs = self.document_graphs[doc_id].x
                
                # 1. Tính Loss cho TTGNN
                sample_loss = loss_fn(
                    anchor_embeddings=query_emb,
                    positive_indices=[pos_idx],
                    negative_indices=[neg_idx],
                    all_embeddings=node_embs
                )
                batch_losses.append(sample_loss)
                
                # 2. Chuẩn hóa Vectors
                q_norm = torch.nn.functional.normalize(query_emb, p=2, dim=1)
                n_norm = torch.nn.functional.normalize(node_embs, p=2, dim=1)
                b_norm = torch.nn.functional.normalize(base_embs, p=2, dim=1)
                
                doc_node_types = self.document_graphs[doc_id].node_types
                leaf_mask = (doc_node_types == 0) | (doc_node_types == 3)
                sims = torch.matmul(q_norm, n_norm.T).squeeze(0)
                sims[~leaf_mask] = -1.0 
                sorted_indices = torch.argsort(sims, descending=True).cpu().tolist()
                
                first_hit_rank = 0
                for rank, node_idx in enumerate(sorted_indices, 1):
                    if node_idx in pos_idx:
                        first_hit_rank = rank
                        break
                
                if first_hit_rank > 0:
                    total_mrr += 1.0 / first_hit_rank
                
                top_5 = set(sorted_indices[:5])
                top_10 = set(sorted_indices[:10])
                pos_set = set(pos_idx)
                
                total_recall_at_5 += len(pos_set.intersection(top_5)) / len(pos_set)
                total_recall_at_10 += len(pos_set.intersection(top_10)) / len(pos_set)
                
                # ==========================================
                # LUỒNG 2: TÍNH METRICS CHO BASELINE RAG
                # ==========================================
                base_sims = torch.matmul(q_norm, b_norm.T).squeeze(0)
                base_sims[~leaf_mask] = -1.0
                base_sorted_indices = torch.argsort(base_sims, descending=True).cpu().tolist()
                
                base_first_hit_rank = 0
                for rank, node_idx in enumerate(base_sorted_indices, 1):
                    if node_idx in pos_idx:
                        base_first_hit_rank = rank
                        break
                        
                if base_first_hit_rank > 0:
                    base_mrr += 1.0 / base_first_hit_rank
                    
                base_top_5 = set(base_sorted_indices[:5])
                base_top_10 = set(base_sorted_indices[:10])
                
                base_recall_at_5 += len(pos_set.intersection(base_top_5)) / len(pos_set)
                base_recall_at_10 += len(pos_set.intersection(base_top_10)) / len(pos_set)
                
                num_samples += 1
            
            loss = torch.stack(batch_losses).mean()
            total_loss += loss.item() * len(document_ids) 
            
        return {
            'val_loss': total_loss / num_samples if num_samples > 0 else 0.0,
            'mrr': total_mrr / num_samples if num_samples > 0 else 0.0,
            'recall@5': total_recall_at_5 / num_samples if num_samples > 0 else 0.0,
            'recall@10': total_recall_at_10 / num_samples if num_samples > 0 else 0.0,
            'base_mrr': base_mrr / num_samples if num_samples > 0 else 0.0,
            'base_recall@5': base_recall_at_5 / num_samples if num_samples > 0 else 0.0,
            'base_recall@10': base_recall_at_10 / num_samples if num_samples > 0 else 0.0
        }


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
