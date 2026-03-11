"""
Complete pipeline for generating training data and training TTGNN.

Usage:
    # Step 1: Generate training data
    python generate_and_train.py --mode generate --num_samples 1000
    
    # Step 2: Train TTGNN
    python generate_and_train.py --mode train --epochs 50
    
    # Step 3: Both (generate then train)
    python generate_and_train.py --mode both --num_samples 1000 --epochs 50
"""

import argparse
import os
import sys
import json
import torch
from pathlib import Path

from hierfinrag.parsing.json_parser import JSONParser
from hierfinrag.graph.builder import GraphBuilder
from hierfinrag.graph.ttgnn import TTGNN
from hierfinrag.training.data_generator import TrainingDataGenerator, load_training_data
from hierfinrag.training.trainer import TTGNNTrainer, TTGNNTrainingDataset
from hierfinrag.training.synthetic_doc_generator import SyntheticDocumentGenerator


def generate_training_data(
    num_samples: int = 1000,
    num_documents: int = 20,
    samples_per_doc: int = None,
    use_synthetic: bool = True,
    input_document: str = None,
    output_path: str = "data/training_data.json",
    synthetic_dir: str = "data/synthetic_documents"
):
    """
    Generate training data using LLM.
    
    Args:
        num_samples: Total number of training samples to generate
        num_documents: Number of synthetic documents to generate (if use_synthetic=True)
        samples_per_doc: Samples per document (auto-calculated if None)
        use_synthetic: Whether to generate synthetic documents
        input_document: Single document path (if use_synthetic=False)
        output_path: Where to save training data
        synthetic_dir: Directory for synthetic documents
    """
    print(f"\n{'='*80}")
    print(f"STEP 1: GENERATING TRAINING DATA")
    print(f"{'='*80}\n")
    
    parser = JSONParser()
    all_samples = []
    
    if use_synthetic:
        # Check for existing documents
        synthetic_path = Path(synthetic_dir)
        existing_docs = list(synthetic_path.glob("*.json")) if synthetic_path.exists() else []
        
        if num_documents == 0 and existing_docs:
            # Use existing documents without generating new ones
            print(f"[1.1] Using {len(existing_docs)} existing documents from {synthetic_dir}/")
            doc_files = [str(f) for f in existing_docs]
            print(f"✓ Found {len(doc_files)} existing documents")
        elif num_documents > 0:
            # Check if we should resume from existing documents
            start_index = 0
            if existing_docs:
                # Find highest index from existing documents
                max_index = -1
                for doc_path in existing_docs:
                    # Extract index from filename pattern: doc_xxx_yyy_ZZZ.json
                    try:
                        filename = doc_path.stem  # Without .json
                        parts = filename.split('_')
                        if len(parts) >= 4:
                            index = int(parts[-1])
                            max_index = max(max_index, index)
                    except:
                        continue
                
                if max_index >= 0:
                    start_index = max_index + 1
                    print(f"[1.1] Found {len(existing_docs)} existing documents (index 0-{max_index})")
                    print(f"      Generating {num_documents} NEW documents starting from index {start_index}...")
                else:
                    print(f"[1.1] Generating {num_documents} synthetic documents...")
            else:
                print(f"[1.1] Generating {num_documents} synthetic documents...")
            
            # Generate synthetic documents
            syn_generator = SyntheticDocumentGenerator(env_path=".env")
            doc_files = syn_generator.generate_dataset(
                num_documents=num_documents,
                output_dir=synthetic_dir,
                start_index=start_index
            )
            
            # Combine with existing documents
            if existing_docs:
                doc_files = [str(f) for f in existing_docs] + doc_files
                print(f"✓ Total documents: {len(existing_docs)} existing + {num_documents} new = {len(doc_files)}")
            
            if not doc_files:
                raise ValueError("No synthetic documents generated!")
        else:
            raise ValueError("No existing documents found and num_documents=0. Set num_documents > 0 to generate new documents.")
        
        # Calculate samples per document
        if samples_per_doc is None:
            samples_per_doc = max(1, num_samples // len(doc_files))
        
        print(f"\n[1.2] Generating {samples_per_doc} training samples per document...")
        print(f"      Total target: {len(doc_files)} docs × {samples_per_doc} samples = {len(doc_files) * samples_per_doc} samples\n")
        
        # Initialize question generator
        question_generator = TrainingDataGenerator(env_path=".env")
        
        # Generate samples from each document
        for i, doc_file in enumerate(doc_files, 1):
            print(f"\n{'─'*80}")
            print(f"Processing document {i}/{len(doc_files)}: {Path(doc_file).name}")
            print(f"{'─'*80}")
            
            try:
                # Parse document
                doc = parser.parse(doc_file)
                print(f"  ✓ Parsed: {doc.title}")
                print(f"    - {len(doc.sections)} sections, {len(doc.paragraphs)} paragraphs, {len(doc.tables)} tables")
                
                # Generate samples from this document
                doc_samples = question_generator.generate_dataset(
                    doc=doc,
                    num_samples=samples_per_doc,
                    output_path=None  # Don't save yet, we'll combine all
                )
                
                all_samples.extend(doc_samples)
                print(f"  ✓ Generated {len(doc_samples)} samples from this document")
                
            except Exception as e:
                print(f"  ✗ Error processing {doc_file}: {e}")
                continue
        
        print(f"\n{'='*80}")
        print(f"✓ Total samples generated: {len(all_samples)}")
        print(f"✓ From {len(doc_files)} documents")
        print(f"{'='*80}\n")
        
    else:
        # Use single document
        if not input_document:
            input_document = "data/mock_vietnamese_financial.json"
        
        print(f"Using single document: {input_document}")
        doc = parser.parse(input_document)
        print(f"✓ Loaded: {doc.title}")
        print(f"  - {len(doc.sections)} sections")
        print(f"  - {len(doc.paragraphs)} paragraphs")
        print(f"  - {len(doc.tables)} tables")
        
        # Initialize LLM data generator
        print("\nInitializing LLM generator...")
        generator = TrainingDataGenerator(env_path=".env")
        
        # Generate training data
        all_samples = generator.generate_dataset(
            doc=doc,
            num_samples=num_samples,
            output_path=None
        )
    
    # Save combined samples
    print(f"\nSaving {len(all_samples)} samples to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    from dataclasses import asdict
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(
            [asdict(s) for s in all_samples],
            f,
            ensure_ascii=False,
            indent=2
        )
    
    print(f"✓ Saved to {output_path}")
    
    return str(output_path)


def train_ttgnn(
    training_data_path: str = "data/training_data.json",
    document_path: str = "data/mock_vietnamese_financial.json",
    documents_dir: str = None,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    temperature: float = 0.07,
    hidden_dim: int = 256,
    num_layers: int = 2,
    num_heads: int = 8,
    save_dir: str = "models/ttgnn"
):
    """
    Train TTGNN model on generated data.
    
    Args:
        training_data_path: Path to training data JSON
        document_path: Path to single document JSON (ignored if documents_dir is provided)
        documents_dir: Directory containing multiple documents (preferred for multi-doc training)
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        temperature: Temperature for contrastive loss
        hidden_dim: Hidden dimension for TTGNN
        num_layers: Number of GNN layers
        num_heads: Number of attention heads
        save_dir: Directory to save model
    """
    print(f"\n{'='*80}")
    print(f"STEP 2: TRAINING TTGNN")
    print(f"{'='*80}\n")
    
    # Load training data
    print(f"Loading training data: {training_data_path}")
    training_samples = load_training_data(training_data_path)
    print(f"✓ Loaded {len(training_samples)} training samples")
    
    parser = JSONParser()
    
    # Determine which documents to use
    if documents_dir:
        # Load all documents from directory
        doc_files = sorted(Path(documents_dir).glob("*.json"))
        if not doc_files:
            raise ValueError(f"No JSON files found in {documents_dir}")
        
        print(f"\nLoading {len(doc_files)} documents from: {documents_dir}")
        documents = []
        for doc_file in doc_files:
            doc = parser.parse(str(doc_file))
            documents.append(doc)
            print(f"  ✓ {doc_file.name}: {doc.title}")
        
    else:
        # Load single document
        print(f"\nLoading single document: {document_path}")
        doc = parser.parse(document_path)
        documents = [doc]
        print(f"✓ Loaded: {doc.title}")
    
    # Build combined graph from all documents
    print(f"\nBuilding combined graph with Vietnamese embeddings from {len(documents)} document(s)...")
    builder = GraphBuilder(use_real_embeddings=True)
    
    # Build individual graphs and combine them
    all_node_features = []
    all_edge_indices = []
    all_edge_attrs = []
    all_node_types = []
    node_metadata = []
    doc_id_to_node_range = {}  # Maps document_id -> (start_idx, end_idx)
    current_node_offset = 0
    
    for doc_idx, doc in enumerate(documents):
        print(f"\n  Processing document {doc_idx+1}/{len(documents)}: {doc.title}")
        
        # Build graph for this document
        graph = builder.build_graph(doc)
        num_nodes = graph.x.shape[0]
        print(f"    → {num_nodes} nodes, {graph.edge_index.shape[1]} edges")
        
        # Track which nodes belong to this document
        doc_id_to_node_range[doc.id] = (current_node_offset, current_node_offset + num_nodes)
        
        # Append node features
        all_node_features.append(graph.x)
        
        # Adjust edge indices by offset and append
        adjusted_edges = graph.edge_index + current_node_offset
        all_edge_indices.append(adjusted_edges)
        
        # Append edge attributes
        all_edge_attrs.append(graph.edge_attr)
        
        # Append node types
        all_node_types.append(graph.node_types)
        
        # Create metadata for nodes with proper prefixing
        for sec in doc.sections:
            node_metadata.append({'id': sec.id, 'type': 'Section', 'text': sec.title})
        for p in doc.paragraphs:
            node_metadata.append({'id': p.id, 'type': 'Paragraph', 'text': p.text})
        for table in doc.tables:
            node_metadata.append({'id': table.id, 'type': 'Table', 'text': table.caption})
            for cell in table.cells:
                cell_id = f"{table.id}_r{cell.row_idx}_c{cell.col_idx}"
                node_metadata.append({'id': cell_id, 'type': 'Cell', 'text': str(cell.value)})
        
        current_node_offset += graph.x.shape[0]
    
    # Combine all graphs into one
    import torch
    from torch_geometric.data import Data
    
    combined_x = torch.cat(all_node_features, dim=0)
    combined_edge_index = torch.cat(all_edge_indices, dim=1)
    combined_edge_attr = torch.cat(all_edge_attrs, dim=0)
    combined_node_types = torch.cat(all_node_types, dim=0)
    
    combined_graph = Data(
        x=combined_x,
        edge_index=combined_edge_index,
        edge_attr=combined_edge_attr,
        node_types=combined_node_types
    )
    
    print(f"\n✓ Combined graph built: {combined_graph.x.shape[0]} total nodes, {combined_graph.edge_index.shape[1]} total edges")
    print(f"✓ Created metadata for {len(node_metadata)} nodes")
    print(f"✓ Document-to-node mapping: {len(doc_id_to_node_range)} documents")
    
    # Initialize TTGNN model
    print(f"\nInitializing TTGNN...")
    model = TTGNN(
        input_dim=combined_graph.x.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )
    print(f"✓ Model initialized:")
    print(f"  - Input dim: {combined_graph.x.shape[1]}")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - Layers: {num_layers}")
    print(f"  - Attention heads: {num_heads}")
    
    # Create training dataset
    print("\nPreparing training dataset...")
    train_dataset = TTGNNTrainingDataset(
        training_samples=training_samples,
        graph_data=combined_graph,
        node_metadata=node_metadata,
        query_encoder=builder.encoder_model,
        doc_id_to_node_range=doc_id_to_node_range
    )
    print(f"✓ Training dataset ready: {len(train_dataset)} samples")
    
    # Initialize trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = TTGNNTrainer(
        model=model,
        graph=combined_graph,
        node_metadata=node_metadata,
        query_encoder=builder.encoder_model,
        device=device,
        doc_id_to_node_range=doc_id_to_node_range
    )
    
    # Train model
    history = trainer.train(
        train_dataset=train_dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        temperature=temperature,
        save_dir=save_dir
    )
    
    # Save training history
    history_path = Path(save_dir) / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved to: {history_path}")
    
    return history


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data and train TTGNN"
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['generate', 'train', 'both'],
        default='both',
        help='Mode: generate data, train model, or both'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1000,
        help='Total number of training samples to generate'
    )
    
    parser.add_argument(
        '--num_documents',
        type=int,
        default=20,
        help='Number of synthetic documents to generate'
    )
    
    parser.add_argument(
        '--samples_per_doc',
        type=int,
        default=None,
        help='Samples per document (auto-calculated if not specified)'
    )
    
    parser.add_argument(
        '--use_synthetic',
        action='store_true',
        default=True,
        help='Generate synthetic documents (default: True)'
    )
    
    parser.add_argument(
        '--no_synthetic',
        action='store_false',
        dest='use_synthetic',
        help='Use single document instead of synthetic'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0005,
        help='Learning rate (default: 0.0005)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.2,
        help='Temperature for contrastive loss (default: 0.2, lower=harder, higher=softer)'
    )
    
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=768,
        help='Hidden dimension for TTGNN (must equal input_dim=768 for semantic alignment)'
    )
    
    parser.add_argument(
        '--document',
        type=str,
        default='data/mock_vietnamese_financial.json',
        help='Path to input document (used if --no_synthetic)'
    )
    
    parser.add_argument(
        '--training_data',
        type=str,
        default='data/training_data.json',
        help='Path to save/load training data'
    )
    
    parser.add_argument(
        '--synthetic_dir',
        type=str,
        default='data/synthetic_documents',
        help='Directory for synthetic documents'
    )
    
    parser.add_argument(
        '--save_dir',
        type=str,
        default='models/ttgnn',
        help='Directory to save model'
    )
    
    args = parser.parse_args()
    
    # Generate training data
    if args.mode in ['generate', 'both']:
        generate_training_data(
            num_samples=args.num_samples,
            num_documents=args.num_documents,
            samples_per_doc=args.samples_per_doc,
            use_synthetic=args.use_synthetic,
            input_document=args.document,
            output_path=args.training_data,
            synthetic_dir=args.synthetic_dir
        )
    
    # Train model
    if args.mode in ['train', 'both']:
        # For training with synthetic documents, use the entire directory
        # This builds a combined graph from all documents
        if args.use_synthetic and args.mode == 'train':
            print(f"\nTraining mode: Using all documents from {args.synthetic_dir}")
            train_ttgnn(
                training_data_path=args.training_data,
                documents_dir=args.synthetic_dir,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                temperature=args.temperature,
                hidden_dim=args.hidden_dim,
                save_dir=args.save_dir
            )
        else:
            # Single document mode
            train_document = args.document
            train_ttgnn(
                training_data_path=args.training_data,
                document_path=train_document,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                temperature=args.temperature,
                hidden_dim=args.hidden_dim,
                save_dir=args.save_dir
            )
    
    print(f"\n{'='*80}")
    print(f"✓ ALL STEPS COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
