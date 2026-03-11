"""
Script to inspect training data samples.
Shows questions with their document context and positive/negative nodes.
"""

import json
from pathlib import Path
from hierfinrag.parsing.json_parser import JSONParser


def inspect_training_samples(
    training_data_path="data/training_data.json",
    documents_dir="data/synthetic_documents",
    num_samples=3
):
    """Inspect training samples to verify correctness."""
    
    # Load training data
    print("="*80)
    print("INSPECTING TRAINING DATA")
    print("="*80)
    
    with open(training_data_path, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    print(f"\nTotal samples: {len(training_data)}")
    
    # Load documents
    parser = JSONParser()
    documents = {}
    doc_files = list(Path(documents_dir).glob("*.json"))
    
    print(f"Loading {len(doc_files)} documents...")
    for doc_file in doc_files:
        doc = parser.parse(str(doc_file))
        documents[doc.id] = doc
    
    print(f"Loaded {len(documents)} documents")
    print()
    
    # Select samples to inspect
    samples_to_inspect = training_data[:num_samples]
    
    for idx, sample in enumerate(samples_to_inspect, 1):
        print("\n" + "="*80)
        print(f"SAMPLE #{idx}")
        print("="*80)
        
        doc_id = sample['document_id']
        
        # Check if document exists
        if doc_id not in documents:
            print(f"❌ ERROR: Document '{doc_id}' not found in loaded documents!")
            print(f"   Available documents: {list(documents.keys())[:5]}...")
            continue
        
        doc = documents[doc_id]
        
        # Display basic info
        print(f"\n📄 Document ID: {doc_id}")
        print(f"   Title: {doc.title}")
        print(f"   Sections: {len(doc.sections)}")
        print(f"   Paragraphs: {len(doc.paragraphs)}")
        print(f"   Tables: {len(doc.tables)}")
        
        # Display question
        print(f"\n❓ Question:")
        print(f"   {sample['query']}")
        
        # Build node content map
        node_content = {}
        
        # Sections
        for sec in doc.sections:
            node_content[sec.id] = {
                'type': 'Section',
                'text': sec.title
            }
        
        # Paragraphs
        for p in doc.paragraphs:
            node_content[p.id] = {
                'type': 'Paragraph',
                'text': p.text[:200] + ('...' if len(p.text) > 200 else '')
            }
        
        # Tables
        for table in doc.tables:
            node_content[table.id] = {
                'type': 'Table',
                'text': table.caption
            }
            # Cells
            for cell in table.cells:
                cell_id = f"{table.id}_r{cell.row_idx}_c{cell.col_idx}"
                node_content[cell_id] = {
                    'type': 'Cell',
                    'text': f"{str(cell.value)} (row {cell.row_idx}, col {cell.col_idx})"
                }
        
        # Display positive nodes
        print(f"\n✅ Positive Nodes ({len(sample['positive_nodes'])} nodes):")
        for i, node_id in enumerate(sample['positive_nodes'][:5], 1):  # Show first 5
            if node_id in node_content:
                node = node_content[node_id]
                print(f"   {i}. [{node['type']}] {node_id}")
                print(f"      Content: {node['text']}")
            else:
                print(f"   {i}. ❌ {node_id} - NOT FOUND IN DOCUMENT")
        
        if len(sample['positive_nodes']) > 5:
            print(f"   ... and {len(sample['positive_nodes']) - 5} more")
        
        # Display negative nodes (sample)
        print(f"\n❌ Negative Nodes ({len(sample['negative_nodes'])} nodes, showing first 3):")
        for i, node_id in enumerate(sample['negative_nodes'][:3], 1):
            if node_id in node_content:
                node = node_content[node_id]
                print(f"   {i}. [{node['type']}] {node_id}")
                print(f"      Content: {node['text']}")
            else:
                print(f"   {i}. ❌ {node_id} - NOT FOUND IN DOCUMENT")
        
        # Verification
        print(f"\n🔍 Verification:")
        pos_found = sum(1 for nid in sample['positive_nodes'] if nid in node_content)
        neg_found = sum(1 for nid in sample['negative_nodes'] if nid in node_content)
        
        print(f"   ✓ Positive nodes in document: {pos_found}/{len(sample['positive_nodes'])}")
        print(f"   ✓ Negative nodes in document: {neg_found}/{len(sample['negative_nodes'])}")
        
        if pos_found == 0:
            print(f"   ⚠️  WARNING: No positive nodes found in document!")
        if neg_found == 0:
            print(f"   ⚠️  WARNING: No negative nodes found in document!")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    total_valid = 0
    total_invalid = 0
    missing_docs = set()
    
    for sample in training_data:
        doc_id = sample['document_id']
        
        if doc_id not in documents:
            missing_docs.add(doc_id)
            total_invalid += 1
            continue
        
        doc = documents[doc_id]
        
        # Build node set
        valid_nodes = set()
        for sec in doc.sections:
            valid_nodes.add(sec.id)
        for p in doc.paragraphs:
            valid_nodes.add(p.id)
        for table in doc.tables:
            valid_nodes.add(table.id)
            for cell in table.cells:
                cell_id = f"{table.id}_r{cell.row_idx}_c{cell.col_idx}"
                valid_nodes.add(cell_id)
        
        # Check if positive nodes exist
        pos_found = sum(1 for nid in sample['positive_nodes'] if nid in valid_nodes)
        neg_found = sum(1 for nid in sample['negative_nodes'] if nid in valid_nodes)
        
        if pos_found > 0 and neg_found > 0:
            total_valid += 1
        else:
            total_invalid += 1
    
    print(f"\nTotal samples: {len(training_data)}")
    print(f"Valid samples: {total_valid}")
    print(f"Invalid samples: {total_invalid}")
    
    if missing_docs:
        print(f"\nMissing documents ({len(missing_docs)}):")
        for doc_id in sorted(missing_docs)[:10]:
            print(f"  - {doc_id}")
        if len(missing_docs) > 10:
            print(f"  ... and {len(missing_docs) - 10} more")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect training data samples")
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to inspect in detail')
    parser.add_argument('--training_data', type=str, default='data/training_data.json', help='Path to training data')
    parser.add_argument('--documents_dir', type=str, default='data/synthetic_documents', help='Directory with documents')
    
    args = parser.parse_args()
    
    inspect_training_samples(
        training_data_path=args.training_data,
        documents_dir=args.documents_dir,
        num_samples=args.num_samples
    )
