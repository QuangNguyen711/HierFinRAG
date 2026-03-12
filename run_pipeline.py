import os
import sys
import json
import torch
from hierfinrag.parsing.json_parser import JSONParser
from hierfinrag.graph.builder import GraphBuilder
from hierfinrag.graph.ttgnn import TTGNN
from hierfinrag.reasoning.fusion import SymbolicNeuralFusion
from hierfinrag.retrieval.hierarchical import HierarchicalRetriever

def main():
    print("Starting HierFinRAG Vietnamese Financial Document Pipeline...")
    print("=" * 80)
    
    # Use the Vietnamese financial document
    input_file = "data/mock_vietnamese_financial.json"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    # 1. Parse Data
    print("\n[1] Parsing Vietnamese financial document...")
    parser = JSONParser()
    doc = parser.parse(input_file)
    print(f"✓ Parsed Document: {doc.title}")
    print(f"  - {len(doc.sections)} Sections")
    print(f"  - {len(doc.paragraphs)} Paragraphs")
    print(f"  - {len(doc.tables)} Tables")
    total_cells = sum(len(table.cells) for table in doc.tables)
    print(f"  - {total_cells} Total table cells")
    
    # 2. Build Graph with Vietnamese Embeddings
    print(f"\n[2] Constructing Graph with Vietnamese Embeddings...")
    builder = GraphBuilder(use_real_embeddings=True)
    graph = builder.build_graph(doc)
    
    print("\n--- Graph Statistics ---")
    print(f"Node Features: {graph.x.shape}")
    print(f"Edge Index:    {graph.edge_index.shape}")
    print(f"Edge Attr:     {graph.edge_attr.shape}")
    node_type_counts = graph.node_types.unique(return_counts=True)
    print(f"Node Types:    {dict(zip(node_type_counts[0].tolist(), node_type_counts[1].tolist()))}")
    print(f"               (0=Paragraph, 1=Section, 2=Table, 3=Cell)")
    
    # 3. Create node metadata for retrieval
    print("\n[3] Creating node metadata...")
    node_metadata = []
    
    # Add sections
    for sec in doc.sections:
        node_metadata.append({
            'id': sec.id,
            'type': 'Section',
            'text': sec.title,
            'content_ids': sec.content_ids
        })
    
    # Add paragraphs
    for p in doc.paragraphs:
        node_metadata.append({
            'id': p.id,
            'type': 'Paragraph',
            'text': p.text[:100] + '...' if len(p.text) > 100 else p.text
        })
    
    # Add tables
    for table in doc.tables:
            node_metadata.append({'id': table.id, 'type': 'Table', 'text': table.caption})
            for cell in table.cells:
                cell_id = f"{table.id}_r{cell.row_idx}_c{cell.col_idx}"
                
                # Trích xuất header an toàn (tránh lỗi out of index nếu bảng thiếu header)
                row_header = table.row_headers[cell.row_idx] if cell.row_idx < len(table.row_headers) else f"Row {cell.row_idx}"
                col_header = table.col_headers[cell.col_idx] if cell.col_idx < len(table.col_headers) else f"Col {cell.col_idx}"
                
                # Format chuỗi hiển thị theo đúng chuẩn bạn muốn
                display_text = f"[{table.caption}][{row_header}][{col_header}][{cell.value}]"
                
                node_metadata.append({'id': cell_id, 'type': 'Cell', 'text': display_text})
    
    print(f"✓ Created metadata for {len(node_metadata)} nodes")
    
    # 4. Initialize TTGNN
    print("\n[4] Running TTGNN...")

    model = TTGNN(
        input_dim=graph.x.shape[1],
        hidden_dim=768,
        num_layers=2,
        num_heads=8
    )

    model_path = "models/ttgnn/best_model.pt"

    checkpoint = torch.load(model_path, map_location="cpu")

    # FIX HERE
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    with torch.no_grad():
        gnn_embeddings = model(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.node_types
        )

    print(f"✓ TTGNN Output Shape: {gnn_embeddings.shape}")
    
    # 5. Initialize Hierarchical Retriever
    print("\n[5] Initializing Hierarchical Retriever...")
    hierarchical_retriever = HierarchicalRetriever(
        encoder_model=builder.encoder_model,
        gnn_model=model
    )
    print("✓ Hierarchical retriever ready")
    
    # 6. Test Two-Stage Hierarchical Retrieval
    print("\n[6] Testing Two-Stage Hierarchical Retrieval (as per paper)...")
    print("=" * 80)
    
    test_queries = [
        "VinFast đóng vai trò gì trong chiến lược dài hạn của Vingroup",
        "Mảng kinh doanh nào có tốc độ tăng trưởng cao nhất năm 2020",
        "Vì sao mảng bán lẻ được xem là nguồn tạo dòng tiền ổn định cho Vingroup",
        "Doanh thu tổng năm 2020 tăng bao nhiêu phần trăm so với năm 2019"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: '{query}'")
        print(f"{'='*80}")
        
        # Run hierarchical retrieval
        results = hierarchical_retriever.retrieve(
            query=query,
            graph=graph,
            node_metadata=node_metadata,
            top_k_sections=8,
            top_k_leafs=10  # Increased to show more results including paragraphs
        )
        
        # Stage 1 Results: Retrieved Sections
        print(f"\n  [Stage 1] Retrieved Sections (Top-2):")
        for rank, (idx, score, meta) in enumerate(results['sections'], 1):
            print(f"    {rank}. Score={score:.4f} | Section: {meta['text']}")
        
        print(f"\n  [Subgraph] Extracted {results['subgraph_size']} nodes from selected sections")
        
        # Stage 2 Results: Retrieved Leaf Nodes (after GNN)
        print(f"\n  [Stage 2] Retrieved Leaf Nodes (Top-10 after GNN on subgraph):")
        
        # Show all results
        for rank, (idx, score, meta) in enumerate(results['leafs'], 1):
            node_type = meta['type']
            text = meta['text'][:120] if len(meta['text']) > 120 else meta['text']
            marker = "📄" if node_type == "Paragraph" else "🔢"
            print(f"    {rank}. Score={score:.4f} | {marker} [{node_type}] {text}")
    
    # 7. Run Symbolic-Neural Fusion
    print("\n\n[7] Testing Symbolic-Neural Fusion...")
    fusion_engine = SymbolicNeuralFusion(llm_client=None)
    
    # Test with Vietnamese queries
    fusion_queries = [
        ("Tính tỷ lệ tăng trưởng lợi nhuận sau thuế từ 2022 sang 2023", [{"type": "Table"}, {"type": "Cell"}]),
        ("Tóm tắt kết quả kinh doanh năm 2023", [{"type": "Section"}, {"type": "Paragraph"}])
    ]
    
    for query, context_types in fusion_queries:
        print(f"\nQuery: {query}")
        result = fusion_engine(query, context_types)
        print(f"Result: {result}")
    
    print("\n" + "=" * 80)
    print("✓ SUCCESS: Vietnamese financial document pipeline completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
