"""
Quick test script for synthetic document generation.

This generates a small number of synthetic documents and training samples
to verify the pipeline works before running the full generation.
"""

import sys
from hierfinrag.training.synthetic_doc_generator import SyntheticDocumentGenerator
from hierfinrag.parsing.json_parser import JSONParser
from hierfinrag.training.data_generator import TrainingDataGenerator

def main():
    print("\n" + "="*80)
    print("TESTING SYNTHETIC DOCUMENT GENERATION PIPELINE")
    print("="*80 + "\n")
    
    # Step 1: Generate 3 synthetic documents
    print("[STEP 1] Generating 3 synthetic documents...\n")
    try:
        syn_gen = SyntheticDocumentGenerator(env_path=".env")
        
        # Test LLM connection first
        print("\n[Pre-check] Testing LLM API connection...")
        if not syn_gen.test_connection():
            print("\n✗ LLM connection test failed!")
            print("Please check your .env file and API credentials.")
            return
        
        print("\n[Generating] Creating synthetic documents...\n")
        doc_files = syn_gen.generate_dataset(
            num_documents=3,
            output_dir="data/test_synthetic"
        )
        print(f"✓ Generated {len(doc_files)} documents\n")
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nMake sure you have LLM_API_KEY in your .env file!")
        return
    
    # Step 2: Generate 5 questions from first document
    print("\n[STEP 2] Generating 5 training samples from first document...\n")
    try:
        parser = JSONParser()
        doc = parser.parse(doc_files[0])
        
        print(f"Document: {doc.title}")
        print(f"  - {len(doc.sections)} sections")
        print(f"  - {len(doc.paragraphs)} paragraphs")
        print(f"  - {len(doc.tables)} tables\n")
        
        question_gen = TrainingDataGenerator(env_path=".env")
        
        # Test LLM connection
        print("\n[Pre-check] Testing LLM API connection for question generation...")
        if not question_gen.test_connection():
            print("\n✗ LLM connection test failed!")
            return
        
        print("\n[Generating] Creating training samples...\n")
        samples = question_gen.generate_dataset(
            doc=doc,
            num_samples=5,
            output_path="data/test_training_samples.json"
        )
        
        print(f"✓ Generated {len(samples)} samples")
        print("\nSample questions:")
        for i, sample in enumerate(samples[:3], 1):
            print(f"\n{i}. {sample.query}")
            print(f"   Positive nodes: {sample.positive_nodes}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*80)
    print("✓ TEST COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")
    print("Next steps:")
    print("1. Check generated documents in: data/test_synthetic/")
    print("2. Check training samples in: data/test_training_samples.json")
    print("3. Run full generation:")
    print("   python generate_and_train.py --mode generate --num_documents 20 --num_samples 1000")
    print()

if __name__ == "__main__":
    main()
