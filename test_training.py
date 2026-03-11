"""
Quick test script to verify training pipeline works.
Generates 10 samples and trains for 3 epochs.
"""

from generate_and_train import generate_training_data, train_ttgnn

print("Testing training pipeline with small dataset...\n")

# Generate 10 samples
print("Step 1: Generating 10 training samples...")
try:
    generate_training_data(
        num_samples=10,
        output_path="data/test_training_data.json"
    )
    print("✓ Data generation successful!\n")
except Exception as e:
    print(f"✗ Data generation failed: {e}\n")
    print("Make sure you have:")
    print("  1. Created .env file with LLM credentials")
    print("  2. Installed openai and python-dotenv packages")
    exit(1)

# Train for 3 epochs
print("Step 2: Training TTGNN for 3 epochs...")
try:
    train_ttgnn(
        training_data_path="data/test_training_data.json",
        num_epochs=3,
        batch_size=4,
        save_dir="models/ttgnn_test"
    )
    print("✓ Training successful!\n")
except Exception as e:
    print(f"✗ Training failed: {e}\n")
    exit(1)

print("="*80)
print("✓ Test completed successfully!")
print("="*80)
print("\nYou can now run the full training with:")
print("  python generate_and_train.py --mode both --num_samples 1000 --epochs 50")
