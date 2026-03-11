# TTGNN Training Guide

This guide shows how to generate training data and train the TTGNN model using supervised contrastive learning.

## Overview

The training pipeline consists of three main steps:
1. **Synthetic Document Generation**: Use LLM to create diverse Vietnamese financial reports
2. **Training Data Generation**: Generate diverse question-answer pairs from documents
3. **Model Training**: Train TTGNN using supervised contrastive learning

## Prerequisites

1. Install dependencies:
```bash
uv pip install openai python-dotenv
```

2. Create `.env` file with your LLM credentials:
```bash
cp .env.example .env
# Edit .env and add your API credentials
```

Example `.env`:
```
LLM_API_KEY=sk-your-api-key-here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4
```

## Quick Start

### Test the Pipeline (Recommended First Step)
```bash
# Generate 3 test documents and 5 sample questions
python test_synthetic_generation.py
```

### Full Training Pipeline
```bash
# Generate 20 documents with 1000 total samples, then train for 50 epochs
python generate_and_train.py \
  --mode both \
  --num_documents 20 \
  --num_samples 1000 \
  --epochs 50
```

## Synthetic Document Generation

The pipeline automatically generates diverse financial documents with:
- **10 Vietnamese companies**: VNPT, Viettel, FPT, MobiFone, VietinBank, etc.
- **Multiple years**: 2020-2024
- **Varied structures**: 3-7 sections, 2-5 tables per document
- **Realistic metrics**: Revenue, profit, financial ratios (sector-specific)
- **Diverse content**: Income statements, balance sheets, ratio analysis, etc.

### Example Document Structure
```json
{
  "id": "doc_vnpt_2023_001",
  "title": "Báo Cáo Tài Chính VNPT 2023",
  "sections": [...],
  "paragraphs": [...],
  "tables": [...]
}
```

## Training Data Structure

Each training sample has query patterns for diversity:
- **specific_value** (25%): Single cell value queries
- **descriptive** (20%): Paragraph content queries
- **comparison** (20%): Multi-cell comparison queries
- **mixed** (20%): Combined paragraph + cell queries
- **summary** (15%): Multi-paragraph summary queries

Example sample:
```json
{
  "id": "sample_0001",
  "query": "Lợi nhuận sau thuế năm 2023 là bao nhiêu?",
  "document_id": "doc_vnpt_2023",
  "positive_nodes": ["p2", "t1_r2_c2"],
  "negative_nodes": ["p1", "p3", "p4", ...],
  "positive_content": {
    "p2": "Lợi nhuận sau thuế đạt 3.800 tỷ đồng...",
    "t1_r2_c2": "3.800 tỷ"
  }
}
```

## Usage

### Option 1: Full Pipeline with Synthetic Documents (Recommended)
```bash
python generate_and_train.py \
  --mode both \
  --num_documents 20 \
  --num_samples 1000 \
  --epochs 50
```

This will:
1. Generate 20 diverse synthetic documents
2. Generate ~50 questions per document (1000 total)
3. Train TTGNN for 50 epochs

### Option 2: Use Single Document (No Synthetic Generation)
```bash
python generate_and_train.py \
  --mode both \
  --no_synthetic \
  --document data/mock_vietnamese_financial.json \
  --num_samples 1000 \
  --epochs 50
```

### Option 3: Generate Documents Only
```bash
python generate_and_train.py \
  --mode generate \
  --num_documents 30 \
  --num_samples 1500
```

### Option 4: Train Only (with existing data)
```bash
python generate_and_train.py --mode train --epochs 50
```

## Advanced Options

### Full Command-Line Arguments
```bash
python generate_and_train.py \
  --mode both \
  --num_samples 1000 \              # Total training samples
  --num_documents 20 \               # Number of synthetic documents
  --samples_per_doc 50 \             # Samples per doc (auto if not specified)
  --use_synthetic \                  # Generate synthetic docs (default)
  --epochs 50 \
  --batch_size 32 \
  --lr 0.001 \
  --temperature 0.07 \
  --hidden_dim 256 \
  --training_data data/training_data.json \
  --synthetic_dir data/synthetic_documents \
  --save_dir models/ttgnn
```

### Flags
- `--use_synthetic`: Generate synthetic documents (default: True)
- `--no_synthetic`: Use single document instead
- `--mode`: `generate`, `train`, or `both`

## Training Process

### 1. Synthetic Document Generation (if enabled)
- LLM generates diverse company reports
- Varies company names, years, sectors, metrics
- Creates realistic tables and narratives
- Ensures structural diversity

### 2. Training Data Generation
- Extracts leaf nodes (paragraphs and cells) from each document
- Samples nodes strategically based on query patterns:
  - **specific_value**: Single cells with numeric data
  - **descriptive**: Individual paragraphs
  - **comparison**: Multiple cells from same table
  - **mixed**: Paragraphs + cells combined
  - **summary**: Multiple related paragraphs
- Uses LLM to generate natural Vietnamese questions
- Tracks used combinations to avoid duplicates
- Processes multiple documents to create diverse dataset

### 3. Model Training
- Uses supervised contrastive loss (Equation 4 from paper)
- Trains TTGNN with GATv2Conv layers
- Batch processing with AdamW optimizer
- Cosine annealing learning rate schedule
- Saves best model and checkpoints
- Training takes ~15-30 minutes for 1000 samples on GPU

## Dataset Statistics

With default settings (20 documents, 1000 samples):
- **Documents**: 20 financial reports from 10 different companies
- **Companies**: VNPT, Viettel, FPT, MobiFone, VietinBank, Vietcombank, etc.
- **Years**: Random mix from 2020-2024
- **Samples per document**: ~50 questions
- **Total samples**: 1000 diverse question-answer pairs
- **Query patterns**: Distributed across 5 pattern types
- **Unique combinations**: Maximized through deduplication

## Loss Function

Following the paper (Equation 4):
```
Lcon = -log(exp(sim(hi, hp)/τ) / Σ exp(sim(hi, hn)/τ))
```

Where:
- `hi`: anchor (query) embedding
- `hp`: positive node embedding
- `hn`: negative node embeddings  
- `τ`: temperature parameter (default: 0.07)
- `sim`: cosine similarity

## Output

After training, you'll have:
- `models/ttgnn/best_model.pt` - Best model checkpoint
- `models/ttgnn/checkpoint_epoch_*.pt` - Periodic checkpoints
- `models/ttgnn/training_history.json` - Loss curve
- `data/training_data.json` - Generated training data

## Using Trained Model

```python
from hierfinrag.graph.ttgnn import TTGNN
from hierfinrag.training.trainer import load_model

# Initialize model
model = TTGNN(input_dim=768, hidden_dim=256, num_layers=2, num_heads=8)

# Load trained weights
model = load_model(model, "models/ttgnn/best_model.pt")

# Use in retrieval pipeline
# (model will now produce meaningful embeddings instead of random ones)
```

## Tips

1. **Start small**: Test with 3 documents and 50 samples first:
   ```bash
   python test_synthetic_generation.py
   ```

2. **Recommended settings** for production:
   - 20-30 documents
   - 1000-2000 total samples (30-50 per document)
   - 50-100 training epochs

3. **Monitor loss**: Loss should decrease steadily; if it plateaus early, try:
   - Lower learning rate
   - Adjust temperature
   - More training samples or documents
   - Different query pattern distribution

4. **GPU recommended**: Training is much faster on GPU (5-10x speedup)

5. **LLM costs**: Approximate API calls:
   - Document generation: ~10-15 calls per document
   - Question generation: 1 call per sample
   - Total for 20 docs + 1000 samples ≈ 1,200-1,300 API calls

6. **Diversity matters**: More documents with fewer samples each > fewer documents with many samples

7. **Check synthetic quality**: Review a few generated documents in `data/synthetic_documents/` before generating large datasets

## Example Output

```
================================================================================
STEP 1: GENERATING TRAINING DATA
================================================================================

[1.1] Generating 20 synthetic documents...
================================================================================
Generating 20 synthetic Vietnamese financial documents...
================================================================================

[1/20]   Generating: VNPT 2023 (5 sections, 3 tables)
    ✓ Saved to doc_vnpt_2023_000.json
[2/20]   Generating: Viettel 2022 (6 sections, 4 tables)
    ✓ Saved to doc_viettel_2022_001.json
...

✓ Generated 20 documents
✓ Saved to: data/synthetic_documents/

[1.2] Generating 50 training samples per document...
      Total target: 20 docs × 50 samples = 1000 samples

────────────────────────────────────────────────────────────────────────────────
Processing document 1/20: doc_vnpt_2023_000.json
────────────────────────────────────────────────────────────────────────────────
  ✓ Parsed: Báo Cáo Tài Chính VNPT 2023
    - 5 sections, 15 paragraphs, 3 tables

Document statistics:
  - Total leaf nodes: 87
  - Paragraphs: 15
  - Cells: 72

Query pattern distribution:
  - comparison: 10 (20.0%)
  - descriptive: 10 (20.0%)
  - mixed: 10 (20.0%)
  - specific_value: 12 (24.0%)
  - summary: 8 (16.0%)

[Sample 1] Pattern: specific_value
  Query: Doanh thu năm 2023 đạt bao nhiêu?
  Positive: 1 nodes - ['t1_r0_c2']
  Negative: 86 nodes

...

  ✓ Generated 50 samples from this document

────────────────────────────────────────────────────────────────────────────────
Processing document 2/20: doc_viettel_2022_001.json
────────────────────────────────────────────────────────────────────────────────
...

================================================================================
✓ Total samples generated: 1000
✓ From 20 documents
================================================================================

Saving 1000 samples to data/training_data.json...
✓ Saved to data/training_data.json

================================================================================
STEP 2: TRAINING TTGNN
================================================================================

Initializing TTGNN...
✓ Model initialized:
  - Input dim: 768
  - Hidden dim: 256
  - Layers: 2
  - Attention heads: 8

Epoch 1/50 - Loss: 3.2145 - LR: 0.001000
  → Saved best model (loss: 3.2145)
Epoch 2/50 - Loss: 2.8934 - LR: 0.000998
  → Saved best model (loss: 2.8934)
...
Epoch 50/50 - Loss: 0.4521 - LR: 0.000002

✓ Training completed!
Best loss: 0.4521

================================================================================
✓ ALL STEPS COMPLETED SUCCESSFULLY!
================================================================================
```
