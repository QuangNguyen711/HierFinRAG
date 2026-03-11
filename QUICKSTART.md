# Quick Start - HierFinRAG Training

## ✅ Files Created

All files have been recreated! Here's what you have:

### Core Training Files
1. **`hierfinrag/training/synthetic_doc_generator.py`** - LLM-based synthetic document generator
2. **`hierfinrag/training/data_generator.py`** - Training data generator with diverse query patterns
3. **`hierfinrag/training/contrastive_loss.py`** - Loss functions for contrastive learning
4. **`hierfinrag/training/trainer.py`** - TTGNN trainer

### Scripts
5. **`test_synthetic_generation.py`** - Quick test (3 docs, 5 samples)
6. **`generate_and_train.py`** - Full pipeline (generate + train)
7. **`test_training.py`** - Small training test

### Configuration
8. **`.env.example`** - Template for LLM credentials
9. **`TRAINING_GUIDE.md`** - Detailed documentation

## 🚀 Getting Started

### 1. Set up LLM credentials

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API key
# LLM_API_KEY=your-actual-api-key-here
# LLM_BASE_URL=https://api.openai.com/v1
# LLM_MODEL_NAME=gpt-4
```

### 2. Test API connection (Quick check - 5 seconds)

```bash
python test_api_connection.py
```

This verifies your LLM is callable before starting generation. You'll see:
- ✓ Connection successful - API is working
- ✗ Connection failed - Check credentials or model availability

### 3. Test the pipeline (3 documents)

```bash
python test_synthetic_generation.py
```

This will:
- Test API connection
- Generate 3 synthetic Vietnamese financial documents (~2-3 min)
- Create 5 sample questions from the first document (~30 sec)
- Verify everything works

**Note**: Includes automatic pre-checks to avoid long waits if API is not responding

### 3. Run full generation (20 documents, 1000 samples)

```bash
python generate_and_train.py \
  --mode both \
  --num_documents 20 \
  --num_samples 1000 \
  --epochs 50
```

This will:
1. Generate 20 diverse company reports (VNPT, Viettel, FPT, etc.)
2. Create ~50 questions per document (1000 total)
3. Train TTGNN for 50 epochs

## 📊 Key Features

### Synthetic Documents
- **10 companies**: VNPT, Viettel, FPT, MobiFone, VietinBank, Vietcombank, Vinamilk, Vingroup, Hòa Phát, PetroVietnam
- **5 years**: 2020-2024
- **Diverse structures**: 3-7 sections, 2-5 tables
- **Realistic metrics**: Sector-specific revenue, growth rates, financial ratios

### Training Data Diversity
- **5 query patterns**:
  - `specific_value` (25%) - Single cell queries
  - `descriptive` (20%) - Paragraph questions
  - `comparison` (20%) - Multi-cell comparisons
  - `mixed` (20%) - Paragraph + cell combinations
  - `summary` (15%) - Multi-paragraph summaries

## ⚙️ Common Commands

```bash
# Test only (quick verification)
python test_synthetic_generation.py

# Generate documents only
python generate_and_train.py --mode generate --num_documents 30 --num_samples 1500

# Train only (with existing data)
python generate_and_train.py --mode train --epochs 50

# Use single document (no synthetic generation)
python generate_and_train.py --mode both --no_synthetic --num_samples 500

# Full pipeline with custom settings
python generate_and_train.py \
  --mode both \
  --num_documents 25 \
  --num_samples 1250 \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.001
```

## 📁 Output Directories

After running, you'll find:
- `data/synthetic_documents/` - Generated financial reports
- `data/training_data.json` - Training samples
- `models/ttgnn/` - Trained model checkpoints

## ⏱️ Estimated Time

- **API connection test**: ~5 seconds
- **Test run** (3 docs, 5 samples): ~2-3 minutes
- **Full generation** (20 docs, 1000 samples): ~30-60 minutes
- **Training** (50 epochs): ~15-30 minutes (GPU) / ~2-3 hours (CPU)

**Optimizations applied**:
- Pre-flight API health checks (fast fail if issues)
- Disabled reasoning mode with `max_completion_tokens`
- Optimized token limits for faster responses
- Parallel-ready architecture for future batching

## 💰 LLM API Costs

For 20 documents + 1000 samples:
- Document generation: ~200-300 API calls
- Question generation: ~1000 API calls
- **Total**: ~1200-1300 API calls

With GPT-4: ~$5-10 USD
With GPT-3.5: ~$1-2 USD

## 📖 Need More Info?

See **`TRAINING_GUIDE.md`** for:
- Detailed usage instructions
- Advanced configuration options
- Training theory and loss functions
- Troubleshooting tips

## ❓ Troubleshooting

**Error: "LLM_API_KEY not found"**
→ Make sure you created `.env` and added your API key

**Error: "Import error"**
→ Install dependencies: `uv sync` or `pip install -e .`

**LLM timeout/errors**
→ Try with fewer documents first, or use GPT-3.5-turbo for faster/cheaper generation

---

**Ready to start? Run:** `python test_synthetic_generation.py`
