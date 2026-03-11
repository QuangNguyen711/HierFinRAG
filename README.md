# HierFinRAG: Hierarchical Multimodal RAG for Financial Document Understanding

This repository contains the official implementation of the paper:

**HierFinRAG—Hierarchical Multimodal RAG for Financial Document Understanding**

**Quang-Vinh Dang**<sup>1,*</sup>, **Ngoc-Son-An Nguyen**<sup>2</sup>, and **Thi-Bich-Diem Vo**<sup>3</sup>

<sup>1</sup> School of Innovation and Computing Technology, British University Vietnam, Hung Yen 16000, Vietnam  
<sup>2</sup> Faculty of Information Technology, Industrial University of Ho Chi Minh City, Ho Chi Minh City 70000, Vietnam  
<sup>3</sup> GiaoHangNhanh, Ho Chi Minh City 70000, Vietnam

\* Author to whom correspondence should be addressed.

**Informatics 2026, 13(2), 30;**  
[https://doi.org/10.3390/informatics13020030](https://doi.org/10.3390/informatics13020030)

---

## Overview

HierFinRAG is a specialized Retrieval-Augmented Generation (RAG) framework designed for complex financial documents. It addresses the challenges of tabular data, cross-referencing, and long-range dependencies in financial reports (e.g., 10-Ks, Annual Reports) by introducing:

1.  **Hierarchical Parsing**: Structured decomposition of documents into Sections, Paragraphs, Tables, and Cells.
2.  **Table-Text Graph Neural Network (TTGNN)**: A graph-based module to capture relationships between textual content and tabular data.
3.  **Symbolic-Neural Fusion**: A reasoning engine that combines LLM generation with precise symbolic execution for numerical accuracy.

## Installation

Ensure you have Python 3.13+ installed.

1.  Clone the repository:
    ```bash
    git clone https://github.com/quangnguyen711/HierFinRAG.git
    cd HierFinRAG
    ```

2.  Install dependencies:
    ```bash
    uv add torch>=2.0.0 \
    torch_geometric>=2.3.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    networkx>=3.0 \
    scikit-learn>=1.2.0 \
    tqdm>=4.65.0
    ```
    *Note: We recommend installing PyTorch with CUDA support first if you have a GPU.*

## Usage

### 1. Run the Main Pipeline
To see the end-to-end processing from document parsing to hierarchical retrieval:

```bash
uv run run_pipeline.py
```
This script demonstrates:
- Parsing Vietnamese financial document (JSON structure).
- Building the heterogeneous graph with Vietnamese embeddings.
- Running hierarchical two-stage retrieval (Section → Leaf nodes).
- Testing TTGNN and Symbolic-Neural Fusion engine.

### 2. Train TTGNN Model

#### Setup LLM credentials
```bash
cp .env.example .env
# Edit .env and add your LLM API key (for generating training questions)
```

#### Generate training data + Train model
```bash
uv run generate_and_train.py --mode both --num_samples 1000 --epochs 50
```

This will:
- Generate 1000 training samples using LLM
- Train TTGNN with supervised contrastive learning
- Save trained model to `models/ttgnn/`

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed instructions.

#### Quick test (10 samples, 3 epochs)
```bash
uv run test_training.py
```

### 3. Generate Figures and Results
To reproduce the experimental results and figures mentioned in the paper:

```bash
uv run run_demo.py
```
This will generate:
- Performance comparison plots (`results/Fig1_Main_Performance.png`, etc.)
- Ablation study tables (`results/Table1_Ablation.md`)
- Qualitative comparison logs (`results/Qualitative_Comparison.md`)

## Repository Structure

- `hierfinrag/`: Core package containing the implementation.
  - `parsing/`: Document layout analysis and JSON parsing.
  - `graph/`: Graph construction and TTGNN model.
  - `retrieval/`: Hierarchical two-stage retrieval.
  - `reasoning/`: Symbolic execution and fusion logic.
  - `training/`: Training data generation and TTGNN trainer.
- `data/`: Directory for input documents and training data.
- `models/`: Directory for trained model checkpoints.
- `results/`: Output directory for experiments and visualizations.
- `generate_and_train.py`: Main script for training pipeline.
- `TRAINING_GUIDE.md`: Detailed training instructions.

## License

This project is licensed under the GPL 3.0 License - see the [LICENSE](LICENSE) file for details.
