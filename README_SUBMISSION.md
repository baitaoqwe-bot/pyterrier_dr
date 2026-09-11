# MSc Dissertation Submission Notes

## Project

**LoRA for Joint Product Quantisation Training in Dense Document Retrieval**  
Author: Tao Bai  
University of Glasgow, MSc project

This repository is a fork of `terrierteam/pyterrier_dr`. The dissertation work extends the existing PyTerrier-DR Joint Product Quantisation (JPQ) training pipeline with LoRA-based query-encoder adaptation.

The submission version corresponds to the `lora-decouple` branch.

## Main project changes

The main implementation changes are:

- **Decoupled PEFT integration** in `pyterrier_dr/jpq/run.py` through `apply_peft()`.
  - The retriever is instantiated normally.
  - LoRA is then attached to the underlying Hugging Face Transformer model.
  - The generic dense-encoder abstraction therefore remains usable without PEFT.
- **LoRA command-line support** in `pyterrier_dr/jpq/run.py`:
  - `--use-lora`
  - `--lora-r`
  - `--lora-alpha`
  - `--lora-dropout`
- **Differentiable query encoding** in `pyterrier_dr/sbert_models.py` through `encode_queries_torch()` so that the JPQ retrieval loss remains connected to the PyTorch computation graph and can update LoRA parameters.
- **PQ/JPQ implementation updates** in `pyterrier_dr/jpq/pq.py` and `pyterrier_dr/jpq/retriever.py`, including GPU-related PQ handling used during the full experiments.
- Development-only examples under `examples/jpq/` were used for lightweight functional checking. They are not the formal experimental evaluation reported in the dissertation.

## Training modes

The project compares three query-encoder adaptation modes using the same JPQ training pipeline:

1. **Frozen JPQ**: original query encoder frozen; JPQ centroids trainable.
2. **Full JPQ**: full query encoder and JPQ centroids trainable.
3. **LoRA JPQ**: original query-encoder backbone frozen; LoRA parameters and JPQ centroids trainable.

A PQ-only mode is also available through `--pq-only`.

## Installation

Create a Python environment and install the project with JPQ dependencies, for example:

```bash
pip install -e ".[jpq]"
pip install peft sentence-transformers
```

FAISS must also be installed for the selected environment.

The final dissertation experiments were run with the following cluster environment:

- Python 3.10.13
- PyTorch 2.6.0 + cu124
- CUDA 12.4 runtime
- PEFT 0.20.0
- SentenceTransformers 6.0.0
- FAISS 1.15.0
- NumPy 2.2.6

The repository `requirements.txt` and `pyproject.toml` define the package-level dependencies; the versions above record the environment used for the dissertation runs.

## Data and base index

The formal experiments use MS MARCO passage retrieval data with the TCT-ColBERT retriever. A pre-built dense base index is required and is supplied to the training script through `--base-index`.

Large datasets, trained indexes, model checkpoints and cluster-generated artefacts are intentionally not included in the source-code submission package.

## Reference LoRA-JPQ command

The command below shows the reference configuration used for the dissertation. Replace the two paths with the local or cluster paths to the existing TCT base index and desired output directory.

```bash
python pyterrier_dr/jpq/run.py \
  --base-index /path/to/msmarco-passage.tct-hnp.flex \
  --target-dir /path/to/output \
  --model-name tct_colbert \
  --pq-impl faiss2opq \
  --M 96 \
  --nbits 8 \
  --pq-sample-size 159744 \
  --pairs-cap 2000000 \
  --valid-every 500 \
  --patience 3 \
  --in-batch-negs \
  --lambda-rank \
  --jpq-negs 200 \
  --use-lora \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.1
```

For the rank study, `--lora-r` was varied over `2`, `4`, `8`, and `16` while the other LoRA settings were held fixed.

## Baseline commands

### Full JPQ

Run the same command without `--use-lora` and without `--frozen-query-encoder`.

### Frozen JPQ

Run the same command without `--use-lora` and add:

```bash
--frozen-query-encoder
```

### PQ-only

Run with:

```bash
--pq-only
```

## Outputs

For each run, the training script creates a target directory containing the JPQ index/model outputs and evaluation runs. Evaluation is performed for the configured development and test collections, and `metrics.csv` files are written under the run directories.

## Reproducibility notes

The training script sets deterministic/reproducibility-related settings used by the project, including a fixed PyTorch seed and single-threaded FAISS behaviour during the training procedure.

Training duration can still vary with cluster allocation, hardware and early stopping. The dissertation therefore reports both end-to-end wall-clock duration and validation-free step timing.

## Repository version

Submission branch: `lora-decouple`

The branch should be used as the source-code version associated with the dissertation submission. The original upstream PyTerrier-DR project remains available separately from the project-specific modifications in this fork.
