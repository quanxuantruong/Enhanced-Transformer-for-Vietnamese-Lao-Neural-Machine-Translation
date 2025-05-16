# Enhanced Transformer for Vietnamese-Lao Neural Machine Translation

This project implements and evaluates an enhanced Transformer-based Neural Machine Translation (NMT) model for the low-resource Vietnamese (Vi) to Lao (Lo) language pair. The model incorporates several modern architectural improvements to enhance translation quality.

## Project Overview

Translating between low-resource languages like Vietnamese and Lao presents significant challenges due to data scarcity and linguistic disparities. This project aims to address these challenges by:

1.  **Building a Transformer model from scratch** using PyTorch.
2.  **Integrating advanced architectural enhancements**:
    *   **Rotary Positional Embedding (RoPE)** for more effective relative positional encoding.
    *   **GeGLU activation function** within the Feed-Forward Networks (FFNs) for potentially better representation learning.
    *   **Pre-Layer Normalization (Pre-LN)** for improved training stability.
3.  **Investigating the impact of vocabulary size** on translation performance.
4.  Training the model with a **fixed learning rate** and **label smoothing**.
5.  Evaluating the model using the **BLEU score**.

The primary goal is to demonstrate the effectiveness of these combined enhancements in a low-resource NMT setting.

## Features

*   **Custom Transformer Implementation**: Built with PyTorch, allowing for detailed control over the architecture.
*   **Rotary Positional Embedding (RoPE)**: Integrated into the multi-head attention mechanism.
*   **GeGLU Activation**: Used in the position-wise feed-forward networks.
*   **Pre-Layer Normalization**: Applied before each sub-layer in the encoder and decoder.
*   **SentencePiece Tokenization**: Utilizes BPE for subword tokenization, with support for different vocabulary sizes.
*   **Fixed Learning Rate Training**: Employs AdamW optimizer with a fixed learning rate.
*   **Label Smoothing**: Incorporated into the loss function for better generalization.
*   **Greedy Decoding**: Used for generating translations during inference and evaluation.
*   **Evaluation Scripts**: Includes code for calculating BLEU scores.
*   **Clear Training and Inference Pipelines**.

## Dataset

The model is trained and evaluated on a Vietnamese-Lao parallel corpus.
*   **Source Language**: Vietnamese (vi)
*   **Target Language**: Lao (lo)

The data should be pre-split into `train`, `dev` (development/validation), and `test` sets for each language (e.g., `train.vi`, `train.lo`, `dev.vi`, `dev.lo`, `test.vi`, `test.lo`).

The dataset used in this project can be found at VLSP or is based on the VLSP2023 corpus.


## Requirements

*   Python 3.7+
*   PyTorch (tested with version X.Y.Z)
*   SentencePiece
*   SacreBLEU
*   tqdm
