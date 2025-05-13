# Enhanced Transformer for Vietnamese-Lao Neural Machine Translation

This project implements an advanced Transformer-based Neural Machine Translation (NMT) model for translating between Vietnamese (Vi) and Lao (Lo), two low-resource languages. The model incorporates several modern architectural and training enhancements to improve translation quality.

![Transformer Architecture (Illustrative)](https://[Placeholder: Link to an illustrative image of Transformer, or your model if you have one])
*(Replace the placeholder link above with an actual image if desired)*

## Features

*   **Transformer Architecture:** Built from scratch using PyTorch.
*   **Rotary Positional Embedding (RoPE):** For improved relative positional encoding.
*   **GeGLU Activation:** Utilized in the Feed-Forward Network (FFN) layers for better expressiveness.
*   **Pre-Layer Normalization (Pre-LN):** For enhanced training stability.
*   **Optimized Training:**
    *   Label Smoothing
    *   AdamW Optimizer
    *   "Attention is All You Need" Learning Rate Scheduler (Inverse Square Root with Warmup)
*   **Beam Search Decoding:** With length penalty for higher quality translations.
*   **Tokenization:** SentencePiece (BPE) for subword tokenization.
*   **Evaluation:** BLEU and METEOR scores.

## Project Structure
├── data/
│ ├── train.vi
│ ├── train.lo
│ ├── dev.vi
│ ├── dev.lo
│ ├── test.vi
│ └── test.lo
├── src/
│ └── main_nmt.py # Main Python script for training and inference
├── spm_vi.model # Trained SentencePiece model for Vietnamese
├── spm_vi.vocab # Vietnamese vocabulary file
├── spm_lo.model # Trained SentencePiece model for Lao
├── spm_lo.vocab # Lao vocabulary file
├── transformer_vi_lo_final_best.pt # Saved best model weights
├── requirements.txt # Python dependencies
├── README.md # This file
└── (Optional: vi_lo_nmt_report.pdf, vi_lo_nmt_slides.pdf)