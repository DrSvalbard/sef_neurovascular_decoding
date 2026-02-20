# SEF Neurovascular Decoding

This repository contains a deep learning pipeline designed to decode neural activity (LFP) from functional Ultrasound (fUS) hemodynamic signals.

## Project Overview
This research aims to bridge the gap between hemodynamics and electrophysiology using a hybrid **TransUNet** architecture. The model leverages cross-attention mechanisms to map the delayed and integrated nature of hemodynamic signals back to their neural origins.

## Key Features
* **Hybrid Architecture**: Combines CNNs for spatial feature extraction with Transformers for long-range temporal dependencies.
* **Rigorous Preprocessing**: Implements strict temporal causal splits to prevent data leakage.
* **Biological Grounding**: Normalization and scaling strategies are designed to account for vascular physiology while maintaining ML methodological integrity.

## Technical Implementation
* **Framework**: PyTorch
* **Core Components**: Dilated causal convolutions, Multi-head attention, and custom Dataset managers for synchronized multimodal data.

---
*Note: This code is a research prototype focusing on methodological feasibility (Proof of Concept) rather than production scaling.*
