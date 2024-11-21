# Language Entailment in NLP 🚀

This project focuses on **textual entailment**—a crucial task in **Natural Language Processing (NLP)** where the goal is to determine if the meaning of one sentence (hypothesis) is logically inferred from another sentence (premise). By leveraging **transformer-based models** like **DeBERTa** and exploring **textual graph-based fragmentation**, the project aims to enhance entailment detection accuracy.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Code Structure](#code-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)
7. [Future Work](#future-work)
8. [References](#references)

---

## Project Overview

The project leverages the **Stanford Natural Language Inference (SNLI) Corpus**, containing sentence pairs classified as:
- **Neutral**
- **Entailment**
- **Contradiction**

We fine-tuned the **DeBERTa model**, a state-of-the-art transformer-based architecture, to classify sentence pairs. Additionally, we incorporated **textual graphs** to break sentences into smaller fragments, aiming to identify relationships at a granular level.

---

## Key Features

- **DeBERTa Integration**: Fine-tunes a transformer-based model for entailment classification.
- **Textual Graph Analysis**: Breaks sentences into fragments for more detailed semantic analysis.
- **Custom Dataset Loader**: Tokenizes and preprocesses input sentences dynamically.
- **Metrics Logging**: Tracks training and validation accuracy.

---

## Code Structure

### Files and Purpose:
1. **`train.py`**:
   - Handles dataset loading, model training, and evaluation.
   - Uses a `DeBERTa` model for fine-tuning on the SNLI dataset.
   - Logs metrics using a custom `Metrics` class and supports Weights & Biases integration.

2. **`helper_fn.py`**:
   - Contains helper functions and classes:
     - **Metrics**: Tracks accuracy and other evaluation metrics.
     - **Collate**: Dynamically batches and pads tokenized inputs.

3. **`textual_graphs.py`**:
   - Implements the `Build_Fragments` class:
     - Breaks sentences into semantic fragments.
     - Extracts root sentences and entities for graph-based analysis.

4. **`Report on Entailment in Language Pragmatics.pdf`**:
   - A detailed report discussing the theoretical underpinnings of textual entailment, methodologies, and the use of textual graphs.

5. **Existing `README.md`**:
   - Initial placeholder for project documentation.

---
## Results
## Results

| Experiment                        | Train Accuray | | Validation Accuracy |
|-----------------------------------|---------------|-|---------------------|
| Without Textual Graphs            | .882          | |.891                 |
| With Textual Graphs               | .880          | |.891                 |



---
## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Language_Entailment.git
    cd Language_Entailment
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the SNLI dataset and place it in the project directory:
    - Train and validation files should be in `.json` format.

4. Ensure you have GPU support (e.g., **NVIDIA A100**) for faster training.

---

