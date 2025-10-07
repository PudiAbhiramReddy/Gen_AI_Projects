# Aspect-Based Sentiment Analysis (ABSA)

A comprehensive implementation of Aspect-Based Sentiment Analysis using both Transformer-based models and LSTM baseline approaches. This project performs two main tasks: Aspect Term Extraction (ATE) and Aspect Sentiment Classification (ASC) on restaurant review data from the SemEval-2014 dataset.

##  Project Overview

This project implements a two-stage pipeline for aspect-based sentiment analysis:

1. **Aspect Term Extraction (ATE)**: Identifies aspect terms in text using BIO tagging
2. **Aspect Sentiment Classification (ASC)**: Classifies the sentiment of extracted aspects into 4 categories: positive, negative, neutral, and conflict

The implementation compares two approaches:
- **Transformer Pipeline**: RoBERTa-based models for both ATE and ASC
- **Baseline LSTM**: Bidirectional LSTM for ASC comparison

##  Dataset

- **Source**: SemEval-2014 Restaurant Reviews Dataset
- **Format**: XML with aspect terms and sentiment annotations
- **Size**: 3,044 sentences with aspect-level annotations
- **Split**: 80% training (2,435 sentences), 20% testing (609 sentences)

##  Architecture

### Aspect Term Extraction (ATE)
- **Model**: RoBERTa-base fine-tuned for token classification
- **Task**: BIO tagging (B-ASP, I-ASP, O)
- **Input**: Raw text sentences
- **Output**: Character-level spans of aspect terms

### Aspect Sentiment Classification (ASC)

#### Transformer Approach
- **Model**: RoBERTa-base fine-tuned for sequence classification
- **Input**: Sentence + aspect term pairs (format: `sentence [SEP] aspect_term`)
- **Output**: 4-class sentiment classification

#### LSTM Baseline
- **Model**: Bidirectional LSTM with attention pooling
- **Features**: Word embeddings + aspect position encoding
- **Architecture**: Embedding → BiLSTM → Aspect Pooling → Dense Layer

##  Quick Start

### Prerequisites

1. **Install required packages**:
   ```bash
   pip install transformers datasets seqeval evaluate accelerate torch scikit-learn nltk pandas numpy
   ```

2. **Download NLTK data** (handled automatically in the notebook):
   ```python
   import nltk
   nltk.download('punkt')
   ```

3. **Ensure you have the dataset**: The `Restaurants_Train.xml` file should be in the project directory.

### Usage

**Train the models** by running the complete notebook:
```bash
jupyter notebook Aspect_based_sentiment_analysis.ipynb
```

The notebook contains all the necessary code for training, evaluation, and inference. Simply execute all cells to train the models and run the complete pipeline.

##  Performance Results

### Aspect Term Extraction (ATE)
- **Precision**: 87.97%
- **Recall**: 90.31%
- **F1-Score**: 89.12%

### End-to-End Performance (Strict Match)

| Model | Precision | Recall | F1-Score | Predicted Pairs |
|-------|-----------|--------|----------|-----------------|
| **Transformer Pipeline** | 73.09% | 75.06% | 74.06% | 457 |
| **LSTM Baseline** | 65.60% | 64.72% | 65.16% | 439 |

*Ground Truth Pairs: 445*

##  Technical Implementation

### Data Processing
- XML parsing for SemEval format
- Character-to-token alignment for aspect spans
- BIO tagging for aspect term extraction
- Sentence-aspect pair creation for sentiment classification

### Model Training
- **ATE**: 10 epochs, learning rate 2e-5, batch size 8
- **ASC (Transformer)**: 5 epochs, learning rate 2e-5, batch size 4
- **ASC (LSTM)**: 20 epochs, learning rate 1e-3, batch size 16

### Evaluation Metrics
- **ATE**: Precision, Recall, F1 (seqeval)
- **ASC**: Accuracy, Weighted F1
- **End-to-End**: Strict match on (term, sentiment) pairs

##  Project Structure

```
├── Aspect_based_sentiment_analysis.ipynb    # Main implementation notebook
├── Restaurants_Train.xml                    # SemEval dataset
├── README.md                               # This file
└── results/                                # Training outputs (created after running notebook)
    ├── ate_full/                           # ATE model checkpoints
    ├── asc_full/                           # ASC model checkpoints
    └── fine_tuned_models/                  # Final trained models
```

**Note**: The trained models are not included in the repository due to size constraints. Users need to train the models by running the notebook, which will create the model files locally.

##  Key Features

- **Two-stage Pipeline**: Modular ATE and ASC components
- **Multiple Approaches**: Transformer vs LSTM comparison
- **Comprehensive Evaluation**: Strict matching with detailed metrics
- **Self-contained**: Complete training and inference pipeline in one notebook
- **Extensible**: Easy to adapt for different domains or datasets
- **Educational**: Well-documented code with detailed explanations

##  Research Applications

This implementation can be used for:
- Restaurant review analysis
- Product review sentiment analysis
- Social media sentiment monitoring
- Customer feedback analysis
- Opinion mining research

##  Dependencies

- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face model library
- **Datasets**: Data processing utilities
- **scikit-learn**: Machine learning utilities
- **NLTK**: Natural language processing
- **pandas/numpy**: Data manipulation




