# Sexism Detection with BiLSTM and Transformer Models

A deep learning project for **fine-grained sexism detection in social media text**, based on **EXIST 2023 Task 2**.

The system classifies tweets into four categories:

- **Non-Sexist**
- **Direct Sexism**
- **Judgemental Sexism**
- **Reported Sexism**

The project compares recurrent neural networks using static word embeddings with modern Transformer architectures, and further explores **multilingual training** and **class-weighted loss** to improve performance on minority classes.

---

## Overview

Sexism detection on social media is challenging because sexist intent is not always expressed explicitly. Messages may contain direct sexist language, report a sexist incident, or condemn sexist behavior through judgement, irony, or contextual framing.

This project investigates that problem using multiple neural architectures:

1. **Baseline BiLSTM**
2. **Stacked BiLSTM**
3. **Twitter-RoBERTa**
4. **XLM-RoBERTa trained on English + Spanish**
5. **Class-weighted XLM-RoBERTa**

The experiments include data preprocessing, majority-vote label aggregation, GloVe embeddings, robust evaluation over multiple random seeds, class-level analysis, confusion matrices, precision-recall curves, and qualitative error analysis.

---

## Task

The project addresses **Task 2 of EXIST 2023**, where the goal is to categorize a message according to the author's intent.

| Label | Description |
|---|---|
| `0` — Non-Sexist | No clear sexist intent |
| `1` — DIRECT | The message is sexist itself or encourages sexism |
| `2` — JUDGEMENTAL | The message describes sexism while judging or condemning it |
| `3` — REPORTED | The message reports a sexist situation experienced by someone |

The original dataset contains annotations from multiple annotators. Final labels are obtained through **majority voting**, and examples without a clear majority are discarded.

---

## Models

### 1. Baseline BiLSTM

A bidirectional LSTM classifier using:

- GloVe word embeddings
- Bidirectional LSTM
- Dense classification layer
- Softmax output over four classes

### 2. Stacked BiLSTM

An extension of the baseline architecture with two Bidirectional LSTM layers, allowing the model to learn deeper sequential representations.

### 3. Twitter-RoBERTa

The Transformer baseline uses:

`cardiffnlp/twitter-roberta-base-hate`

This model is already adapted to social-media and hate-speech language, making it suitable for noisy Twitter text.

### 4. Multilingual XLM-RoBERTa

XLM-RoBERTa is trained using both **English and Spanish** examples and evaluated on the English test split.

This experiment investigates whether multilingual representations improve robustness and generalization.

### 5. Class-Weighted XLM-RoBERTa

Because the dataset is strongly imbalanced, a custom weighted loss assigns larger penalties to errors on minority classes.

This improves sensitivity to underrepresented labels such as **Judgemental** and **Reported** sexism.

---

## Data Processing

The notebook performs the following preprocessing steps:

- Load training, validation, and test JSON files
- Aggregate Task 2 annotations through majority voting
- Remove samples without a clear majority
- Filter the English-only dataset for the main experiments
- Encode class labels numerically
- Remove:
  - emojis
  - hashtags
  - mentions
  - URLs
  - special characters
  - curly quotation marks
- Perform tokenization and lemmatization

For the multilingual extension, both English and Spanish data are used.

---

## Word Embeddings

The BiLSTM models use pretrained **GloVe embeddings**.

The vocabulary contains:

- words available in GloVe
- training-set words missing from GloVe
- special tokens such as padding and unknown words

The notebook also analyzes out-of-vocabulary behavior on the evaluation data.

---

## Evaluation

Models are evaluated using:

- **Macro Precision**
- **Macro Recall**
- **Macro F1-score**
- **Accuracy**
- Per-class precision, recall, and F1
- Confusion matrices
- Precision-recall curves

To improve robustness, the main model families are trained using multiple random seeds:

```text
1, 42, 123
```

Results are summarized using the average and standard deviation across runs.

---

## Main Findings

The experiments show a clear performance advantage for Transformer-based architectures over the BiLSTM baselines.

Key observations include:

- Transformer models achieve substantially stronger Macro F1 than BiLSTMs.
- Twitter-RoBERTa performs particularly well on the majority **Non-Sexist** class.
- Overall accuracy can be misleading because of strong class imbalance.
- **Judgemental** and **Reported** sexism are substantially harder to recognize.
- Multilingual training alone does not fully solve the imbalance problem.
- Applying a **class-weighted loss** improves recall for minority classes and yields the strongest balanced performance among the evaluated configurations.
- Static GloVe embeddings are more vulnerable to slang, rare words, and creative spelling than Transformer subword representations.

The notebook reports Transformer Macro F1 around **0.53**, compared with approximately **0.41–0.44** for the BiLSTM variants.

---

## Error Analysis

The notebook includes both quantitative and qualitative error analysis.

### Reported vs. Non-Sexist

Tweets reporting sexist incidents in neutral or journalistic language are often predicted as Non-Sexist because explicit sexist wording may be absent.

### Implicit and Judgemental Sexism

Judgemental examples frequently rely on:

- sarcasm
- irony
- subtle framing
- pragmatic context

These signals remain difficult for all evaluated models.

### Aggressive Language False Positives

Some Non-Sexist tweets containing hostile or emotionally charged language are incorrectly classified as Direct Sexism, suggesting that the models may sometimes confuse aggression with sexist intent.

---

## Project Structure

```text
Sexism-Detection-main/
│
└── sexism_detection.ipynb
```

The notebook contains the complete workflow:

```text
Dataset Loading
      ↓
Majority Voting
      ↓
Data Cleaning
      ↓
GloVe Vocabulary
      ↓
BiLSTM Models
      ↓
Multi-Seed Evaluation
      ↓
Twitter-RoBERTa
      ↓
XLM-RoBERTa EN + ES
      ↓
Class-Weighted XLM-R
      ↓
Model Comparison
      ↓
Error Analysis
```

---

## Model Weights

Trained **BiLSTM and Transformer model weights** are available on Google Drive:

https://drive.google.com/drive/folders/1O5oG586iDtR5kMNCZL3KoGppjCDyhXIM

---

## Dataset

The project uses a reduced version of the **EXIST 2023** dataset prepared for the University of Bologna NLP course.

Course data repository:

https://github.com/lt-nlp-lab-unibo/nlp-course-material/tree/main/2025-2026/Assignment%201/data

For information about the original EXIST shared task:

https://clef2023.clef-initiative.eu/index.php?page=Pages/labs.html#EXIST

---

## Running the Notebook

The project is designed to run in **Google Colab** or a Python/Jupyter environment.

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Sexism-Detection-main
```

### 2. Install the main dependencies

```bash
pip install numpy pandas matplotlib scikit-learn nltk gensim tensorflow torch transformers datasets accelerate
```

### 3. Download the dataset

Place the following files in the expected notebook location:

```text
training.json
validation.json
test.json
```

If running locally, update the dataset paths in the notebook from `/content/...` to your local paths.

### 4. Launch Jupyter

```bash
jupyter notebook sexism_detection.ipynb
```

---

## Technologies

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn
- NLTK
- Gensim
- GloVe
- TensorFlow / Keras
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Twitter-RoBERTa
- XLM-RoBERTa

---

## Reproducibility

The experiments use fixed random seeds and evaluate models over multiple runs where required.

The notebook reports both mean and standard deviation instead of relying on a single training run, providing a more reliable comparison across architectures.

---

## Limitations

Several challenges remain:

- Severe class imbalance
- Very small minority classes
- Difficulty modeling sarcasm and irony
- Ambiguity between reported sexism and neutral discussion
- Domain-specific social-media language
- Multilingual transfer does not automatically improve minority-class recognition

Future work could explore focal loss, oversampling, data augmentation, instruction-tuned language models, contrastive learning, or hierarchical formulations of the EXIST task.

---

## Acknowledgments

This project was developed for an NLP assignment at the **University of Bologna**.

Original assignment credits:

- Federico Ruggeri
- Eleonora Mancini
- Paolo Torroni

The dataset originates from the **EXIST 2023** shared task.

---

## License

This repository is intended for educational and research purposes.

Please consult the original EXIST dataset and model licenses before redistributing datasets, pretrained checkpoints, or derived artifacts.
