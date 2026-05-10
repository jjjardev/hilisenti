# HiliSenti: A Multi‑Domain Sentiment Analysis Dataset and Model for Hiligaynon

[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-HiliSenti--v1-yellow)](https://huggingface.co/datasets/jjjardev/hilisenti-v1)
[![DOI](https://zenodo.org/badge/DOI/10.57967/hf/8737.svg)](https://doi.org/10.57967/hf/8737)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/Dataset%20License-CC%20BY--NC--SA%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) _(paper forthcoming)_

**HiliSenti** is the first large‑scale, multi‑domain sentiment analysis dataset for **Hiligaynon**, an Austronesian language spoken by over 10 million people in the Philippines. The dataset contains **23,337** real‑world Hiligaynon sentences—many exhibiting natural code‑switching with Tagalog and English—manually annotated for sentiment (Negative, Neutral, Positive). Alongside the dataset, we provide a fine‑tuned **XLM‑RoBERTa‑large** model that achieves state‑of‑the‑art performance on Hiligaynon sentiment classification.

---

## Table of Contents

- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Using the Dataset](#using-the-dataset)
  - [Training the Model](#training-the-model)
- [Citation](#citation)
- [License](#license)

---

## Dataset

The **HiliSenti‑v1** dataset is publicly available on the Hugging Face Hub:

- **Hub repository:** [jjjardev/hilisenti‑v1](https://huggingface.co/datasets/jjjardev/hilisenti-v1)
- **DOI:** [10.57967/hf/8737](https://doi.org/10.57967/hf/8737)

It is split into `train` (18,854), `validation` (2,241), and `test` (2,242) sets.

### Quick load

```python
from datasets import load_dataset
dataset = load_dataset("jjjardev/hilisenti-v1")
print(dataset["train"][0])
# {
#   "sentence": "Kasadya gid sang MassKara festival subong nga tuig!",
#   "label": 2  (0=Negative, 1=Neutral, 2=Positive)
# }
```

A full **Dataset Card** (covering sources, annotation process, biases, and limitations) is available on the Hugging Face page.

---

## Model

We fine‑tuned **xlm‑roberta‑large** (355 M parameters) using cross‑lingual transfer learning. The training script in this repository reproduces the entire training pipeline.

- **Model weights:** The full fine‑tuned checkpoint is currently available upon request for non‑commercial research use.
  A **LoRA adapter** (lightweight, open‑source) is planned for release after the publication of the accompanying paper.
- **Inference code:** Coming soon – you will be able to load the adapter on top of the base `xlm‑roberta‑large` model using a few lines of code.

---

## Results

Evaluation on the held‑out **test set** (2,242 sentences) using the fine‑tuned model:

| Metric        | Overall   | Negative | Neutral  | Positive |
| ------------- | --------- | -------- | -------- | -------- |
| **Precision** | 0.94      | 0.95     | 0.93     | 0.93     |
| **Recall**    | 0.94      | 0.95     | 0.90     | 0.95     |
| **F1‑score**  | **0.94**  | **0.95** | **0.91** | **0.94** |
| **Accuracy**  | **93.5%** |          |          |          |

- **Macro F1:** 93.36%
- **Balanced accuracy:** 93.29%

The model easily exceeds the project’s original target of **≥80% F1‑score**, demonstrating that cross‑lingual transfer learning can effectively handle an extremely low‑resource language like Hiligaynon.

---

## Repository Structure

```
hilisenti/
├── README.md               ← you are here
├── .gitignore
├── requirements.txt        # Python dependencies
├── code/
│   └── train.py            # training script (loads dataset from HF)
└── dataset/
    └── README.md           # link to the official dataset on Hugging Face
```

> **Note:** The actual CSV files (`train.csv`, `val.csv`, `test.csv`) are **not** stored in this repository. They are maintained exclusively on Hugging Face to ensure a single, canonical source.

---

## Quick Start

### Prerequisites

- Python 3.10 or later
- A machine with a GPU (training was done on a Tesla T4; the script also runs on CPU for testing, but training will be slow)
- [Hugging Face Hub](https://huggingface.co) account (optional – only needed if you want to push a model)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/jjjardev/hilisenti.git
   cd hilisenti
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Using the Dataset

You can load the dataset directly from Hugging Face in any Python script:

```python
from datasets import load_dataset

dataset = load_dataset("jjjardev/hilisenti-v1")
print(f"Train samples: {len(dataset['train'])}")
```

### Training the Model

To reproduce the training of the HiliSenti sentiment classifier:

```bash
cd code
python train.py
```

The script will:

- Download the dataset from Hugging Face
- Apply Hiligaynon‑specific text normalisation
- Fine‑tune `xlm-roberta-large` with the same hyper‑parameters used in the paper
- Save the best model checkpoint and tokenizer to `./hilisenti_model/`

You can customise training parameters (batch size, number of epochs, etc.) at the top of `train.py`.

---

## Citation

If you use the **dataset** or the **code** in your research, please cite the following:

```bibtex
@dataset{jarder2026hilisenti,
  author    = {Jarder, Jessie James T.},
  title     = {HiliSenti v1: A Multi‑Domain Sentiment Analysis Dataset for Hiligaynon},
  year      = {2026},
  publisher = {Hugging Face},
  doi       = {10.57967/hf/8737},
  url       = {https://huggingface.co/datasets/jjjardev/hilisenti-v1}
}
```

An **ACL‑style paper** describing the full methodology, experiments, and analysis is in preparation and will be linked here once published.

---

## License

- **Dataset:** [CC BY‑NC‑SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) – free for non‑commercial use with attribution.
- **Code (this repository):** [MIT License](https://opensource.org/licenses/MIT) – you may freely use, modify, and distribute the code, even commercially, as long as the original copyright notice is included.

---

_For questions or collaborations, please open an issue on this repository or reach out through the Hugging Face dataset page._
