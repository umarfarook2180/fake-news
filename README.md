# GATE: Graph-Augmented Transformer Ensemble for Fake News Detection

This repository implements a hybrid ML framework combining BERT, RoBERTa, and GNNs using a meta-learned ensemble strategy for robust fake news detection.

## 🔍 Highlights
- Transformer-based semantic encoding
- GNNs for relational user-article structure
- Meta-learned ensemble to dynamically weight predictions
- Benchmarked on LIAR and FakeNewsNet

## 📁 Structure

```
gete_fake_news_package/
├── train/              # Training scripts
├── evaluate/           # Evaluation logic
├── models/             # Transformer, GNN, Ensemble modules
├── utils/              # Preprocessing, metrics (to be added)
├── data/               # Datasets
├── results/            # Output plots, metrics
├── notebooks/          # Jupyter notebooks for analysis
```

## 🚀 Run

```bash
pip install -r requirements.txt
python train/train.py
```

## 📊 Evaluation

```bash
python evaluate/evaluate.py
```

## 📜 License

MIT License
