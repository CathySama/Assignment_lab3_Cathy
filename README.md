# DistilBERT Full Fine-Tuning (3‑Class) — Reproducible Template

This repository accompanies the notebook **`lab3-2.ipynb`**, which performs **full fine‑tuning** of a pretrained Transformer (**`distilbert-base-uncased`**) for a **3‑class sentiment** task. It uses **PyTorch + HuggingFace Transformers**, early stopping, and macro‑F1 model selection. The code is clean, simple, and ready to reproduce on your machine.

> Short & focused — includes: dataset description, preprocessing, model, training config, evaluation, and a brief reflection — without unnecessary complexity.

---

## 1) What this project does

- **Task**: Multi‑class text classification (3 classes).
- **Backbone**: `distilbert-base-uncased` (a compact, distilled BERT with good speed/quality tradeoff).
- **Frameworks**: PyTorch, HuggingFace Transformers/Datasets, Evaluate, scikit‑learn.
- **Training**: Full fine‑tuning, LoRA.
- **Outputs**: Best checkpoint and metrics saved under `runs/fpb_full_ft(or fpb_lora2)/`.


├─ main.py                 # single entry point to reproduce results
├─ run.sh                  # simple shell wrapper (optional)
├─ configs/
│  ├─ finetune_full.yaml   # full FT hyperparams
│  └─ finetune_lora.yaml   # LoRA hyperparams
├─ src/                    # training, data, eval modules
├─ data/                   # local datasets
├─ outputs/                # logs, checkpoints, metrics
└─ requirements.txt        # Python deps

---

## 2) Environment
- OS: Linux
- Python: **3.10.18**
pip install -r requirements.txt

---

## 3) Data (simple & flexible)

Source: https://huggingface.co/datasets/takala/financial_phrasebank
  
This project reads a single .txt file with sentence–label pairs and builds train/validation/test splits. The loader automatically scans for a subset file in the current directory (DATA_DIR="./") using this priority:
	1.	Sentences_AllAgree.txt
	2.	Sentences_75Agree.txt
	3.	Sentences_66Agree.txt
	4.	Sentences_50Agree.txt

Put one of these files next to main.py (or adjust DATA_DIR). The first one found is used.


---

## 4) Preprocessing (as in the notebook)

- **Tokenizer**: `AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)`  
- **Function**: a `tok_fn` that truncates to `max_length=128` for safety.
- **Mapping**: apply `map(tok_fn, batched=True)` on each split.
- **Keep only**: `input_ids`, `attention_mask`, `label` (any extra columns are removed).
- **Collator**: `DataCollatorWithPadding` to pad dynamically per batch.

Why 128 tokens?
- Most short‑text sentiment tasks fit within 128 subwords, which speeds up training and reduces memory.

---

## 5) Model & Metrics

- **Model**: `AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)`  
  (classification head on top of DistilBERT; the notebook sets `num_labels=3`)
- **Metrics**: implemented via a `compute_metrics` function with **accuracy** and **macro‑F1**.  
  Macro‑F1 treats each class equally — better than accuracy when classes are imbalanced or unevenly hard.

---

## 6) Training Configuration (exactly as coded)

These are the core `TrainingArguments` used in the notebook:

```python
TrainingArguments(
    output_dir="runs/fpb_full_ft",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=20,
    weight_decay=0.05,
    fp16=True,                    # enable on CUDA
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_steps=100,
    report_to="none",
    seed=42,
    greater_is_better=True
)
```

- **Early stopping**: The notebook adds `EarlyStoppingCallback` so training stops when validation macro‑F1 stops improving.  
- **Why LR=1e‑5?** Lower LR stabilizes full fine‑tuning on compact models and tends to yield reliable convergence.
- **Why weight decay=0.05?** Mild L2 regularization improves generalization.
- **Why macro‑F1 for model selection?** Robust under class imbalance and more informative than accuracy alone.

> Tip: On some Transformers versions the correct kwarg is `evaluation_strategy` (the notebook uses `eval_strategy`). If you face warnings, rename to `evaluation_strategy="epoch"` — behavior is the same.

---

## 7) How to run

1. Open **`lab3-2.ipynb`** in Jupyter/VSCode.  
2. Run cells **top to bottom**.  
3. Watch training logs; best model is kept automatically (`load_best_model_at_end=True`).  
4. Artifacts and metrics are saved under: `runs/fpb_full_ft/`.

Optional: convert the notebook to a script and run headless:
```bash
jupyter nbconvert --to script lab3-2.ipynb
python lab3-2.py
```

---


## 9) Reproducibility

- **Seeds**: `seed=42` with a helper to fix global seeds.
- **Determinism**: tokenization + preprocessing are defined in code and can be reused verbatim.
- **Pinned libs**: you may freeze versions in `requirements.txt` for strict repeats.

For stronger claims, you can run **multiple seeds** (e.g., 41/42/43) and report mean ± std of macro‑F1.

---
