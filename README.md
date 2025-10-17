# DistilBERT Full Fine-Tuning (3‑Class) — Reproducible Template

This repository accompanies the notebook **`lab3-2.ipynb`**, which performs **full fine‑tuning** of a pretrained Transformer (**`distilbert-base-uncased`**) for a **3‑class sentiment** task. It uses **PyTorch + HuggingFace Transformers**, early stopping, and macro‑F1 model selection. The code is clean, simple, and ready to reproduce on your machine.

> Short & focused — includes: dataset description, preprocessing, model, training config, evaluation, and a brief reflection — without unnecessary complexity.

---

## 1) What this project does

- **Task**: Multi‑class text classification (3 classes).
- **Backbone**: `distilbert-base-uncased` (a compact, distilled BERT with good speed/quality tradeoff).
- **Frameworks**: PyTorch, HuggingFace Transformers/Datasets, Evaluate, scikit‑learn.
- **Training**: Full fine‑tuning (update all parameters), early stopping on **macro‑F1**.
- **Outputs**: Best checkpoint and metrics saved under `runs/fpb_full_ft/`.

---

## 2) Environment

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch transformers datasets evaluate scikit-learn peft pandas matplotlib
```

> If your GPU supports half precision, the notebook already enables **FP16** during training.

---

## 3) Data (simple & flexible)

The notebook expects three splits: **`ds_train`**, **`ds_valid`**, **`ds_test`**.  
They are standard HuggingFace `Dataset` objects (the notebook constructs them with `Dataset.from_...`).

- **Minimum columns**: a text field and an integer **label** in `{0,1,2}` (for 3‑class).  
- If your raw data are Python lists/CSV, you can first build a `Dataset` via `Dataset.from_dict(...)` or `load_dataset('csv', ...)` and **rename** columns to the ones used in the notebook.

> Keep raw data **out of the repo** if files are large. Prefer links or scripts to fetch data.

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

## 8) Evaluation & What to report

The notebook computes validation/test **accuracy** and **macro‑F1**. In your report, include:

- **Scores**: validation & test accuracy / macro‑F1.
- **Confusion matrix** (optional): helps reveal which classes are confused.
- A **short error analysis**: 3–5 representative mistakes and reasons (sarcasm, short context, OOV terms, etc.).

If you plot learning curves (loss/F1 vs epoch), initial flat points can appear due to:
- **LR warmup / small updates** at the start.
- **Padding-heavy batches** early on.
- Few evaluation points if `eval_strategy="epoch"` (metrics update once per epoch).

---

## 9) Reproducibility

- **Seeds**: `seed=42` with a helper to fix global seeds.
- **Determinism**: tokenization + preprocessing are defined in code and can be reused verbatim.
- **Pinned libs**: you may freeze versions in `requirements.txt` for strict repeats.

For stronger claims, you can run **multiple seeds** (e.g., 41/42/43) and report mean ± std of macro‑F1.

---

## 10) Small extensions (optional, easy marks)

- Swap backbone to `bert-base-uncased` or `roberta-base` and compare metrics/latency.
- Try a shorter `max_length` (like 96) for faster training; or 256 for potential gains.
- Class imbalance remedies: class weights or focal loss.
- Calibration check (ECE) and reliability plots.
- Export a few misclassified examples with predicted probability to inspect patterns.

---

## 11) Folder policy (keep repo light)

- **Do not** commit large datasets or model weights.  
- Use `.gitignore` to exclude `runs/`, `*.bin`, `*.safetensors`, `.ipynb_checkpoints/`, etc.

Suggested `.gitignore` entries:
```gitignore
runs/
*.bin
*.safetensors
.ipynb_checkpoints/
__pycache__/
.DS_Store
.venv/
```

---

## 12) A short reflection (2–4 sentences)

Full fine‑tuning of DistilBERT with a small LR (1e‑5) and early stopping provides a **stable** baseline for 3‑class sentiment. Macro‑F1 selection improves robustness against class imbalance. Most residual errors stem from ambiguous/short texts or context‑dependent sentiment (e.g., sarcasm). Future work may include larger backbones (RoBERTa), data augmentation, and class‑aware losses.

---

### At a glance (copy‑paste checklist)

- ✅ DistilBERT full FT, 3‑class, macro‑F1 early stop  
- ✅ Tokenization @ 128, dynamic padding  
- ✅ Best model & metrics saved to `runs/fpb_full_ft/`  
- ✅ Reproducible seeds & clear dependencies  
- ✅ Lightweight repo policy (no large binaries)  
