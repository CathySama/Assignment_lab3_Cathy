# DistilBERT Full Fine-Tuning (3‑Class) 

This repository accompanies the notebook **`lab3-2.ipynb`**, which performs **full fine‑tuning and LoRA fine-tuning** of a pretrained Transformer (**`distilbert-base-uncased`**) for a **3‑class sentiment** task. It uses **PyTorch + HuggingFace Transformers**, early stopping, and macro‑F1 model selection. The code is clean, simple, and ready to reproduce on your machine.


---

## 1) What this project does

- **Task**: Multi‑class text classification (3 classes).
- **Backbone**: `distilbert-base-uncased` (a compact, distilled BERT with good speed/quality tradeoff).
- **Frameworks**: PyTorch, HuggingFace Transformers/Datasets, Evaluate, scikit‑learn.
- **Training**: Full fine‑tuning, LoRA.
- **Outputs**: Best checkpoint and metrics saved under `runs/fpb_full_ft(or fpb_lora2)/`.

```python
├─ main.py                      
├─ finphrase/
│  ├─ __init__.py
│  ├─ cli.py                    
│  ├─ data.py                   
│  ├─ tokenization.py            
│  ├─ metrics.py                
│  ├─ plots.py                   
│  └─ trainers.py               
└─ requirements.txt         
```

---

## 2) Environment
- OS: Linux
- Python: **3.10.18**
  
pip install -r requirements.txt

---

## 3) Data (simple & flexible)

Source: https://huggingface.co/datasets/takala/financial_phrasebank
  
This project reads a single .txt file with sentence–label pairs and builds train/validation/test splits. 

The loader automatically scans for a subset file in the current directory (DATA_DIR="./") using this priority:
	1.	Sentences_AllAgree.txt
	2.	Sentences_75Agree.txt
	3.	Sentences_66Agree.txt
	4.	Sentences_50Agree.txt

Put one of these files next to main.py (or adjust DATA_DIR). The first one found is used.

---

## 4) Training Configuration (exactly as coded)

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
    fp16=True,                    
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_steps=100,
    report_to="none",
    seed=42,
    greater_is_better=True
)
```

```python
TrainingArguments(
    output_dir="runs/fpb_lora2",
    eval_strategy="steps",     
    eval_steps=100,                  
    save_strategy="steps",            
    save_steps=100,                  
    save_total_limit=2,

    load_best_model_at_end=True, 
    metric_for_best_model="f1_macro", 
    greater_is_better=True,
    learning_rate=1e-5,
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=32,
    num_train_epochs=20, 
    weight_decay=0.0, 
    fp16=torch.cuda.is_available(),

    logging_strategy="steps",
    logging_steps=100,

    report_to="none", seed=42
)
```

---

## 5) How to run

```python

# Method 1
python -m finphrase.cli --run full
python -m finphrase.cli --run lora --lora_targets q_lin,k_lin,v_lin,out_lin
python -m finphrase.cli --run both

# Method 2
python main.py --run both

# Method 3
# Run lab3.ipynb file# 
```

---

