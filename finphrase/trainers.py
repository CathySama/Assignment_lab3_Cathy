import os, json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model

from .metrics import build_metrics_fn
from .plots import plot_series, plot_confusion

def count_params(m):
    tot = sum(p.numel() for p in m.parameters())
    trn = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return tot, trn, 100.0 * trn / tot

def _save_eval_curves(logs_df: pd.DataFrame, outdir: str, prefix: str = ""):
    eval_cols = [c for c in logs_df.columns if c.startswith("eval_")]
    df_eval = logs_df.dropna(subset=eval_cols).groupby("step", as_index=False).last()
    if "eval_loss" in df_eval:
        plot_series(df_eval["step"], df_eval["eval_loss"], "step", "eval_loss",
                    f"{prefix}Eval loss vs step", os.path.join(outdir, f"{prefix.lower().strip()}eval_loss.png"))
    metric_cols = [m for m in eval_cols if m not in {
        "eval_loss","eval_runtime","eval_samples_per_second","eval_steps_per_second",
        "eval_mem_cpu_alloc_delta","eval_mem_gpu_alloc_delta"
    }]
    if len(metric_cols) > 0 and len(df_eval) > 0:
        import matplotlib.pyplot as plt
        plt.figure()
        for m in metric_cols:
            plt.plot(df_eval["step"], df_eval[m], label=m[5:])
        plt.xlabel("step"); plt.ylabel("metric"); plt.title(f"{prefix}Eval metrics"); plt.legend(); plt.grid(True)
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, f"{prefix.lower().strip()}eval_metrics.png"), dpi=200, bbox_inches="tight")
        plt.close()

# Full fine-tuning
def train_full(checkpoint, tds_train, tds_valid, tds_test, tok, collator,
               outdir="runs/fpb_full_ft", lr=1e-5, epochs=20,
               bsz_train=16, bsz_eval=32, weight_decay=0.05, fp16=True, seed=42):
    os.makedirs(outdir, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)
    metrics_fn = build_metrics_fn()
    args = TrainingArguments(
        output_dir=outdir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=bsz_train,
        per_device_eval_batch_size=bsz_eval,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        fp16=bool(fp16 and torch.cuda.is_available()),
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=100,
        report_to="none",
        seed=seed,
        greater_is_better=True
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=tds_train, eval_dataset=tds_valid,
        tokenizer=tok, data_collator=collator,
        compute_metrics=metrics_fn
    )
    trainer.train()

    valid_metrics = trainer.evaluate()
    test_metrics  = trainer.evaluate(tds_test)
    print("VALID =>", valid_metrics)
    print("TEST  =>", test_metrics)

    logs = pd.DataFrame(trainer.state.log_history)
    logs.to_csv(os.path.join(outdir, "training_log.csv"), index=False)

    tr = logs[logs["loss"].notna()][["step", "loss"]]
    if len(tr) > 0:
        plot_series(tr["step"], tr["loss"], "step", "train_loss",
                    "Training loss", os.path.join(outdir, "train_loss.png"))
    _save_eval_curves(logs, outdir, prefix="")

    pred = trainer.predict(tds_test)
    logits = pred.predictions
    y_true = pred.label_ids
    y_pred = logits.argmax(axis=-1)

    print("Classification report (test):")
    print(classification_report(y_true, y_pred,
                                target_names=["negative","neutral","positive"], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    pd.DataFrame(cm, index=["T_neg","T_neu","T_pos"], columns=["P_neg","P_neu","P_pos"]) \
      .to_csv(os.path.join(outdir, "cm_abs.csv"))
    plot_confusion(cm, ["negative","neutral","positive"], "Confusion Matrix (abs)",
                   os.path.join(outdir, "cm_abs.png"))
    plot_confusion(cm, ["negative","neutral","positive"], "Confusion Matrix (row-normalized)",
                   os.path.join(outdir, "cm_norm.png"), normalize=True)

    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    conf  = probs.max(axis=1)
    margin = np.partition(probs, -2, axis=1)[:, -1] - np.partition(probs, -2, axis=1)[:, -2]
    raw_sentences = tds_test.remove_columns(["input_ids","attention_mask"]).to_pandas()["sentence"].tolist()
    errors = []
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if t != p:
            errors.append({
                "sentence": raw_sentences[i],
                "true": ["negative","neutral","positive"][t],
                "pred": ["negative","neutral","positive"][p],
                "conf": float(conf[i]),
                "margin": float(margin[i])
            })
    pd.DataFrame(errors).sort_values(["conf","margin"], ascending=[False, False]) \
      .to_csv(os.path.join(outdir, "errors_full_ft.csv"), index=False)

    trainer.save_model(os.path.join(outdir, "best_model"))
    tok.save_pretrained(os.path.join(outdir, "best_model"))

    tot, trn, pct = count_params(trainer.model)
    summary = {
        "valid": valid_metrics, "test": test_metrics,
        "params_total": int(tot), "params_trainable": int(trn),
        "trainable_pct": round(pct, 2)
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return trainer, summary


# LORA training
def train_lora(checkpoint, tds_train, tds_valid, tds_test, tok, collator,
               outdir="runs/fpb_lora2", lr=1e-5, epochs=20,
               bsz_train=16, bsz_eval=32, weight_decay=0.0, fp16=True, seed=42,
               lora_r=8, lora_alpha=16, lora_dropout=0.05,
               target_modules=("q_lin","v_lin"), eval_steps=100, log_steps=100):
    os.makedirs(outdir, exist_ok=True)
    base = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)
    lcfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=lora_r, lora_alpha=lora_alpha,
                      lora_dropout=lora_dropout, target_modules=list(target_modules))
    model = get_peft_model(base, lcfg)

    metrics_fn = build_metrics_fn()
    args = TrainingArguments(
        output_dir=outdir,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        learning_rate=lr,
        per_device_train_batch_size=bsz_train,
        per_device_eval_batch_size=bsz_eval,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        fp16=bool(fp16 and torch.cuda.is_available()),
        logging_strategy="steps",
        logging_steps=log_steps,
        report_to="none",
        seed=seed
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=tds_train, eval_dataset=tds_valid,
        tokenizer=tok, data_collator=collator,
        compute_metrics=metrics_fn
    )
    trainer.train()

    test_metrics = trainer.evaluate(tds_test)
    print("TEST (LoRA) =>", test_metrics)

    log = pd.DataFrame(trainer.state.log_history)
    log.to_csv(os.path.join(outdir, "training_log.csv"), index=False)

    tr = log.dropna(subset=["loss"])
    if len(tr) > 0:
        plot_series(tr["step"], tr["loss"], "step", "train_loss",
                    "LoRA: train loss", os.path.join(outdir, "lora_train_loss.png"))
    ev = log.dropna(subset=["eval_loss"])
    if len(ev) > 0:
        plot_series(ev["step"], ev["eval_loss"], "step", "eval_loss",
                    "LoRA: eval loss", os.path.join(outdir, "lora_eval_loss.png"))
  
    from .plots import plt as _plt 
    eval_cols = [c for c in log.columns if c.startswith("eval_")]
    metric_cols = [m for m in eval_cols if m not in {
        "eval_loss","eval_runtime","eval_samples_per_second","eval_steps_per_second",
        "eval_mem_cpu_alloc_delta","eval_mem_gpu_alloc_delta"
    }]
    if len(metric_cols) > 0 and len(ev) > 0:
        import matplotlib.pyplot as plt
        plt.figure()
        for m in metric_cols:
            plt.plot(ev["step"], ev[m], label=m[5:])
        plt.xlabel("step"); plt.ylabel("metric"); plt.title("LoRA: eval metrics"); plt.legend(); plt.grid(True)
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, "lora_eval_metrics.png"), dpi=200, bbox_inches="tight")
        plt.close()

    pred = trainer.predict(tds_test)
    y_true = pred.label_ids
    y_pred = pred.predictions.argmax(axis=-1)
    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred, target_names=["neg","neu","pos"], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    pd.DataFrame(cm, index=["T_neg","T_neu","T_pos"], columns=["P_neg","P_neu","P_pos"]) \
      .to_csv(os.path.join(outdir, "cm_lora.csv"))
    plot_confusion(cm, ["neg","neu","pos"], "LoRA: Confusion Matrix (abs)",
                   os.path.join(outdir, "cm_abs.png"))
    plot_confusion(cm, ["neg","neu","pos"], "LoRA: Confusion Matrix (row-normalized)",
                   os.path.join(outdir, "cm_norm.png"), normalize=True)

    probs = torch.softmax(torch.from_numpy(pred.predictions), dim=-1).numpy()
    conf = probs.max(axis=1)
    err = []
    raw_sentences = tds_test.remove_columns(["input_ids","attention_mask"]).to_pandas()["sentence"].tolist()
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if t != p:
            err.append({"true": t, "pred": p, "conf": float(conf[i]), "sentence": raw_sentences[i]})
    pd.DataFrame(sorted(err, key=lambda x: -x["conf"]))[:20] \
      .to_csv(os.path.join(outdir, "errors_lora_top20.csv"), index=False)

    trainer.save_model(os.path.join(outdir, "best_model"))
    tok.save_pretrained(os.path.join(outdir, "best_model"))

    tot, trn, pct = count_params(trainer.model)
    summary = {
        "test": test_metrics,
        "params_total": int(tot), "params_trainable": int(trn),
        "trainable_pct": round(pct, 2)
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return trainer, summary