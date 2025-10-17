import os, argparse, json, numpy as np, torch
# make sure matplotlib works in headless mode
import matplotlib
matplotlib.use("Agg")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .data import autodetect_subset, read_finphrase_txt, split_to_datasets
from .tokenization import tokenize_datasets
from .trainers import train_full, train_lora, count_params

# CLI
def parse_args():
    p = argparse.ArgumentParser(description="FinPhrase TXT fine-tuning (Full vs LoRA)")
    # Data
    p.add_argument("--data_dir", type=str, default="./", help="Folder containing Sentences_*Agree.txt")
    p.add_argument("--subset_path", type=str, default=None, help="Explicit path to a TXT file (overrides auto-detect)")
    p.add_argument("--seed", type=int, default=42)
    # Model / tokenization
    p.add_argument("--checkpoint", type=str, default="distilbert-base-uncased")
    p.add_argument("--max_length", type=int, default=128)
    # Which run
    p.add_argument("--run", type=str, choices=["full","lora","both"], default="full")
    # Training hyperparams (full)
    p.add_argument("--full_outdir", type=str, default="runs/fpb_full_ft")
    p.add_argument("--full_lr", type=float, default=1e-5)
    p.add_argument("--full_epochs", type=int, default=20)
    p.add_argument("--full_bsz_train", type=int, default=16)
    p.add_argument("--full_bsz_eval", type=int, default=32)
    p.add_argument("--full_weight_decay", type=float, default=0.05)
    # Training hyperparams (lora)
    p.add_argument("--lora_outdir", type=str, default="runs/fpb_lora2")
    p.add_argument("--lora_lr", type=float, default=1e-5)
    p.add_argument("--lora_epochs", type=int, default=20)
    p.add_argument("--lora_bsz_train", type=int, default=16)
    p.add_argument("--lora_bsz_eval", type=int, default=32)
    p.add_argument("--lora_weight_decay", type=float, default=0.0)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", type=str, default="q_lin,v_lin",
                   help="Comma-separated target modules (e.g., q_lin,k_lin,v_lin,out_lin)")
    p.add_argument("--lora_eval_steps", type=int, default=100)
    p.add_argument("--lora_log_steps", type=int, default=100)
    # Mixed precision
    p.add_argument("--no_fp16", action="store_true", help="Disable fp16 even if CUDA is available")
    return p.parse_args()

def main():
    args = parse_args()
    print("Args:", vars(args))

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Data
    subset_path = args.subset_path or autodetect_subset(args.data_dir)
    print(f"Using subset file:", os.path.basename(subset_path))
    df = read_finphrase_txt(subset_path)

    print("Total samples:", len(df))
    print("Label counts:\n", df["label"].value_counts())
    print("Label ratio (%):\n", (df["label"].value_counts(normalize=True) * 100).round(2))
    print("\nSample rows:\n", df.sample(5, random_state=args.seed)[["sentence","label"]])

    ds_train, ds_valid, ds_test = split_to_datasets(df, seed=args.seed)
    print(f"Splits => train: {len(ds_train)} | valid: {len(ds_valid)} | test: {len(ds_test)}")

    tok, collator, tds_train, tds_valid, tds_test = tokenize_datasets(
        ds_train, ds_valid, ds_test, checkpoint=args.checkpoint, max_length=args.max_length
    )

    fp16 = not args.no_fp16

    full_trainer = None
    lora_trainer = None

    if args.run in ("full", "both"):
        full_trainer, full_sum = train_full(
            checkpoint=args.checkpoint,
            tds_train=tds_train, tds_valid=tds_valid, tds_test=tds_test,
            tok=tok, collator=collator,
            outdir=args.full_outdir, lr=args.full_lr, epochs=args.full_epochs,
            bsz_train=args.full_bsz_train, bsz_eval=args.full_bsz_eval,
            weight_decay=args.full_weight_decay, fp16=fp16, seed=args.seed
        )

    if args.run in ("lora", "both"):
        targets = tuple(t.strip() for t in args.lora_targets.split(",") if t.strip())
        lora_trainer, lora_sum = train_lora(
            checkpoint=args.checkpoint,
            tds_train=tds_train, tds_valid=tds_valid, tds_test=tds_test,
            tok=tok, collator=collator,
            outdir=args.lora_outdir, lr=args.lora_lr, epochs=args.lora_epochs,
            bsz_train=args.lora_bsz_train, bsz_eval=args.lora_bsz_eval,
            weight_decay=args.lora_weight_decay, fp16=fp16, seed=args.seed,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=targets, eval_steps=args.lora_eval_steps, log_steps=args.lora_log_steps
        )

    if args.run == "both":
        rows = []
        if full_trainer is not None:
            m = full_trainer.evaluate(tds_test)
            tot, trn, pct = count_params(full_trainer.model)
            rows.append({"run": "full", "acc": m.get("eval_accuracy"), "f1": m.get("eval_f1_macro"),
                         "trainable_%": round(pct, 2)})
        if lora_trainer is not None:
            m = lora_trainer.evaluate(tds_test)
            tot, trn, pct = count_params(lora_trainer.model)
            rows.append({"run": "lora", "acc": m.get("eval_accuracy"), "f1": m.get("eval_f1_macro"),
                         "trainable_%": round(pct, 2)})
        import pandas as pd
        df_cmp = pd.DataFrame(rows)
        print("\nComparison (TEST):\n", df_cmp)
        outdir = os.path.commonpath([args.full_outdir, args.lora_outdir]) if os.path.commonpath([args.full_outdir, args.lora_outdir]) else "."
        df_cmp.to_csv(os.path.join(outdir, "cmp_full_vs_lora.csv"), index=False)
    print("\nDone.")