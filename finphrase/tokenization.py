from transformers import AutoTokenizer, DataCollatorWithPadding

def tokenize_datasets(ds_train, ds_valid, ds_test, checkpoint: str, max_length: int = 128):
    tok = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

    def tok_fn(b):
        return tok(b["sentence"], truncation=True, max_length=max_length)

    tds_train = ds_train.map(tok_fn, batched=True)
    tds_valid = ds_valid.map(tok_fn, batched=True)
    tds_test  = ds_test .map(tok_fn, batched=True)

    keep = ["input_ids", "attention_mask", "label"]
    tds_train = tds_train.remove_columns([c for c in tds_train.column_names if c not in keep])
    tds_valid = tds_valid.remove_columns([c for c in tds_valid.column_names if c not in keep])
    tds_test  = tds_test .remove_columns([c for c in tds_test .column_names  if c not in keep])

    collator = DataCollatorWithPadding(tok)
    return tok, collator, tds_train, tds_valid, tds_test