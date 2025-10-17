import os, re
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

CANDIDATES = [
    "Sentences_AllAgree.txt",   # prefer clean labels
    "Sentences_75Agree.txt",
    "Sentences_66Agree.txt",
    "Sentences_50Agree.txt"
]

LAB2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LAB = {v: k for k, v in LAB2ID.items()}

def autodetect_subset(data_dir: str) -> str:
    for name in CANDIDATES:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No Sentences_*Agree.txt found in {data_dir}.")

def read_finphrase_txt(path: str) -> pd.DataFrame:
    """支持：
       text @label  |  text\\tlabel  |  "text" , label  |  fallback: '... positive'
    """
    rows = []
    pat = re.compile(r'^\s*"?(.+?)"?\s*(?:@|\t|,)\s*(positive|negative|neutral)\s*$', re.I)
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    m = pat.match(line)
                    if m:
                        rows.append({"sentence": m.group(1), "label": m.group(2).lower()})
                    else:
                        parts = re.split(r'\s+', line)
                        lab = parts[-1].lower()
                        if lab in LAB2ID:
                            rows.append({"sentence": " ".join(parts[:-1]), "label": lab})
            break
        except UnicodeDecodeError:
            continue
    df = pd.DataFrame(rows).dropna().drop_duplicates()
    if len(df) == 0:
        raise ValueError("Parsed 0 rows. Check file format.")
    df["label_id"] = df["label"].map(LAB2ID)
    return df

def split_to_datasets(df: pd.DataFrame, seed: int = 42):
    train_df, test_df = train_test_split(
        df, test_size=0.10, random_state=seed, stratify=df["label_id"]
    )
    train_df, valid_df = train_test_split(
        train_df, test_size=0.10, random_state=seed, stratify=train_df["label_id"]
    )
    ds_train = Dataset.from_pandas(
        train_df[["sentence", "label_id"]].rename(columns={"label_id": "label"}),
        preserve_index=False
    )
    ds_valid = Dataset.from_pandas(
        valid_df[["sentence", "label_id"]].rename(columns={"label_id": "label"}),
        preserve_index=False
    )
    ds_test = Dataset.from_pandas(
        test_df[["sentence", "label_id"]].rename(columns={"label_id": "label"}),
        preserve_index=False
    )
    assert set(ds_train.features.keys()) == {"sentence", "label"}
    assert all(v in {0, 1, 2} for v in ds_train["label"])
    return ds_train, ds_valid, ds_test