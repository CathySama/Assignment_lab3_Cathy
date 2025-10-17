import os
import matplotlib.pyplot as plt

def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def plot_series(x, y, xlabel, ylabel, title, out_png):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title); plt.grid(True)
    _ensure_dir(out_png)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def plot_confusion(cm, labels, title, out_png, normalize=False):
    import numpy as np
    plt.figure()
    data = cm if not normalize else (cm / cm.sum(axis=1, keepdims=True))
    plt.imshow(data, interpolation='nearest')
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            txt = f"{data[i, j]:.2f}" if normalize else f"{data[i, j]}"
            plt.text(j, i, txt, ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True")
    _ensure_dir(out_png)
    plt.tight_layout(); plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()