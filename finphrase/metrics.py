import numpy as np
import evaluate

def build_metrics_fn():
    acc = evaluate.load("accuracy")
    f1  = evaluate.load("f1")
    def metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        return {
            "accuracy": acc.compute(predictions=preds, references=p.label_ids)["accuracy"],
            "f1_macro": f1.compute(predictions=preds, references=p.label_ids, average="macro")["f1"],
        }
    return metrics