import matplotlib.pyplot as plt
import json
import os

def save_history(history, cfg=None, save_path="history.json"):
    """
    Keras history + best scores + 사용한 config를 저장
    """
    hist = history.history
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Best metric 계산
    best_scores = {}
    if "val_loss" in hist:
        best_scores["best_val_loss"] = min(hist["val_loss"])
    if "val_accuracy" in hist:
        best_scores["best_val_accuracy"] = max(hist["val_accuracy"])
    if "val_top_k_categorical_accuracy" in hist:
        best_scores["best_val_top_k_accuracy"] = max(hist["val_top_k_categorical_accuracy"])

    # 저장 dict 구성
    save_dict = {
        "history": hist,
        "best_scores": best_scores
    }

    # config 저장
    if cfg is not None:
        save_dict["config"] = cfg  # dict 형태로 바로 저장 가능

    # 저장
    with open(save_path, "w") as f:
        json.dump(save_dict, f, indent=2)

    print(f"✅ History + Best Scores + Config saved to {save_path}")

def plot_history(history, metrics=("loss", "accuracy")):
    """
    Keras history 객체를 받아 metric 별 학습 그래프를 그립니다.
    """
    hist = history.history
    for metric in metrics:
        if metric in hist:
            plt.figure()
            plt.plot(hist[metric], label=f"train_{metric}")
            val_key = f"val_{metric}"
            if val_key in hist:
                plt.plot(hist[val_key], label=f"val_{metric}")
            plt.title(f"{metric.title()} over epochs")
            plt.xlabel("Epochs")
            plt.ylabel(metric.title())
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print(f"⚠️ metric '{metric}' not found in history.")