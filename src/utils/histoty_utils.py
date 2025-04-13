import matplotlib.pyplot as plt
import json
import os

from settings import *


def save_history(history, file_name="history.json", cfg=None, save_dir="."):
    """
    실험 폴더에 학습 history, best scores, config를 저장
    """
    hist = history.history
    os.makedirs(save_dir, exist_ok=True)

    # Best metric 계산
    best_scores = {}
    if "val_loss" in hist:
        best_scores["best_val_loss"] = min(hist["val_loss"])
    if "val_accuracy" in hist:
        best_scores["best_val_accuracy"] = max(hist["val_accuracy"])
    if "val_top_k_categorical_accuracy" in hist:
        best_scores["best_val_top_k_accuracy"] = max(hist["val_top_k_categorical_accuracy"])

    # 저장할 딕셔너리 구성
    save_dict = {
        "history": hist,
        "best_scores": best_scores,
    }

    # config가 있다면 함께 저장
    if cfg is not None:
        save_dict["config"] = cfg

    # 파일로 저장
    save_path = os.path.join(save_dir, file_name)
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