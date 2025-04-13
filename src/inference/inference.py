import os
import numpy as np
import pandas as pd
import tensorflow as tf
from settings import *

def run_inference(model, test_ds, save_dir, file_name="submission.csv"):
    """
    모델로 추론 후 sample_submission 형식에 맞춰 저장
    - model: 학습된 keras model
    - test_ds: tf.data.Dataset (test 데이터셋)
    - sample_path: sample_submission.csv 경로
    - save_dir: 결과 저장 폴더 경로
    - name: 저장 파일 이름 (확장자는 자동으로 .csv)
    """
    # 예측
    preds = model.predict(test_ds, verbose=2)
    pred_labels = np.argmax(preds, axis=1)

    # sample submission 로드
    submission = pd.read_csv(SUB_CSV)
    submission["label"] = pred_labels

    # 저장 경로 생성
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{file_name}")
    submission.to_csv(save_path, index=False)

    print(f"✅ submission.csv 저장 완료! → {save_path}")
