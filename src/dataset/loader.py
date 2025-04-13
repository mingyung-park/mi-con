import numpy as np
import pandas as pd
import os


def load_csv(csv_path):
    """
    CSV 파일에서 이미지 데이터와 레이블, ID 불러오기
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"CSV 파일 로딩 중 오류 발생: {e}")

    if "ID" not in df.columns:
        raise KeyError("'ID' 컬럼이 존재하지 않습니다.")        

    image_ids = df["ID"].values
    labels = df["label"].values if "label" in df.columns else None

    try:
        image_data = df.drop(columns=["ID", "label"], errors='ignore').values
    except Exception as e:
        raise ValueError(f"이미지 데이터를 추출할 수 없습니다: {e}")

    # reshape 가능 여부 확인
    num_pixels = image_data.shape[1]
    expected_pixels = 32 * 32
    if num_pixels != expected_pixels:
        raise ValueError(f"이미지 픽셀 수가 맞지 않습니다: {num_pixels} (예상: {expected_pixels})")

    try:
        images = image_data.reshape(-1, 32, 32, 1).astype(np.uint8)
    except Exception as e:
        raise ValueError(f"이미지 reshape 중 오류 발생: {e}")

    return images, labels, image_ids