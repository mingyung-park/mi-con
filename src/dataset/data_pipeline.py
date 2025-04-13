import tensorflow as tf
from .preprocess import parse_image
import numpy as np

def build_dataet(images, labels=None, batch_size=32, training=True, normalize=True):
    """
    NumPy 데이터를 TensorFlow Dataset으로 변환 (예외처리 포함)
    """
    # --- 기본 검증 ---
    if not isinstance(images, (tf.Tensor, list, tuple)):
        if not isinstance(images, np.ndarray):
            raise TypeError(f"`images`는 NumPy array 또는 Tensor여야 합니다. 현재: {type(images)}")
        if images.ndim != 4:
            raise ValueError(f"`images`는 4차원이어야 합니다.(N_samples,X,Y,dim)\n 현재 shape: {images.shape}")

    if labels is not None and len(images) != len(labels):
        raise ValueError(f"`images`와 `labels`의 길이가 일치하지 않습니다. {len(images)} vs {len(labels)}")

    if batch_size <= 0:
        raise ValueError(f"`batch_size`는 1 이상의 자연수여야 합니다. 현재: {batch_size}")

    # --- Dataset 생성 ---
    try:
        # train
        if labels is not None:
            dataset = tf.data.Dataset.from_tensor_slices((images, labels))
            dataset = dataset.map(lambda x, y: parse_image(x, y, normalize), num_parallel_calls=4)
        # test
        else:
            dataset = tf.data.Dataset.from_tensor_slices(images)
            dataset = dataset.map(lambda x: parse_image(x, normalize=normalize), num_parallel_calls=4)
    except Exception as e:
        raise RuntimeError(f"Dataset 생성 중 오류 발생: {e}")

    if training:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)

    return dataset