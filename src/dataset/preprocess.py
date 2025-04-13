import tensorflow as tf

def parse_image(image, label=None, normalize=True):
    """
    단일 이미지에 대한 전처리 함수
    - image: tf.Tensor of shape (H, W, C)
    - label: Optional
    - normalize: True이면 [0,1] 정규화 적용
    """
    image = tf.cast(image, tf.float32)
    if normalize:
        image = image / 255.0  # [0,1] 정규화

    if label is not None:
        return image, label
    return image