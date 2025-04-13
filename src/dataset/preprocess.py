import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

def parse_image(image, label=None, normalize=True):
    """
    단일 이미지에 대한 전처리 함수
    - image: tf.Tensor of shape (H, W, C)
    - label: Optional
    """
    image = tf.cast(image, tf.float32)

    if label is not None:
        return image, label
    return image


def encode_labels(labels, label_encoder=None):
    if label_encoder is None:
        label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return encoded_labels, label_encoder