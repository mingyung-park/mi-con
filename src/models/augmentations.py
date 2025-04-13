import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer

class Normalize(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, training=None):
        return tf.cast(inputs, tf.float32) / 255.0

class Posterize(Layer):
    def __init__(self, bits=4, **kwargs):
        super().__init__(**kwargs)
        self.bits = bits

    def call(self, inputs, training=None):
        if not training:
            return inputs
        shift = 8 - self.bits
        x = tf.cast(inputs * 255.0, tf.uint8)
        x = tf.bitwise.left_shift(tf.bitwise.right_shift(x, shift), shift)
        return tf.cast(x, tf.float32) / 255.0

def get_augmentation_layer(aug_cfg):
    if not aug_cfg.get("use", False):
        return tf.keras.Sequential([], name="no_augmentation")

    aug_layers = []

    # Custom
    if aug_cfg.get("posterize", 0) > 0:
        aug_layers.append(Posterize(bits=aug_cfg["posterize"]))

    # Built-in augmentations
    if aug_cfg.get("flip", False):
        aug_layers.append(layers.RandomFlip("horizontal"))
    if aug_cfg.get("rotation", 0) > 0:
        aug_layers.append(layers.RandomRotation(aug_cfg["rotation"]))
    if aug_cfg.get("zoom", 0) > 0:
        aug_layers.append(layers.RandomZoom(aug_cfg["zoom"]))
    if aug_cfg.get("translation", 0) > 0:
        aug_layers.append(layers.RandomTranslation(
            height_factor=aug_cfg["translation"],
            width_factor=aug_cfg["translation"]
        ))
    if aug_cfg.get("contrast", 0) > 0:
        aug_layers.append(layers.RandomContrast(aug_cfg["contrast"]))
    if aug_cfg.get("gaussian_noise", 0) > 0:
        aug_layers.append(layers.GaussianNoise(aug_cfg["gaussian_noise"]))

    return tf.keras.Sequential(aug_layers, name="augmentation")

def get_input_pipeline(cfg, aug_cfg):
    layers_list = []
    if aug_cfg.get("use", False):
        layers_list.append(get_augmentation_layer(aug_cfg))
    if cfg.get("normalize", True):
        layers_list.append(Normalize())
    return tf.keras.Sequential(layers_list, name="input_preprocessing")
