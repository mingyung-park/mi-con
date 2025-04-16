import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from .regularizers import get_regularizer
from .head import build_mlp_head

def build_efficientnet(cfg, num_classes=10, input_shape=(224, 224, 3)):
    model_cfg = cfg.get("model", {})
    reg_cfg = model_cfg.get("regularizer", {})

    # 정규화 설정 (개별적으로 None일 수 있음)
    regularizers_dict = {
        "kernel": get_regularizer(reg_cfg.get("kernel")),
        "bias": get_regularizer(reg_cfg.get("bias")),
        "activity": get_regularizer(reg_cfg.get("activity")),
    }

    hidden_layers = model_cfg.get("hidden_layers", [])

    # base model
    base_model = EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling='avg'
    )
    base_model.trainable = model_cfg.get("trainable", True)

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs)

    # MLP Head (없으면 그냥 통과)
    x = build_mlp_head(x, hidden_layers, regularizers=regularizers_dict)

    # 최종 Dense layer 인자 구성 (None은 제외)
    output_kwargs = {
        "units": num_classes,
        "activation": model_cfg.get("final_activation", "softmax")  # default: softmax
    }
    if regularizers_dict["kernel"]:
        output_kwargs["kernel_regularizer"] = regularizers_dict["kernel"]
    if regularizers_dict["bias"]:
        output_kwargs["bias_regularizer"] = regularizers_dict["bias"]
    if regularizers_dict["activity"]:
        output_kwargs["activity_regularizer"] = regularizers_dict["activity"]

    outputs = layers.Dense(**output_kwargs)(x)

    return models.Model(inputs=inputs, outputs=outputs)
