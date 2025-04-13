from tensorflow.keras import layers, initializers

def build_mlp_head(x, layer_cfg_list=None, regularizers=None):
    if not layer_cfg_list:
        return x

    kernel_reg = regularizers.get("kernel") if regularizers else None
    bias_reg = regularizers.get("bias") if regularizers else None
    activity_reg = regularizers.get("activity") if regularizers else None

    for layer_cfg in layer_cfg_list:
        # -------------레이어 설정
        units = layer_cfg.get("units")
        if not units:
            continue  # units가 없는 경우 skip

        # Optional config: 모두 기본값 처리
        dropout = layer_cfg.get("dropout", 0.0)
        initializer_name = layer_cfg.get("initializer", None)
        activation = layer_cfg.get("activation", "relu")
        use_bias = layer_cfg.get("use_bias", True)
        use_bn = layer_cfg.get("batch_norm", False)

        # Dense kwargs 구성 (initializer 포함 여부도 조건적으로)
        dense_kwargs = {
            "units": units,
            "use_bias": use_bias,
            "activation": None if use_bn else activation,  # BN 있는 경우 activation 분리 적용
            "kernel_regularizer": kernel_reg,
            "bias_regularizer": bias_reg,
            "activity_regularizer": activity_reg,
        }

        # Optional: initializer 설정
        if initializer_name:
            try:
                dense_kwargs["kernel_initializer"] = initializers.get(initializer_name)
            except ValueError:
                raise ValueError(f"Invalid initializer name: {initializer_name}")

        # -------------레이어 구성
        # Dense → BatchNorm → Activation → Dropout
        x = layers.Dense(**dense_kwargs)(x)

        if use_bn:
            x = layers.BatchNormalization()(x)
            if activation:
                x = layers.Activation(activation)(x)

        if dropout > 0:
            x = layers.Dropout(dropout)(x)

    return x
