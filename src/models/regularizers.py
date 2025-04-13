from tensorflow.keras import regularizers

def get_regularizer(reg_cfg: dict):
    """
    config에 따라 정규화 함수 생성
    """
    if not reg_cfg or not reg_cfg.get("type"):
        return None

    reg_type = reg_cfg.get("type")
    l1 = reg_cfg.get("l1", 0.0)
    l2 = reg_cfg.get("l2", 0.0)

    if reg_type == "l1":
        return regularizers.l1(l1)
    elif reg_type == "l2":
        return regularizers.l2(l2)
    elif reg_type == "l1_l2":
        return regularizers.l1_l2(l1=l1, l2=l2)
    else:
        raise ValueError(f"지원되지 않는 regularizer type: {reg_type}")