from .efficientnet import build_efficientnet

# 다른 백본들 추가 가능

def get_model(cfg):
    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name", "efficientnet").lower()
    input_shape = tuple(model_cfg.get("input_shape", [224, 224, 3]))
    num_classes = model_cfg.get("num_classes", 10)

    if name.startswith("efficientnet"):
        return build_efficientnet(cfg, num_classes=num_classes, input_shape=input_shape)
    else:
        raise ValueError(f"❌ 지원하지 않는 모델 이름입니다: {name}")