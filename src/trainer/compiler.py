from tensorflow.keras import optimizers, losses, metrics as keras_metrics
from trainer.optimizer import build_optimizer

def compile_model(model, cfg):
    compile_cfg = cfg.get("compile", {})

    optimizer_cfg = compile_cfg.get("optimizer", "adam")
    optimizer = build_optimizer(optimizer_cfg)

    loss_name = compile_cfg.get("loss", "categorical_crossentropy")
    loss_fn = losses.get(loss_name)

    metric_names = compile_cfg.get("metrics", ["accuracy"])
    metric_fns = [keras_metrics.get(m) for m in metric_names]

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metric_fns
    )
    return model