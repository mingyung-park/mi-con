from train.trainer import train_model
from tensorflow.keras import callbacks
import tensorflow as tf


def train_model(model, train_ds, val_ds, cfg):
    training_cfg = cfg.get("training", {})
    epochs = training_cfg.get("epochs", 20)
    callbacks_cfg = training_cfg.get("callbacks", {})
    
    callbacks = build_callbacks(callbacks_cfg)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )
    return history


def build_callbacks(callbacks_cfg):
    cb_list = []

    # ModelCheckpoint
    ckpt_cfg = callbacks_cfg.get("model_checkpoint", {})
    if ckpt_cfg.get("use", True):
        cb_list.append(callbacks.ModelCheckpoint(
            filepath=ckpt_cfg.get("filepath", "best_model.h5"),
            monitor=ckpt_cfg.get("monitor", "val_loss"),
            save_best_only=ckpt_cfg.get("save_best_only", True),
            save_weights_only=ckpt_cfg.get("save_weights_only", False),
            mode=ckpt_cfg.get("mode", "auto"),
            verbose=2
        ))

    # EarlyStopping
    es_cfg = callbacks_cfg.get("early_stopping", {})
    if es_cfg.get("use", False):
        cb_list.append(callbacks.EarlyStopping(
            monitor=es_cfg.get("monitor", "val_loss"),
            patience=es_cfg.get("patience", 5),
            mode=es_cfg.get("mode", "auto"),
            restore_best_weights=es_cfg.get("restore_best_weights", True),
            verbose=2
        ))

    # TensorBoard
    tb_cfg = callbacks_cfg.get("tensorboard", {})
    if tb_cfg.get("use", False):
        log_dir = tb_cfg.get("log_dir", "./logs")
        os.makedirs(log_dir, exist_ok=True)
        cb_list.append(callbacks.TensorBoard(log_dir=log_dir))

    return cb_list