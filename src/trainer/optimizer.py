import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers.schedules import (
    ExponentialDecay, CosineDecay, CosineDecayRestarts, LearningRateSchedule
)

def build_scheduler(sched_cfg):
    if not sched_cfg:
        return None

    sched_type = sched_cfg.get("type", "cosine").lower()
    init_lr = sched_cfg.get("initial_learning_rate", 0.001)
    decay_steps = sched_cfg.get("decay_steps", 1000)

    # 기본 scheduler
    if sched_type == "cosine":
        base_sched = CosineDecay(
            initial_learning_rate=init_lr,
            decay_steps=decay_steps,
            alpha=sched_cfg.get("alpha", 0.0)
        )
    elif sched_type == "exp":
        base_sched = ExponentialDecay(
            initial_learning_rate=init_lr,
            decay_steps=decay_steps,
            decay_rate=sched_cfg.get("decay_rate", 0.9),
            staircase=sched_cfg.get("staircase", True)
        )
    else:
        raise ValueError(f"지원되지 않는 scheduler type: {sched_type}")

    # warmup 설정
    warmup_cfg = sched_cfg.get("warmup")
    if warmup_cfg:
        return WarmUp(
            initial_lr=warmup_cfg.get("initial_lr", 1e-6),
            warmup_steps=warmup_cfg.get("steps", 200),
            base_schedule=base_sched
        )
    else:
        return base_sched


def build_optimizer(opt_cfg):
    if isinstance(opt_cfg, str):
        return optimizers.get(opt_cfg)

    name = opt_cfg.get("name", "adam").lower()

    # 학습률 스케줄러
    scheduler_cfg = opt_cfg.get("scheduler")
    if scheduler_cfg:
        learning_rate = build_scheduler(scheduler_cfg)
    else:
        learning_rate = opt_cfg.get("learning_rate", 0.001)

    weight_decay = opt_cfg.get("weight_decay", 0.0)

    if name == "adam":
        return optimizers.Adam(learning_rate=learning_rate)

    elif name == "adamw":
        return optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )

    elif name == "sgd":
        return optimizers.SGD(
            learning_rate=learning_rate,
            momentum=opt_cfg.get("momentum", 0.0),
            nesterov=opt_cfg.get("nesterov", False)
        )

    elif name == "sgdw":
        return optimizers.SGD(
            learning_rate=learning_rate,
            momentum=opt_cfg.get("momentum", 0.0),
            nesterov=opt_cfg.get("nesterov", False),
            weight_decay=weight_decay
        )

    elif name == "rmsprop":
        return optimizers.RMSprop(learning_rate=learning_rate)

    else:
        raise ValueError(f"지원되지 않는 optimizer: {name}")


class WarmUp(LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, base_schedule):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.base_schedule = base_schedule

    def __call__(self, step):
        warmup_lr = self.initial_lr + (self.base_schedule(0) - self.initial_lr) * (
            tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)
        )
        return tf.cond(
            step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: self.base_schedule(step - self.warmup_steps),
        )

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "base_schedule": self.base_schedule.get_config(),
        }