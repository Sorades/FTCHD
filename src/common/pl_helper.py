from pathlib import Path

from lightning import Trainer


def resolve_pl_dir(trainer: "Trainer"):
    """Determines model checkpoint save directory at runtime. Reference attributes from the trainer's logger to
    determine where to save checkpoints. The path for saving weights is set in this priority:

    1.  The ``ModelCheckpoint``'s ``dirpath`` if passed in
    2.  The ``Logger``'s ``log_dir`` if the trainer has loggers
    3.  The ``Trainer``'s ``default_root_dir`` if the trainer has no loggers

    """

    if len(trainer.loggers) > 0:
        if trainer.loggers[0].save_dir is not None:
            save_dir = trainer.loggers[0].save_dir
        else:
            save_dir = trainer.default_root_dir
        name = trainer.loggers[0].name
        version = trainer.loggers[0].version
        version = version if isinstance(version, str) else f"version_{version}"
        ckpt_path = Path(save_dir) / str(name) / version
    else:
        # if no loggers, use default_root_dir
        ckpt_path = Path(trainer.default_root_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    return ckpt_path
