import torch
import os.path
from pathlib import Path
from jsonargparse import lazy_instance
from pytorch_lightning import Callback
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.loggers import TensorBoardLogger

import cmrxrecon.data.modules
import cmrxrecon.models.cine
from cmrxrecon.data.result_writer import OnlineValidationWriter

torch.set_float32_matmul_precision("medium")


class LogChkptPath(Callback):
    def on_train_start(self, trainer, pl_module):
        modelname = pl_module.__class__.__name__
        log_dir = Path(trainer.log_dir).absolute()
        for logger in trainer.loggers:
            logger.log_hyperparams({"log_dir": str(log_dir), "model_name": modelname})


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(OnlineValidationWriter, "output")

    def before_instantiate_classes(self) -> None:
        config = getattr(self.config, self.subcommand)
        if "ckpt_path" in config:
            ckpt_path = config["ckpt_path"]
            if ckpt_path is not None and os.path.isdir(ckpt_path):
                chkpt = max(Path(ckpt_path).rglob("*.ckpt"), key=os.path.getmtime)
                config["ckpt_path"] = str(chkpt)
            elif ckpt_path == "latest" and "config" in config and len(config["config"]) > 0:
                chkpt = max(Path(config["config"][0]).parent.rglob("*.ckpt"), key=os.path.getmtime)
                config["ckpt_path"] = str(chkpt)

        if self.subcommand in ["fit"]:
            # logging only for certain subcommands
            neptunelogger = {
                "class_path": "NeptuneLogger",
                "init_args": dict(
                    project="ptb/cmrxrecon-cine",
                    log_model_checkpoints=False,
                ),
            }
            tensorboardlogger = {
                "class_path": "TensorBoardLogger",
                "init_args": dict(save_dir="."),
            }
            config.trainer.logger = [tensorboardlogger, neptunelogger]
        elif self.subcommand in ["validate", "test"] and "ckpt_path" in config:
            chkpath = Path(config["ckpt_path"])
            csvlogger = {
                "class_path": "CSVLogger",
                "init_args": dict(
                    save_dir=str(chkpath.parent),
                    name="validation",
                    version=str(chkpath.stem),
                ),
            }
            config.trainer.logger = [csvlogger]
            self.save_config_callback = None

        else:
            config.trainer.logger = None
            self.save_config_callback = None


if __name__ == "__main__":
    defaultargs = dict(
        logger=False,
        log_every_n_steps=10,
        devices=[0],
        accumulate_grad_batches=4,
        check_val_every_n_epoch=None,
        val_check_interval=250,
        max_steps=10000,
        max_epochs=None,
        callbacks=[lazy_instance(LogChkptPath), lazy_instance(OnlineValidationWriter)],
    )

    cli = CLI(
        cmrxrecon.models.cine.CineModel,
        cmrxrecon.data.modules.CineData,
        subclass_mode_model=True,
        trainer_defaults=defaultargs,
        auto_configure_optimizers=False,
    )
