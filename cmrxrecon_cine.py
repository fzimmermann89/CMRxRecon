from lightning_fabric import loggers
from pytorch_lightning.cli import LightningCLI
import cmrxrecon.data.modules
from cmrxrecon.data.result_writer import OnlineValidationWriter
import cmrxrecon.models.cine
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
import torch

torch.set_float32_matmul_precision("medium")


class CLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        config = getattr(self.config, self.subcommand)
        if self.subcommand in ["fit"]:
            # logging only for certain subcommands
            neptunelogger = NeptuneLogger(
                project="ptb/cmrxrecon-cine",
                log_model_checkpoints=False,
            )
            tensorboardlogger = TensorBoardLogger(".")
            config.trainer.logger = [tensorboardlogger, neptunelogger]
        else:
            self.save_config_callback = None


if __name__ == "__main__":
    defaultargs = dict(
        logger=False,
        log_every_n_steps=10,
        devices=[2],
        accumulate_grad_batches=2,
        check_val_every_n_epoch=None,
        val_check_interval=250,
        max_steps=10000,
        max_epochs=None,
        callbacks=[OnlineValidationWriter()],
    )

    cli = CLI(
        cmrxrecon.models.cine.CineModel,
        cmrxrecon.data.modules.CineData,
        subclass_mode_model=True,
        trainer_defaults=defaultargs,
        auto_configure_optimizers=False,
    )
