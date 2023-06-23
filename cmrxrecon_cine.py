from pytorch_lightning.cli import LightningCLI
import cmrxrecon.data.modules
import cmrxrecon.models.cine
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
import torch

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    neptunelogger = NeptuneLogger(
        project="ptb/cmrxrecon-cine",
        log_model_checkpoints=False,
    )
    tensorboardlogger = TensorBoardLogger(".")

    defaultargs = dict(
        logger=[tensorboardlogger, neptunelogger],
        log_every_n_steps=10,
        devices=[1],
        accumulate_grad_batches=4,
        check_val_every_n_epoch=None,
        val_check_interval=250,
        max_steps=5000,
        max_epochs=None,
    )

    cli = LightningCLI(
        cmrxrecon.models.cine.CineModel,
        cmrxrecon.data.modules.CineData,
        subclass_mode_model=True,
        trainer_defaults=defaultargs,
        auto_configure_optimizers=False,
    )
