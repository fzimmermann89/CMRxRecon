from pytorch_lightning.cli import LightningCLI
import cmrxrecon.data.modules
import cmrxrecon.models.cine
import cmrxrecon.models.mapping


from pytorch_lightning.loggers.neptune import NeptuneLogger


if __name__ == "__main__":
    logger = NeptuneLogger(
        project="ptb/cmrxrecon-cine",
        log_model_checkpoints=False,
    )

    defaultargs = dict(
        logger=logger,
        max_epochs=10,
        log_every_n_steps=10,
        devices=[1],
    )

    cli = LightningCLI(
        cmrxrecon.models.cine.CineModel,
        cmrxrecon.data.modules.CineData,
        subclass_mode_data=True,
        subclass_mode_model=True,
        trainer_defaults=defaultargs,
    )
