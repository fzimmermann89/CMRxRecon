from abc import ABC
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from neptune.new.types import File as neptuneFile


class MappingModel(pl.LightningModule, ABC):
    ...
