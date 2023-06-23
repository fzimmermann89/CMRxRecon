import torch
import pytorch_lightning as pl
from cmrxrecon.models.base import BaseModel


class MappingModel(BaseModel):
    def __init__(self, lr=1e-3, weight_decay=1e-4, schedule=True):
        super().__init__()
