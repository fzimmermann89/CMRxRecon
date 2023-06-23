from abc import ABC
from neptune.new.types import File as neptuneFile
from cmrxrecon.models.base import BaseModel


class CineModel(BaseModel):
    def __init__(self, lr=1e-3, weight_decay=1e-4, schedule=True):
        super().__init__()
