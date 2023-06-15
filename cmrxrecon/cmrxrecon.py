from lightning.pytorch.cli import LightningCLI
import cmrxrecon.data.modules
import cmrxrecon.models.cine
import cmrxrecon.models.mapping


if __name__ == "__main__":
    cli = LightningCLI()
