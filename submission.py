import cmrxrecon.data.modules
import cmrxrecon.models.cine
import cmrxrecon.models.mapping
import torch

torch.set_float32_matmul_precision("medium")

### MODEL SELECTION ###
TASK1_MODEL = cmrxrecon.models.cine.BasicUNet()
TASK2_MODEL = None
#######################


def task1(model):
    # Run the model for task 1
    # Save the data in the format expected by the evaluation script
    ...


def task2(model):
    # Run the model for task 2
    # Save the data in the format expected by the evaluation script
    ...


if __name__ == "__main__":
    print("Hello world!")
    if TASK1_MODEL is not None:
        task1(TASK1_MODEL)
    if TASK2_MODEL is not None:
        task2(TASK2_MODEL)
    print("Goodbye world!")
