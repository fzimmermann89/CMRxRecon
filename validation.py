from pathlib import Path
import tempfile
from cmrxrecon.data.submit_utils import run4Ranking
import numpy as np

## For validation we need to:
# 1. run the model on the validation data
# 2. save the output in the format expected by the evaluation script
# a) reorder the axes to (sx, sy, sz, t/w)
# b) crop the output using the run4Ranking function
# c) save the output as a .mat file
# d) compress the files into .zip files
