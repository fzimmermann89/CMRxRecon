#!/bin/sh
#Entrypoint for the submission

python cmrxrecon_cine.py predict --config pretrained/config.yaml --ckpt_path pretrained/checkpoint.ckpt --trainer.devices=[0,] --data.test_data_dir=/input/MultiCoil/Cine/TestSet --data.return_csm=False --output.output_dir=/output --output.resize=False --output.swap=True --output.zip=False "$@"
