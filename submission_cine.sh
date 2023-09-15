#!/bin/sh
#Entrypoint for the CINE docker submission

python cmrxrecon_cine.py predict --config pretrained/cine/config.yaml --ckpt_path pretrained/cine/checkpoint.ckpt --trainer.devices=[0,] --data.test_data_dir=/input/MultiCoil/Cine/TestSet --data.return_csm=False --data.return_kfull=False --output.output_dir=/output --output.resize=False --output.swap=True --output.zip=False "$@"
