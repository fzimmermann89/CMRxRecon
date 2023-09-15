#!/bin/sh
#Entrypoint for the MAPPING docker submission

python cmrxrecon_mapping.py predict --config pretrained/mapping/config.yaml --ckpt_path pretrained/mapping/checkpoint.ckpt --trainer.devices=[0,] --data.test_data_dir=/input/MultiCoil/Mapping/TestSet --data.return_csm=False --output.output_dir=/output --output.resize=False --output.swap=True --output.zip=False "$@"
