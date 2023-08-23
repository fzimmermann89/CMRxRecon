#!/bin/sh
#Entrypoint for the submission

python cmrxrecon_cine.py predict --config pretrained/config.yaml --ckpt_path pretrained/checkpoint.ckpt --trainer.devices=[0,] --data.test_data_dir=/input --data.return_csm=False "$@"
