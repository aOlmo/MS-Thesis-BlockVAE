#!/bin/bash

conf_base_dir="configurations"
dataset=$1
combinations=$2
one=1

f=$((combinations-one))

if [ -z "$1" ] || [ -z "$2" ]
  then
    echo "Usage: run_configs.sh <dataset> <number_of_config_files_to_run>"
    echo "---------------------------------------------------------------------------"
    echo "Example: run_configs.sh mnist 20 "
fi

for i in `seq 0 $f`;
do
	echo "Starting Block_VAE ${i}"
	echo "-----------------------------------------------------------"
	python block_vae.py -c ${conf_base_dir}/${dataset}_conf_${i}.cfg
    echo "-----------------------------------------------------------"
    echo "Starting Block_CNN ${i}"
	echo "-----------------------------------------------------------"
	python block_cnn.py -c ${conf_base_dir}/${dataset}_conf_${i}.cfg
	echo "-----------------------------------------------------------"
done

