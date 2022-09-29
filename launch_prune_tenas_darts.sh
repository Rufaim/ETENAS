#!/bin/bash

SAVE_FOLDER=${1:-"./output/prune-tenas-darts"}
BASE_DATAPATH=${2:-"./NAS_data"}

source ./venv/bin/activate


for dataset in "ImageNet16-120" "cifar10" "cifar100"; do
  if [ $dataset == "ImageNet16-120" ]; then
    data_path="ImageNet16"
    output_path="ImageNet16"
  elif [ $dataset == "cifar100" ]; then
    data_path="cifar.python"
    output_path="cifar100"
  elif [ $dataset == "cifar10" ]; then
    data_path="cifar.python"
    output_path="cifar10"
  fi
  for run in {1..10}; do
    python3 ./prune_tenas.py --save_dir "${SAVE_FOLDER}/${output_path}" --max_nodes 4 \
      --dataset ${dataset} --data_path "${BASE_DATAPATH}/${data_path}" --search_space_name darts --super_type nasnet-super \
      --arch_nas_dataset ${BASE_DATAPATH}/NAS-Bench-201-v1_0-e61699-simple.pkl --track_running_stats 1 --workers 0 \
      --precision 3 --init kaiming_norm --repeat 3 --batch_size 14 --prune_number 3
  done
done


deactivate
