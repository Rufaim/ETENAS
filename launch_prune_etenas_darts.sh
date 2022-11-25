#!/bin/bash

SAVE_FOLDER=${1:-"./output/prune-etenas-darts"}
BASE_DATAPATH=${2:-"./NAS_data"}

source ./venv/bin/activate
declare -a methods=(
    "mean,mean.json"
    "frobenius,frob.json"
    "relu_distance,dist.json"
    "expected_relu_100,exp_100.json"
    "expected_relu_100+relu_distance,exp_100_dist.json"
    "frob+relu_distance,frob_dist.json"
    "frobenius+expected_relu_100,frob_exp_100.json"
    "frobenius+expected_relu_100+relu_distance,frob_exp_100_dist.json"
    "frobenius+mean,frob_mean.json"
    "frobenius+mean+relu_distance,frob_mean_dist.json"
    "frobenius+mean+expected_relu_100,frob_mean_exp_100.json"
    "frobenius+mean+expected_relu_100+relu_distance,frob_mean_exp_100_dist.json"
    "mean+relu_distance,mean_dist.json"
    "mean+expected_relu_100,mean_exp_100.json"
    "mean+expected_relu_100+relu_distance,mean_exp_100_dist.json"
)


for elem in ${methods[@]}
do
  IFS=","; read -a method <<< "$elem"
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
      python3 ./prune_etenas.py --save_dir "${SAVE_FOLDER}/${method[0]}/${output_path}" --max_nodes 4 \
        --dataset ${dataset} --data_path "${BASE_DATAPATH}/${data_path}" --search_space_name darts --super_type nasnet-super \
        --arch_nas_dataset ${BASE_DATAPATH}/NAS-Bench-201-v1_0-e61699-simple.pkl --track_running_stats 1 --workers 0 \
        --init kaiming_norm --repeat 3 --batch_size 14 --prune_number 3 \
        --rankers_config "configs/rankers_configs/${method[1]}"
    done
  done
done


deactivate
