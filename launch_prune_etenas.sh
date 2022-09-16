#!/bin/bash

NUM_SAMPLES=${1:-250}
SAVE_FOLDER=${2:-"./output/plotted_metrics"}
BASE_DATAPATH=${3:-"./NAS_data"}

source ./venv/bin/activate

#declare -a methods=(
#    "mean,mean.json"
#    "frobenius,frob.json"
#    "condition,cond.json"
#    "eigenvalue_score,eig.json"
#    "relu_distance,dist.json"
#    "expected_relu_1000,exp_1000.json"
#    "expected_relu_500,exp_500.json"
#    "expected_relu_100,exp_100.json"
#    "lga,lga.json"
#    "frobenius+lga,frob_lga.json"
#    "all,all.json"
#)
declare -a methods=(
#  "mean_test,mean.json"
  "eigenvalue_score_test,eig.json"
)



for elem in ${methods[@]}
do
  IFS=",";read -a method <<< "$elem"
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
      python3 ./prune_etenas.py --save_dir "./output/prune-etenas-nas-bench-201/${method[0]}/${output_path}" --max_nodes 4 \
        --dataset ${dataset} --data_path "NAS_data/${data_path}" --search_space_name nas-bench-201 --super_type basic \
        --arch_nas_dataset ./NAS_data/NAS-Bench-201-v1_0-e61699-simple.pkl --track_running_stats 1 --workers 0 \
        --precision 3 --init kaiming_norm --repeat 3 --batch_size 72 --prune_number 1 \
        --rankers_config "configs/rankers_configs/${method[1]}"
    done
  done
done
