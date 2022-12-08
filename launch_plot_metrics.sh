#!/bin/bash

NUM_SAMPLES=${1:-250}
SAVE_FOLDER=${2:-"./output/plotted_metrics"}
BASE_DATAPATH=${3:-"./NAS_data"}

source ./venv/bin/activate

methods=( "synflow" "logsynflow" "zen_score"
        "acc_ntk" "mse_ntk" "lga_ntk" "fro_ntk" "mean_ntk" "cond_ntk" "eig_ntk"
        "acc_ntk_corr" "mse_ntk_corr" "lga_ntk_corr" "fro_ntk_corr" "mean_ntk_corr" "cond_ntk_corr" "eig_ntk_corr"
        "acc_nngp" "mse_nngp" "lga_nngp" "fro_nngp" "mean_nngp" "cond_nngp" "eig_nngp"
        "acc_nngp_corr" "mse_nngp_corr" "lga_nngp_corr" "fro_nngp_corr" "mean_nngp_corr" "cond_nngp_corr" "eig_nngp_corr"
        "acc_nngp_read" "mse_nngp_read" "lga_nngp_read" "fro_nngp_read" "mean_nngp_read" "cond_nngp_read" "eig_nngp_read"
        "acc_nngp_read_corr" "mse_nngp_read_corr" "lga_nngp_read_corr" "fro_nngp_read_corr" "mean_nngp_read_corr" "cond_nngp_read_corr" "eig_nngp_read_corr"
        "regs_dist_full" "regs_dist_max" "regs_dist_mean")


batch_size_default=72

for method in ${methods[@]}
do
  for dataset in "ImageNet16-120" "cifar10" "cifar100"
  do
    data_path="cifar.python"
    if [ $dataset == "ImageNet16-120" ]; then
      data_path="ImageNet16"
    fi
    if [ ${method:0:9} == "regs_dist" ]; then
      batch_size=150
    elif [ ${method:0:8} == "regs_num" ]; then
        batch_size=1000
    else
      batch_size=${batch_size_default}
    fi
    python3 ./plot_metric_distribution.py --save_path ${SAVE_FOLDER} --max_nodes 4 \
                              --dataset ${dataset} --data_path "${BASE_DATAPATH}/${data_path}" \
                              --arch_nas_dataset "${BASE_DATAPATH}/NAS-Bench-201-v1_0-e61699-simple.pkl" \
                              --track_running_stats 1 --workers 0 --rand_seed 0 --precision 3 --init kaiming_norm \
                               --repeat 3 --batch_size ${batch_size} --num_samples ${NUM_SAMPLES} --method ${method}
  done
done
