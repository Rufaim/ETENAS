# Enhanced Training-Free Neural Architecture Search (ETE-NAS)

[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

# Installation
* Clone this repo
* Install dependencies:
```bash
pip install -r requirements.txt
```

# Usage
### 0. Prepare the dataset
* Download NAS-Bench-201 `NAS-Bench-201-v1_0-e61699.pth` from [here](https://github.com/D-X-Y/NAS-Bench-201) and generate simplified API
```bash
python3 -m lib/nas_201_api -p <path to NAS-Bench-201-v1_0-e61699.pth> -o NAS_data/NAS-Bench-201-v1_0-e61699-simple.pkl
```
* Download DARTS JSONs from [here](https://github.com/facebookresearch/nds)
* Please follow the guideline [here](https://github.com/D-X-Y/AutoDL-Projects#requirements-and-preparation) to prepare ImageNet dataset, and also download NAS-Bench-201 database.

### 1. Search
#### [NAS-Bench-201 Space](https://openreview.net/forum?id=HJxyZkBKDr)
```python
python3 prune_etenas.py --save_dir ./output/prune-nas-bench-201/cifar10 --max_nodes 4 --dataset cifar10 --data_path NAS_data/cifar.python --search_space_name nas-bench-201 --super_type basic --arch_nas_dataset NAS_data/NAS-Bench-201-v1_0-e61699-simple.pkl --track_running_stats 1 --workers 0 --precision 3 --init kaiming_norm --repeat 3 --rand_seed 0 --batch_size 72 --prune_number 1 --rankers_config configs/rankers_configs/frob.json
python3 prune_etenas.py --save_dir ./output/prune-nas-bench-201/cifar100 --max_nodes 4 --dataset cifar10 --data_path NAS_data/cifar.python --search_space_name nas-bench-201 --super_type basic --arch_nas_dataset NAS_data/NAS-Bench-201-v1_0-e61699-simple.pkl --track_running_stats 1 --workers 0 --precision 3 --init kaiming_norm --repeat 3 --rand_seed 0 --batch_size 72 --prune_number 1 --rankers_config configs/rankers_configs/frob.json
python3 prune_etenas.py --save_dir ./output/prune-nas-bench-201/ImageNet16 --max_nodes 4 --dataset ImageNet16-120 --data_path NAS_data/ImageNet16 --search_space_name nas-bench-201 --super_type basic --arch_nas_dataset NAS_data/NAS-Bench-201-v1_0-e61699-simple.pkl --track_running_stats 1 --workers 0 --precision 3 --init kaiming_norm --repeat 3 --rand_seed 0 --batch_size 72 --prune_number 1 --rankers_config configs/rankers_configs/frob.json
```

#### [DARTS Space](https://openreview.net/forum?id=S1eYHoC5FX) ([NASNET](https://openaccess.thecvf.com/content_cvpr_2018/html/Zoph_Learning_Transferable_Architectures_CVPR_2018_paper.html))
```python
python3 prune_etenas.py --save_dir ./output/prune-darts/cifar10 --max_nodes 4 --dataset cifar10 --data_path NAS_data/cifar.python --search_space_name darts --super_type nasnet-super --track_running_stats 1 --workers 0 --precision 3 --init kaiming_norm --repeat 3 --rand_seed 0 --batch_size 72 --prune_number 3 --rankers_config configs/rankers_configs/frob.json
python3 prune_etenas.py --save_dir ./output/prune-darts/ImageNet16 --max_nodes 4 --dataset ImageNet16-120 --data_path NAS_data/ImageNet16 --search_space_name darts --super_type nasnet-super --track_running_stats 1 --workers 0 --precision 3 --init kaiming_norm --repeat 3 --rand_seed 0 --batch_size 72 --prune_number 3 --rankers_config configs/rankers_configs/frob.json
```

### 2. Evaluation
* For architectures searched on `nas-bench-201`, the accuracies are immediately available at the end of search (from the console output).
* For architectures searched on `darts`, please use [DARTS_evaluation](https://github.com/chenwydj/DARTS_evaluation) for training the searched architecture from scratch and evaluation.


# Citation
```
mock citation
```

# Acknowledgement
* Code base from [NAS-Bench-201](https://github.com/D-X-Y/AutoDL-Projects/blob/master/docs/NAS-Bench-201.md).
* Code base from [TENAS](https://github.com/VITA-Group/TENAS)
