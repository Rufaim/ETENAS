import argparse
import pickle
import copy
import os
from .api import NASBench201API


def get_accuracy_from_archinfo(info):
    output = {}
    for dset, subset in [("cifar10", "ori-test"), ("cifar100", "x-test"), ("ImageNet16-120", "x-test")]:
        output[dset] = info.get_metrics(dset, subset)["accuracy"]
    return output


def process_NAS_201(path):
    api = NASBench201API(path)

    archstr2index = copy.deepcopy(api.archstr2index)
    arch2metric = {}
    for idx in api.arch2infos_full:
        arch2metric[idx] = get_accuracy_from_archinfo(api.arch2infos_full[idx])
    return archstr2index, arch2metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Convert NAS-bench-201 API to simpified version")
    parser.add_argument("-p", "--path_nas_dataset", type=str, help="The path to load the architecture dataset")
    parser.add_argument("-o", "--output", type=str, help="output simplified dataset")
    args = parser.parse_args()

    assert args.output.endswith(".pkl")
    assert args.path_nas_dataset.endswith(".pth")

    dir = os.path.dirname(args.output)
    if len(dir) > 0:
        os.makedirs(dir, exist_ok=True)

    archstr2index, arch2metric = process_NAS_201(args.path_nas_dataset[0])

    with open(args.output, "wb") as f:
        pickle.dump(archstr2index, f)
        pickle.dump(arch2metric, f)
