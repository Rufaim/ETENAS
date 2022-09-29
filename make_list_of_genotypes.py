import re
import os
import numpy as np
import argparse
from glob import glob

DATASETS = ["cifar10", "cifar100", "ImageNet16"]
TIME_REGEX = re.compile(r"Time spent: (\d+.\d+) s")
GEN_REGEX = re.compile(r"Genotype\(normal=\[.+\], reduce_concat=\[(\d(, )?)+\]\)")


def parse_logfile(path):
    with open(path) as file:
        text = file.read()
    time_match = TIME_REGEX.search(text)
    assert time_match is not None
    time_match = float(time_match.groups(1)[0])
    gen_match = GEN_REGEX.search(text)
    assert gen_match is not None
    gen_match = gen_match.group()
    return time_match, gen_match


def means_n_vars(d):
    mean = np.mean(d)
    std = np.std(d)
    return f"{mean:.02f}({std:.02f})"


def main(input, output, basename):
    time_lines = []
    gen_lines = []
    for dset in DATASETS:
        time, gens = zip(*(parse_logfile(seed) for seed in input[dset]))
        time_lines.append(f"{dset}: {means_n_vars(time)} sec\n")
        gen_lines.extend([f"{basename}_{dset}_{i}={g}\n"  for i,g in enumerate(gens)])

    with open(output, "w") as file:
        file.writelines(time_lines)
        file.write("\n")
        file.writelines(gen_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LaTeX metric table builder")
    parser.add_argument("-o", "--output", type=str, default="table.tex", help="output file with table")
    parser.add_argument("-b", "--basename", type=str, default="TENAS", help="basename for genotypes")
    parser.add_argument("input", type=str, help="path to experiment folder")
    args = parser.parse_args()

    output = args.output
    basename = args.basename
    input = {dset: glob(os.path.join(args.input,dset,"**/*.log"),recursive=True) for dset in DATASETS}
    main(input, output, basename)
