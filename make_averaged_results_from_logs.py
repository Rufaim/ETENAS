import re
import os
import numpy as np
import argparse
from glob import glob

DATASETS = ["cifar10", "cifar100", "ImageNet16"]
TIME_REGEX = re.compile(r"Time spent: (\d+.\d+) sec")
ACC_REGEX = re.compile(r"Test Accuracy (\d+.\d+) on")

def parse_logfile(path):
    with open(path) as file:
        text = file.read()
    time_match = TIME_REGEX.search(text)
    assert time_match is not None
    time_match = float(time_match.groups(1)[0])
    acc_match = ACC_REGEX.search(text)
    if acc_match is None:
        acc_match = 0.0
    else:
        acc_match = float(acc_match.groups(1)[0])
    return time_match, acc_match

def means_n_vars(d):
    mean = np.mean(d)
    std = np.std(d)
    return f"{mean:.02f}({std:.02f})"

def main(input, output):
    table_head = r"""\begin{tabular}{l|c|c|c|c|c|c}
        \hline
        \multirow{2}{*}{Methods} & \multicolumn{2}{c|}{CIFAR-10} & \multicolumn{2}{c|}{CIFAR-100} & \multicolumn{2}{c}{ImageNet16-120} \\ \cline{2-7}
        & Accuracy & Time (sec) & Accuracy & Time (sec) & Accuracy & Time (sec) \\ \hline
        """
    table_bottom = r"\end{tabular}"
    table_lines = [table_head]
    filler = " & "
    for experiment_name, iterations in input:
        table_line = [experiment_name, filler]
        for dset in DATASETS:
            time, acc = zip(*(parse_logfile(seed) for seed in iterations[dset]))
            table_line.append(means_n_vars(acc))
            table_line.append(filler)
            table_line.append(means_n_vars(time))
            table_line.append(filler)
        table_line = table_line[:-1]
        table_line.append(r" \\ \hline")
        table_line.append("\n")
        table_lines.append("".join(table_line))

    with open(output, "w") as file:
        file.write(table_head)
        file.writelines(table_lines)
        file.write(table_bottom)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("LaTeX metric table builder")
    parser.add_argument("-o", "--output", type=str, default="table.tex", help="output file with table")
    parser.add_argument("input", nargs="+", type=str, help="path to experiment folder(s)")
    args = parser.parse_args()

    output = args.output
    input = [(os.path.basename(os.path.normpath(inp)),
                {dset: glob(os.path.join(inp,dset,"**/*.log"),recursive=True) for dset in DATASETS} )
             for inp in args.input]

    main(input, output)
