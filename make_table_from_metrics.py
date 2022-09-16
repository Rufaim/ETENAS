import numpy as np
import re
import argparse
from glob import glob
from itertools import chain
from collections import defaultdict
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kendalltau

class MetricsParser(object):
    REGEX_PARSER = re.compile(r"([\w\b]+)_(cifar100|cifar10|ImageNet16-120)")
    KNOWN_EXPERIMENTS = {
        "cond_ntk_v1": ("Condition number of NTK (draft)\cite{chen2020tenas}", 11),
        "regs_num": ("Expected number of ReLU regions\cite{chen2020tenas}", 12),
        "acc_nngp_v1": ("Accuracy of NNGP (draft)\cite{park2020towards}", 13),

        "acc_ntk": ("Accuracy of NTK\cite{jacot2018neural}", 21),
        "mse_ntk": ("MSEA of NTK", 22),
        "lga_ntk": ("Label-Gradient Alignment of NTK\cite{mok2022demystifying}", 23),
        "fro_ntk": ("Frobenius norm of NTK\cite{xu2021knas}", 24),
        "mean_ntk": ("Mean value of NTK\cite{xu2021knas}", 25),
        "cond_ntk": ("Condition number of NTK\cite{chen2020tenas}", 26),
        "eig_ntk": ("Eigenvalue score of NTK", 27),

        "acc_ntk_corr": ("Accuracy of NTK(correlation)", 32),
        "mse_ntk_corr": ("MSEA of NTK(correlation)", 32),
        "lga_ntk_corr": ("Label-Gradient Alignment of NTK(correlation)", 33),
        "fro_ntk_corr": ("Frobenius norm of NTK(correlation)", 34),
        "mean_ntk_corr": ("Mean value of NTK(correlation)", 35),
        "cond_ntk_corr": ("Condition number of NTK(correlation)", 36),
        "eig_ntk_corr": ("Eigenvalue score of NTK(correlation)\cite{mellor2021neural}", 37),

        "acc_nngp": ("Accuracy of NNGP\cite{park2020towards}", 41),
        "mse_nngp": ("MSEA of NNGP", 42),
        "lga_nngp": ("Label-Gradient Alignment of NNGP", 43),
        "fro_nngp": ("Frobenius norm of NNGP", 44),
        "mean_nngp": ("Mean value of NNGP", 45),
        "cond_nngp": ("Condition number of NNGP", 46),
        "eig_nngp": ("Eigenvalue score of NNGP", 47),

        "acc_nngp_corr": ("Accuracy of NNGP(correlation)", 51),
        "mse_nngp_corr": ("MSEA of NNGP(correlation)", 52),
        "lga_nngp_corr": ("Label-Gradient Alignment of NNGP(correlation)", 53),
        "fro_nngp_corr": ("Frobenius norm of NNGP(correlation)", 54),
        "mean_nngp_corr": ("Mean value of NNGP(correlation)", 55),
        "cond_nngp_corr": ("Condition number of NNGP(correlation)", 56),
        "eig_nngp_corr": ("Eigenvalue score of NNGP(correlation)", 57),

        "acc_nngp_read": ("Accuracy of NNGP(readout)", 61),
        "mse_nngp_read": ("MSEA of NNGP(readout)", 62),
        "lga_nngp_read": ("Label-Gradient Alignment of NNGP(readout)",63),
        "fro_nngp_read": ("Frobenius norm of NNGP(readout)", 64),
        "mean_nngp_read": ("Mean value of NNGP(readout)", 65),
        "cond_nngp_read": ("Condition number of NNGP(readout)", 66),
        "eig_nngp_read": ("Eigenvalue score of NNGP(readout)", 67),

        "acc_nngp_read_corr": ("Accuracy of NNGP(readout, correlation)", 71),
        "mse_nngp_read_corr": ("MSEA of NNGP(readout, correlation)", 72),
        "lga_nngp_read_corr": ("Label-Gradient Alignment of NNGP(readout, correlation)", 73),
        "fro_nngp_read_corr": ("Frobenius norm of NNGP(readout, correlation)", 74),
        "mean_nngp_read_corr": ("Mean value of NNGP(readout, correlation)", 75),
        "cond_nngp_read_corr": ("Condition number of NNGP(readout, correlation)", 76),
        "eig_nngp_read_corr": ("Eigenvalue score of NNGP(readout, correlation)", 77),

        "acc_nngp_train": ("Accuracy of NNGP(10 train batches)", 81),
        "mse_nngp_train": ("MSEA of NNGP(10 train batches)", 82),
        "lga_nngp_train": ("Label-Gradient Alignment of NNGP(10 train batches)", 83),
        "fro_nngp_train": ("Frobenius norm of NNGP(10 train batches)", 84),
        "mean_nngp_train": ("Mean value of NNGP(10 train batches)", 85),
        "cond_nngp_train": ("Condition number of NNGP (10 train batches)", 86),
        "eig_nngp_train": ("Eigenvalue score of NNGP (10 train batches)", 87),

        "acc_nngp_bb": ("Accuracy of NNGP(more batches)", 91),
        "mse_nngp_bb": ("MSEA of NNGP(more batches)", 92),
        "lga_nngp_bb": ("Label-Gradient Alignment of NNGP(more batches)", 93),
        "fro_nngp_bb": ("Frobenius norm of NNGP(more batches)", 94),
        "mean_nngp_bb": ("Mean value of NNGP(more batches)", 95),
        "cond_nngp_bb": ("Condition number of NNGP(more batches)", 96),
        "eig_nngp_bb": ("Eigenvalue score of NNGP(more batches)", 97),

        "acc_nngp_train_it": ("Accuracy of NNGP(20 train iterations)", 101),
        "mse_nngp_train_it": ("MSEA of NNGP(20 train iterations)", 102),
        "lga_nngp_train_it": ("Label-Gradient Alignment of NNGP(20 train iterations)", 103),
        "fro_nngp_train_it": ("Frobenius norm of NNGP(20 train iterations)", 104),
        "mean_nngp_train_it": ("Mean value of NNGP(20 train iterations)", 105),
        "cond_nngp_train_it": ("Condition number of NNGP(20 train iterations)", 106),
        "eig_nngp_train_it": ("Eigenvalue score of NNGP(20 train iterations)", 107),

        "regs_dist_full" : ("ReLU regions distance(1 batch)\cite{mellor2021neural}", 111),
        "regs_dist_max": ("ReLU regions distance(3 batches, max over batch)", 112),
        "regs_dist_mean": ("ReLU regions distance(3 batches, mean over batch)", 113),
    }

    def __call__(self, path):
        match = MetricsParser.REGEX_PARSER.search(path)
        if match is None:
            return None, None

        experiment, dataset = match.groups()
        experiment, position = MetricsParser.KNOWN_EXPERIMENTS.get(experiment, (experiment, 10000))
        return experiment, dataset, position


def correlation_pearson(accuracy, metric):
    return np.corrcoef(accuracy, metric)[0, 1]

def correlation_kt(accuracy, metric):
    return kendalltau(accuracy, metric)[0]

def r2_score(accuracy, metric):
    accuracy = MinMaxScaler().fit_transform(accuracy[:,None])[:,0]
    metric = MinMaxScaler().fit_transform(metric[:,None])
    feats = np.concatenate([metric, np.log(metric+1), np.sqrt(metric), metric**2],axis=-1)
    regressor = HuberRegressor().fit(feats, accuracy)
    reg_score = regressor.score(feats[~regressor.outliers_], accuracy[~regressor.outliers_])
    return reg_score


def main(input: str, output: str):
    parser = MetricsParser()

    experiments = defaultdict(dict)
    positions = {}
    for i in input:
        experiment, dataset, position = parser(i)
        experiments[experiment][dataset] = i
        positions[experiment] = position

    for experiment in experiments:
        for dataset in experiments[experiment]:
            path = experiments[experiment][dataset]
            data = np.load(path, allow_pickle=True)
            metric = data["metric"]
            accuracy = data["accuracy"]

            corr_p = correlation_pearson(accuracy, metric)
            corr_kt = correlation_kt(accuracy, metric)
            r2 = r2_score(accuracy, metric)

            experiments[experiment][dataset] = {
                "time": float(data["time_per_iteration"]),
                "correlation_pearson": corr_p,
                "correlation_kt": corr_kt,
                "r2": r2,
            }

    table_head =r"""	\begin{tabular}{l|c|c|c|c|c|c|c|c|c}
        \hline
        \multirow{2}{*}{Methods} & \multicolumn{3}{c|}{CIFAR-10} & \multicolumn{3}{c|}{CIFAR-100} & \multicolumn{3}{c}{ImageNet16-120} \\ \cline{2-10}
        & Kend-$\tau$ & $R^2_{\text{adj}}$ & Time (sec) & Kend-$\tau$ & $R^2_{\text{adj}}$ & Time (sec) & Kend-$\tau$ & $R^2_{\text{adj}}$ & Time (sec) \\ \hline
    """
    table_bottom =r"\end{tabular}"

    table_lines = []
    for experiment in experiments:
        to_write = [f"{experiment} & "]
        for dataset in ["cifar10", "cifar100", "ImageNet16-120"]:
            description = experiments[experiment].get(dataset, None)
            if description is None:
                to_write.append("& & ")
                continue
            to_write.append("{correlation_kt:.03f} & {r2:.03f} & {time:.02f} & ".format(**description))
        to_write[-1] = to_write[-1][:-2]
        to_write.append(r"\\ \hline")
        to_write.append("\n")
        table_lines.append(("".join(to_write), positions.get(experiment, 10000)))

    table_lines = [x[0] for x in sorted(table_lines, key=lambda x: x[1])]

    with open(output, "w") as file:
        file.write(table_head)
        file.writelines(table_lines)
        file.write(table_bottom)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("LaTeX metric table builder")
    parser.add_argument("-o", "--output", type=str, default="table.tex", help="output file with table")
    parser.add_argument("input", nargs="+", type=str, help="input mask to npz files")
    args = parser.parse_args()

    output = args.output
    input = chain(*[glob(inp) for inp in args.input])

    main(input, output)
