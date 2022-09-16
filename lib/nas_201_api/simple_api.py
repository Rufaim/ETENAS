import pickle
import copy
from typing import Text


class SimpleApi(object):
    '''This is a simplified version of NASBench201API that is meant to load faster, but does not contain weigths of models.
    For search purposes that require only final accuracies this method works faster.
    '''
    def __init__(self, filename: Text, verbose: bool=True):
        self.verbose = verbose
        self.filename = filename

        if verbose:
            print('try to create the NAS-Bench-201 simplified api from {:}'.format(filename))

        with open(self.filename, "rb") as f:
            self.archstr2index = pickle.load(f)
            self.arch2metric = pickle.load(f)

    def query_index_by_arch(self, arch):
        if isinstance(arch, str):
            arch_index = arch
        elif hasattr(arch, "tostr"):
            arch_index = arch.tostr()
        else:
            return -1
        return self.archstr2index.get(arch_index, -1)

    def query_by_index(self, arch_index: int):
        """Plase note that unlike full API this one returns a dictionary of test metrics on datasets"""
        assert arch_index in self.arch2metric, 'arch_index [{:}] does not in arch2info'.format(arch_index)
        archInfo = copy.deepcopy(self.arch2metric[arch_index])
        return archInfo

    def __repr__(self):
        return ("(NAS-Bench-201(Simple API)({total} architectures, file={filename})".format(total=len(self.archstr2index), filename=self.filename))
