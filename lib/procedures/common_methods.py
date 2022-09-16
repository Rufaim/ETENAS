import torch
import numpy as np
from enum import Enum


def accuracy(targ, pred):
    correct_predictions = (torch.argmax(pred, dim=1) == torch.argmax(targ, dim=1)).to(torch.float32)
    return torch.mean(correct_predictions).item()

def accuracy_mse(targ, pred):
    pred = torch.softmax(pred, dim=-1)
    diff = (targ - pred)**2
    return torch.mean(diff).item()

def label_gradient_alignment(mat, labels):
    mat_normalized = mat - torch.mean(mat)
    labels_normalized = torch.matmul(labels, labels.T)
    labels_normalized[labels_normalized<1] = -1
    labels_normalized = labels_normalized - torch.mean(labels_normalized)

    score = mat_normalized * labels_normalized / (torch.norm(mat_normalized, 2) * torch.norm(labels_normalized, 2))
    return torch.sum(score).item()

def frobenius_norm(mat):
    return torch.norm(mat, p="fro").item()

def mean(mat):
    return torch.mean(mat).item()

def conditional_number(mat):
    eigenvalues = torch.linalg.eigvalsh(mat, UPLO='U')
    return np.nan_to_num((eigenvalues[-1].item() / eigenvalues[0]).item(), copy=True, nan=100000.0)

def eigenvalue_score(mat):
    eigenvalues = torch.linalg.eigvalsh(mat, UPLO='U')
    k = 1 # 1e-5
    return -torch.sum(torch.log(eigenvalues + k) + 1. / (eigenvalues + k)).item()


class MetricType(Enum):
    ACC = accuracy
    MSE = accuracy_mse
    FRO = frobenius_norm
    MEAN = mean
    COND = conditional_number
    EIG = eigenvalue_score
    LGA = label_gradient_alignment

    def require_only_matrix(self):
        if self is MetricType.FRO:
            return True
        if self is MetricType.MEAN:
            return True
        if self is MetricType.COND:
            return True
        if self is MetricType.EIG:
            return True
        return False

