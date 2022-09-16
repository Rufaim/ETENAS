import numpy as np
import torch


def round_to(number, precision, eps=1e-8):
    # round to significant figure
    dtype = type(number)
    if number == 0:
        return number
    sign = np.sign(number)
    number = np.abs(number) + eps
    power = np.floor(np.log10(number)) + 1
    if dtype == int:
        return int(sign * round(number*10**(-power), precision) * 10**(power))
    else:
        return sign * round(number*10**(-power), precision) * 10**(power)

# def round_to(number, precision, eps=1e-8):
#     # round to significant figure
#     dtype = type(number)
#     if number == 0:
#         return number
#     sign = number / abs(number)
#     number = abs(number) + eps
#     power = math.floor(math.log(number, 10)) + 1
#     if dtype == int:
#         return int(sign * round(number*10**(-power), precision) * 10**(power))
#     else:
#         return sign * round(number*10**(-power), precision) * 10**(power)

def is_single_path(network):
    arch_parameters = network.get_alphas()
    edge_active = torch.cat([(torch.nn.functional.softmax(alpha, 1) > 0.01).float().sum(1) for alpha in arch_parameters], dim=0)
    for edge in edge_active:
        assert edge > 0
        if edge > 1:
            return False
    return True