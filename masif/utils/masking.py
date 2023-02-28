import torch


def mask_lcs_randomly(lc_tensor, dataset=None):
    """# FIXME: move me to utils.masking.py
    Given a dataset's learning curves determine a random index for each
    curve, and set all values after that index to 0.
    :param lc_tensor: [n_fidelities, n_algos]
    :param dataset: if necessary, this allows access to the dataset object.
    :return: [n_algos, n_fidelities]
    """
    n_algos, n_fidelities = lc_tensor.shape
    mask = torch.zeros_like(lc_tensor)
    mask_idx = torch.randint(0, n_fidelities, (n_algos,)).view(-1, 1)
    for i, idx in enumerate(mask_idx):
        mask[i, 0:idx] = 1
    return lc_tensor * mask, mask.bool()


def mask_lcs_to_max_fidelity(lc_tensor, max_fidelity, *args, **kwargs):
    """
    In the case of SH & the Masked Fidelity evaluation protocol, we

    :param lc_tensor: [n_fidelities, n_algos]
    :param max_fidelity: int
    :param dataset: if necessary, this allows access to the dataset object.
    :return: [n_algos, max_fidelity], the mask of the tensor
    """
    # n_algos, n_fidelities = learning_curves.shape
    mask = torch.zeros_like(lc_tensor)
    mask[:, 0:max_fidelity] = 1
    return lc_tensor * mask, mask.bool()
