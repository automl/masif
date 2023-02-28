from typing import Iterable, List

import random


def train_test_split(n, share):
    train_split = random.sample(list(range(n)), k=int(n * share))
    test_split = list(set(range(n)) - set(train_split))

    return train_split, test_split


def train_valid_test_split(n, valid_share=0.1, test_share=0.1):
    #
    train_split = random.sample(list(range(n)), k=int(n * (1 - valid_share - test_share)))
    valid_split = random.sample(list(set(range(n)) - set(train_split)), k=int(n * valid_share))
    test_split = list(set(range(n)) - set(train_split) - set(valid_split))

    return train_split, valid_split, test_split


def train_valid_fix_test_split(n, valid_share=0.1, test_share=0.1, fold_idx=0):
    # here test_share can be either a float or an iterable object. In the former case, we simply take the last n samples
    # as the test sets. While for the later case, we take all the values inside test_share as test sets.
    if isinstance(test_share, Iterable):
        test_split = list(test_share)
        n_train_val = n - len(test_split)
    else:
        if fold_idx * test_share > 1:
            raise ValueError('Number of fold is too large!')

        n_test_split = int(test_share * n)
        n_test_split = n_test_split or 1
        n_train_val = n - n_test_split

        test_split = list(range(n_train_val  - fold_idx * n_test_split, n - fold_idx * n_test_split))

    train_split = random.sample(list(set(range(n)) - set(test_split)), k=int(n_train_val * (1 - valid_share)))
    valid_split = random.sample(
        list(set(range(n)) - set(train_split) - set(test_split)), k=int(n_train_val * valid_share)
    )

    return train_split, valid_split, test_split


def leave_one_out(n: int, idx: List[int]):
    """
    Leave one out cross validation
    This method allows to be explicit about which dataset is left out, when we want to run
    the experiments from command line. The idea is to run the experiment with a single holdout
    in one run with a fixed seed. Do n runs each with a different holdout and seed!
    """

    train_split = list(set(range(n)) - set(idx))
    valid_split = random.sample(train_split, k=1)
    test_split = idx

    return train_split, valid_split, test_split
