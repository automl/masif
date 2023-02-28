import warnings
from functools import partial
from typing import List, Callable

import torch

from masif.trainer.base_trainer import BaseTrainer

warnings.warn("Slice_evaluator module is deprecated. ", DeprecationWarning)


class SliceEvaluator(BaseTrainer):
    def __init__(self, model, max_fidelities: List, masking_fn: Callable, *args, **kwargs):
        """
        This BaseTrainer subclass trains as usual, but evaluates the model on a set of slices of
        different length. In particular, it evaluates the model's capability to "forsee" the future
        conditioned on only partial learning curves
        This class is designed to ev aluate the modelduring test time on a fixed
        number of fidelities (i.e. there is a highest accessible fidelity (via masking).

        In particular it is build with successive halving in mind. Consider its definition.
        """
        super().__init__(model, *args, **kwargs)
        self.max_fidelities = max_fidelities
        self.masking_fn = masking_fn

    def evaluate(self, test_loader, valid_loss_fn, fn_name, *args, **kwargs):
        for fidelity in self.max_fidelities:
            test_loader.dataset.masking_fn = partial(self.masking_fn, max_fidelity=fidelity)

            # TODO: make one fwd pass and compute all validation losses on the fwd pass
            #  to drastically reduce the number of fwd passes!
            losses = torch.zeros(len(test_loader))
            for i, (X, y) in enumerate(test_loader):
                self.to_device(X)
                self.to_device(y)

                y_hat = self.model.forward(**X)
                losses[i] = valid_loss_fn(y_hat, y["final_fidelity"])

            wandb.log({f"max fidelity: {fn_name}": losses.mean(), "fidelity": fidelity})

    def run(self, train_loader, test_loader, epochs, log_freq, *args, **kwargs):
        # TODO: check what happens when this condition is not met
        if epochs != log_freq:
            warnings.warn("SliceEvaluator is intended to be run with log_freq == epochs")
        super().run(train_loader, test_loader, epochs=epochs, log_freq=log_freq, *args, **kwargs)


if __name__ == "__main__":
    """
    Experiment to run SH baseline on the fixed fidelity slices (i.e. maximum available fidelity)
    and compare that against a trainable model, that is evaluated on the same maximum slices.
    """

    from torch.utils.data import DataLoader

    from masif.models.baselines.successive_halving import SuccessiveHalving
    from masif.losses.spearman import SpearmanLoss
    from masif.utils.masking import mask_lcs_to_max_fidelity, mask_lcs_randomly
    from masif.data.lcbench.example_data import data_path, pipe_lc, pipe_meta
    from masif.evaluation.topkregret import TopkMaxRegret

    from masif.utils.util import seed_everything

    import wandb

    for seed in range(35):  # to vary across datasets & learning processes.
        seed_everything(seed)

        from masif.data import Dataset_Join_Dmajor, Dataset_LC, DatasetMetaFeatures
        from masif.utils.traintestsplit import leave_one_out

        # train_split, test_split = train_test_split(n=35, share=0.8)
        train_split, test_split = leave_one_out(n=35, idx=[seed])
        test_dataset = Dataset_Join_Dmajor(
            meta_dataset=DatasetMetaFeatures(path=data_path / "meta_features.csv", transforms=pipe_meta),
            lc=Dataset_LC(path=data_path / "logs_subset.h5", transforms=pipe_lc, metric="Train/train_accuracy"),
            split=test_split,
        )

        # Show, that we can also input another model and train it beforehand.
        train_dataset = Dataset_Join_Dmajor(
            meta_dataset=DatasetMetaFeatures(path=data_path / "meta_features.csv", transforms=pipe_meta),
            lc=Dataset_LC(path=data_path / "logs_subset.h5", transforms=pipe_lc, metric="Train/train_accuracy"),
            split=train_split,
            masking_fn=mask_lcs_randomly,
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # (evaluate SH) ---------------------------------------------
        budgets = list(range(1, 52))
        wandb.init(entity="example", mode="online", project="masif-tmlr", job_type="base: sh, leave one out", group="sh")
        model = SuccessiveHalving(budgets=budgets, eta=2)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # batch=1, because of SH

        sliceevaluator = SliceEvaluator(
            model,
            max_fidelities=budgets,
            # creates a learning curve for these fidelities
            masking_fn=mask_lcs_to_max_fidelity,
        )

        # sliceevaluator.evaluate(test_loader, valid_loss_fn=SpearmanLoss(), fn_name='spearman')
        epochs = 1
        sliceevaluator.run(
            train_loader,  # FIXME: make this optional! (since sh won't need it)
            test_loader,
            train_loss_fn=SpearmanLoss(),
            valid_loss_fns={
                "spearman": SpearmanLoss(),
                "top1_regret": TopkMaxRegret(1),
                "top3_regret": TopkMaxRegret(3),
            },
            epochs=epochs,  # <----
            log_freq=epochs,  # <----
        )
        wandb.finish()

        # (Train + evaluate another model) ----------------------------
        from masif.models.masif_wp import masif_WP
        from masif.utils.mlp import MLP

        wandb.init(
            entity="example",
            mode="online",
            project="masif-tmlr",
            job_type="base: masif_wp rnd. masking, leave one out",
            group="masif_wp",
        )

        n_algos = 58
        n_meta_features = 107
        model = masif_WP(
            encoder=MLP(hidden_dims=[n_meta_features, 300, 200]),
            decoder=MLP(hidden_dims=[200, n_algos]),
            input_dim=n_algos,
            n_layers=2,
        )
        # model = PlackettTest(encoder=MLP(hidden_dims=[n_meta_features, 100, n_algos])) # constant in
        # fidelity
        sliceevaluator = SliceEvaluator(
            model,
            max_fidelities=budgets,
            masking_fn=mask_lcs_to_max_fidelity,
            optimizer=partial(torch.optim.Adam, lr=1e-3),
        )

        # NOTICE: for wandb tracking to be sensible, we need to train the model fully first.
        epochs = 1000
        sliceevaluator.run(
            train_loader,
            test_loader,
            train_loss_fn=SpearmanLoss(),
            valid_loss_fns={
                "spearman": SpearmanLoss(),
                "top1_regret": TopkMaxRegret(1),
                "top3_regret": TopkMaxRegret(3),
            },
            epochs=epochs,  # <----
            log_freq=epochs,  # <----
        )
        wandb.finish()
    print("done")
