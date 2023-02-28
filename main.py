import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf

# A logger for this file
log = logging.getLogger(__name__)
import copy
import torch

OmegaConf.register_new_resolver("device_ident", lambda _: torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"))

OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))
OmegaConf.register_new_resolver("len", lambda l: len(l))
OmegaConf.register_new_resolver("range", lambda start, stop, step: list(range(start, stop, step)))

import os
import pathlib
import random
import string

import wandb
from hydra.utils import get_original_cwd

from masif.utils.util import print_cfg, seed_everything


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


@hydra.main(config_path="configs", config_name="base")
def pipe_train(cfg: DictConfig) -> None:
    model_opts = set(cfg.model.get('model_opts', []))

    model_type = copy.deepcopy(cfg.wandb.group)

    housekeeping(cfg)

    seed_everything(cfg.seed)

    # Create / find + setup the data -------------------------------------------
    # optionally download / resubset the dataset
    if cfg.dataset.dataset_raw.enable:
        # fixme: rather than enable: save a meta info string of the dataset_raw config,
        #  that generated the data and check if it is consistent; else trigger recompute!
        call(cfg.dataset.dataset_raw, _recursive_=False)

    # hack: dataset_meta_features must be instantiated to know n for a traintest split index
    # of the data, that can be passed on.

    if "dataset_meta" in cfg.dataset:
        dataset_meta_features = instantiate(cfg.dataset.dataset_meta)
    else:
        dataset_meta_features = instantiate(cfg.dataset.lc_meta)

    # train test split by dataset major
    train_split, valid_split, test_split = call(cfg.train_test_split, n=len(dataset_meta_features))

    cfg.dynamically_computed.n_datasets = dataset_meta_features.shape[0]
    del dataset_meta_features

    loaders = {}
    # Create the dataloaders (conditional on their existence)
    if "train_dataset_class" in cfg.dataset.keys():
        train_set = instantiate(cfg.dataset.train_dataset_class, split=train_split)
        train_loader = instantiate(cfg.dataset.train_dataloader_class, dataset=train_set)
        loaders["train_loader"] = train_loader

    if "valid_dataset_class" in cfg.dataset.keys():
        valid_set = instantiate(cfg.dataset.valid_dataset_class, split=valid_split)
        valid_loader = instantiate(cfg.dataset.valid_dataloader_class, dataset=valid_set)
        loaders["valid_loader"] = valid_loader

    if "test_dataset_class" in cfg.dataset.keys():
        test_set = instantiate(cfg.dataset.test_dataset_class, split=test_split)
        test_loader = instantiate(cfg.dataset.test_dataloader_class, dataset=test_set)
        loaders["test_loader"] = test_loader

    # Dynamically computed configurations.
    # maybe change later to resolvers? https://omegaconf.readthedocs.io/en/2.2_branch/usage.html#access-and-manipulation
    ref = list(loaders.values())[0].dataset

    if ref.meta_dataset is not None:
        cfg.dynamically_computed.n_data_meta_features = ref.meta_dataset.shape[1]
    else:
        cfg.dynamically_computed.n_data_meta_features = 0
    cfg.dynamically_computed.n_algos = ref.learning_curves.shape[1]
    cfg.dynamically_computed.n_fidelities = ref.learning_curves.shape[-1]

    if hasattr(train_set, 'meta_algo') and hasattr(train_set.meta_algo, 'transformed_df'):
        cfg.dynamically_computed.n_algo_meta_features = train_set.meta_algo.transformed_df.shape[-1]

    if model_type.startswith('masif_M_transformer'):
        cfg.model.decoder.hidden_dims[0] += ref.learning_curves.shape[1] * cfg.model.transformer_lc.encoder_layer.d_model

    # cfg.dynamically_computed.n_algo_meta_features = ref.lcs.transformed_df.shape[1]

    wandb.config.update(
        {
            "dynamically_computed.n_algos_meta_features": cfg.dynamically_computed.n_algo_meta_features,
            "dynamically_computed.n_data_meta_features": cfg.dynamically_computed.n_data_meta_features,
        }
    )

    model = instantiate(cfg.model)
    model.to(cfg.device)
    model.device = cfg.device
    torch.device(cfg.device)

    trainer = instantiate(cfg.trainer.trainerobj, model)

    prediction, ground_truth = trainer.run(**loaders, **cfg.trainer.run_call)
    log.info("Done!")

    dataset_name = cfg.dataset.name

    fold_idx = cfg.train_test_split.get('fold_idx', 0)
    seed = cfg.seed
    model = cfg.wandb.tags[-1]

    if not isinstance(prediction, torch.Tensor):
        return

    obj_dir = pathlib.Path.home() / 'Project' / 'masif_data' / dataset_name / f'fold_{fold_idx}' / model / f'seed_{seed}'
    if not obj_dir.exists():
        os.makedirs(str(obj_dir))

    #  assert prediction.shape == ground_truth.shape
    # assert len(prediction) == len(test_split)
    # assert train_set.learning_curves.shape[-1] == prediction.shape[-1]

    torch.save(prediction,  str(obj_dir / 'prediction.pt'))
    torch.save(ground_truth,  str(obj_dir / 'ground_truth.pt'))


def housekeeping(cfg: DictConfig) -> None:
    dict_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    print_cfg(cfg)

    dataset = cfg.dataset.name

    fold_idx = cfg.train_test_split.get('fold_idx', 0)

    # SET job type as DATASET name
    cfg.wandb.job_type = f'{dataset}_{fold_idx}'

    cfg.wandb.tags = [dataset, f'fold_idx_{fold_idx}', f'seed_{cfg.seed}', *cfg.wandb.tags]

    log.info(get_original_cwd())
    hydra_job = (
        os.path.basename(os.path.abspath(os.path.join(HydraConfig.get().run.dir, "..")))
        + "_"
        + os.path.basename(HydraConfig.get().run.dir)
    )
    cfg.wandb.id = hydra_job + "_" + id_generator() 
    wandb.init(**cfg.wandb, config=dict_cfg)
    hydra_cfg = HydraConfig.get()
    command = f"{hydra_cfg.job.name}.py " + " ".join(hydra_cfg.overrides.task)
    if not OmegaConf.is_missing(hydra_cfg.job, "id"):
        slurm_id = hydra_cfg.job.id
    else:
        slurm_id = None
    wandb.config.update({"command": command, "slurm_id": slurm_id})
    log.info("Hydra initialized a new config_raw")
    log.debug(str(cfg))


if __name__ == "__main__":
    pipe_train()
