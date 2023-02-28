from typing import Tuple

import torch

from masif.models.hierarchical_transformer import HierarchicalTransformer
from masif.trainer.base_trainer import BaseTrainer


class Trainer_Hierarchical_Transformer(BaseTrainer):
    def __init__(self, model: HierarchicalTransformer, optimizer):
        super().__init__(model, optimizer)

    def train(self, train_dataloader):
        for X, y in train_dataloader:

            # fixme: Move dict parsing to model!
            X_lc = X["X_lc"].float()
            X_meta_features = X["X_meta_features"].float()
            tgt_algo_features = X["tgt_algo_features"].float()
            tgt_meta_features = X["tgt_meta_features"].float()
            query_algo_features = X["query_algo_features"].float()  # [batch_size, n_query_algo], n_algo_feat
            query_algo_lc = X["query_algo_lc"].float()
            tgt_algo_lc = y["tgt_algo_lc"].float()  # [batch_size, L, n_query_algo, 1]
            # ------------------------------------------------------------------

            final_fidelity = tgt_algo_lc[:, -1]  # FIXME: make this part of the y dict!

            decoder_input_shape = query_algo_lc.shape
            batch_size, lc_length, n_query_algos_all, _ = decoder_input_shape

            # TODO deprec:
            # batch_size = decoder_input_shape[0]
            # lc_length = decoder_input_shape[1]
            # n_query_algos_all = decoder_input_shape[2]

            # flatten the first two dimensions todo what?
            # todo : doc what is a query algo
            query_algo_lc = torch.transpose(query_algo_lc, 1, 2).reshape(batch_size * n_query_algos_all, lc_length, 1)
            query_algo_features = query_algo_features.reshape(batch_size * n_query_algos_all, -1)

            # for each query set, select exactly n_query_algos[i] learning curves
            n_query_algos = torch.randint(0, n_query_algos_all + 1, (batch_size,))
            query_idx = torch.cat([torch.randperm(n_query_algos_all) for _ in range(batch_size)])
            query_idx = query_idx < n_query_algos.repeat_interleave(n_query_algos_all)

            query_algo_features = query_algo_features[query_idx]  # [sum(n_query_algos), n_query_algo]
            query_algo_lc = query_algo_lc[query_idx]  # [sum(n_query_algos), L, 1]

            # randomly mask out the tail of each learning curves. each learning curve needs to have at least 1 time step
            # and could be completely evaluated
            # TODO consider more sophisticated approach: dynamically reducing the mask sizes...
            n_query_lc = len(query_algo_lc)

            query_algo_lc, query_algo_padding_mask = self.mask_learning_curves(
                query_algo_lc, lc_length=lc_length, lower=1, upper=lc_length + 1, n_lc=n_query_lc
            )

            n_query_algos_all_list = n_query_algos.tolist()  # FIXME: this is not used
            # query_algo_features = torch.split(query_algo_features, n_query_algos_all_list)
            # query_algo_lc = torch.split(query_algo_lc, n_query_algos_all_list)
            # query_algo_padding_mask = torch.split(query_algo_padding_mask, n_query_algos_all_list)

            # same as above, mask the learning curve of the target algorithm. However, we allow zero evaluations while
            # the full fidelity value should not be presented here
            tgt_algo_lc, tgt_algo_padding_mask = self.mask_learning_curves(
                tgt_algo_lc, lc_length=lc_length, lower=0, upper=lc_length, n_lc=batch_size
            )

            self.optimizer.zero_grad()
            # Dataset meta features and final fidelity values


            features = {
                "tgt_algo_features": tgt_algo_features,
                "tgt_meta_features": tgt_meta_features,
                "query_algo_features": query_algo_features,
                "n_query_algo": n_query_algos,
                "query_algo_lc": query_algo_lc,
                "query_algo_padding_mask": query_algo_padding_mask,
                "tgt_algo_lc": tgt_algo_lc,
                "tgt_algo_padding_mask": tgt_algo_padding_mask,
            }
            self.to_device(features)

            predict = self.model(X_lc.to(self.device), X_meta_features.to(self.device), **features)

            lstm_loss = self.loss_fn(input=predict, target=final_fidelity.to(self.device))
            lstm_loss.backward()

            self.optimizer.step()

    def mask_learning_curves(
        self,
        lc: torch.Tensor,
        n_lc: int,
        lc_length: int,
        lower: int,
        upper: int,
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """
        mask the learning curves with 0. The masked learning curve has length between [lower, uppper)
        Args:
            lc: torch.Tensor,
                learning curves
            n_lc: int
                number of learning curves # FIXME: cant we infer this?
            lc_length: int,
                length of the learning curves # FIXME: cant we infer this?
            lower: int,
                minimal length of the learning curves after the masking
            upper: int
                maximal length of the learning curves after masking
        Returns:
            masked_lc: torch.Tensor
                masked learning curves. Masked values are 0
            padding_mask: torch.BoolTensor
                a tensor indicating which learning curves are masked # FIXME: what for?

        """
        n_reserved_lc = torch.randint(lower, upper, (n_lc, 1))
        padding_mask = torch.arange(0, lc_length) >= n_reserved_lc

        masked_lc = ~padding_mask.unsqueeze(-1) * lc

        return masked_lc, padding_mask
