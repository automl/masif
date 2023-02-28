import torch

from masif.models.transformerMLP import masifTransformerMLP


class masifTransformerSubsequent(masifTransformerMLP):
    def __init__(self, transformer_algos, *args, **kwargs):
        super(masifTransformerSubsequent, self).__init__(*args, **kwargs)
        self.transformer_algos = transformer_algos

    def forward(self, learning_curves: torch.Tensor, mask: torch.Tensor, dataset_meta_features,
                **kwargs):
        n_datasets, n_algos, n_fidelities = learning_curves.shape
        dataset_metaf_encoding = self.dataset_metaf_encoder(dataset_meta_features)

        # create a uniform (latent) vector representation out of the (partially masked) sequence,
        # that summarizes the lc information obtained.
        lc_encoding = self.forward_lc(self, learning_curves, mask)

        # cross correlate the algorithm curves by changing the dimension across which is
        # attended. Notice that the encoding dimension is E= n_fidelities, S =n_algos.
        lc_encoding = self.transformer_alogs(
            lc_encoding.permute(0, 2, 1)
        )

        # flatten the lc_encoding into a single vector, concat with the dataset meta features
        # and shoot it through the decoder.
        return self.decoder(
            torch.cat([lc_encoding.view(1, -1), dataset_metaf_encoding], dim=1)
        )
