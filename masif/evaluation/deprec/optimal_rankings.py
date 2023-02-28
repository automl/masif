"""
As a proof of concept for m-shot, we need to show that we can find configurations
in a zero_shot manner we could reach. To do so, we train our manifold & the algo
instantiations and densely sample the latent space and compute the ranking for
each of these points (using the model's rank prediction module)
"""

import torch
from hydra.utils import call
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from masif.data import DatasetMetaFeatures


class ZeroShotOptimalDistance:
    def __init__(
        self,
        model,
        ranking_loss,
        scaler=MinMaxScaler(),
        batch: int = 20,
    ):
        self.model = model
        self.ranking_loss = ranking_loss
        self.batch = batch  # TODO unfix here!
        self.scaler = call(scaler)

    def forward(self, dataset_meta_features: DatasetMetaFeatures, final_performances, steps):
        """
        Defining the evaluation protocol. Since ZeroShotOptimalDistance
        is a static method, forward does not depend on some running meth
        """
        self.dataset_meta_features = dataset_meta_features
        self.final_performances = final_performances

        self.n_algos = len(self.final_performances[0])
        self.n_datas = len(self.dataset_meta_features)

        model_dist = self.encode_loader()
        cuboid_scores, model_scores = self.get_scores(steps, model_dist)
        return self._compare_rankings(
            cuboid_scores=cuboid_scores, model_scores=cuboid_scores, A0=self.final_performances
        )

    def encode_loader(self):
        # preallocate & gather all the embeddings for the dataset at hand
        self.model.eval()
        with torch.no_grad():
            model_embedding = self.model.encode(self.dataset_meta_features)
            self.cuboid_dims = self._get_bounds(model_embedding)
            dist_mat = torch.cdist(model_embedding, self.model.Z_algo)

        return dist_mat

    def _get_bounds(self, model_embedding):
        """
        To calculate the distances exhaustively, we need to find the cuboid
        bounds we would like to sample in
        """
        # Consider: could we use the algo vectors as bounds?

        # find the bounds of the cuboid
        # (columnwise min & max)
        min_bounds = torch.min(model_embedding, dim=0).values.tolist()
        max_bounds = torch.max(model_embedding, dim=0).values.tolist()

        # splice them such that we have the zip of them
        cuboid_dims = [dims for dims in zip(min_bounds, max_bounds)]
        return cuboid_dims

    def get_scores(self, steps, model_dist):
        """
        :return score: must be normed  to [0,1]
        """
        self.model.eval()
        with torch.no_grad():
            # grid the alternatives
            support = [torch.linspace(l, h, steps) for l, h in self.cuboid_dims]
            latent_coords = torch.meshgrid(support)

            # TODO check carefully if stacking is done as expected
            cuboid = torch.stack(latent_coords, dim=-1).reshape((-1, 2))

            #  calc the rankings for each grid point (relative to the model's algorithm vector
            #  instantiations
            dist_mat = torch.cdist(cuboid, self.model.Z_algo)

            # invert the distmats - such that lowest distance is highest score
            dist_mat = dist_mat - torch.max(dist_mat)
            model_dist = model_dist - torch.max(model_dist)

            # actual ranking
            # _, rankings = torch.topk(dist_mat, largest=False, k=self.n_algos)

            # scores for ndcg@k loss
            cuboid_scores = self.scaler.fit_transform(dist_mat)
            model_scores = self.scaler.transform(model_dist)

        return cuboid_scores, model_scores

    def _compare_rankings(self, cuboid_scores, model_scores, A0):
        """
        Try to find positions in the latent space that produce an ordinal better ranking

        :param cuboid_rankings: tensor of all ranking vectors for each grid point
        :param scores: tensor for all score vectors for all dataset points (from loader)
        :param A0: tensor: algorithm performances.

        :return:torch.Tensor: count of how many better solutions are there for
        each dataset.
        """
        from hydra.utils import call

        _, true_rankings = torch.topk(A0, largest=True, k=self.n_algos)

        # now compare model rankings with true rankings so that we see how far
        # we have gotten with trainer
        # fixme: dependence on hydra call is bad if we do not pass a cfg value

        # call(self.ranking_loss, true_rankings, scores)  # overall loss

        # from sklearn.metrics import ndcg_score
        newshape = true_rankings.shape[0], -1, true_rankings.shape[1]
        newshape_predicted = true_rankings.shape[0], 1, true_rankings.shape[1]
        # get the model ndcg values
        model_ndcg = torch.zeros(len(true_rankings))
        for i, (truth, predicted) in enumerate(
            zip(
                true_rankings.reshape(newshape).detach().numpy(),
                model_scores.reshape(newshape_predicted),
            )
        ):
            model_ndcg[i] = call(self.ranking_loss, truth, predicted)

        # get the ranking ndcg against ground truth for each value
        cuboid_ndcg = torch.zeros((len(true_rankings), len(cuboid_scores)))
        newshape_cuboid = cuboid_scores.shape[0], -1, cuboid_scores.shape[1]
        # from joblib import Parallel
        # Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
        # Parallel  # TODO: optimize this loop, because it is very slow
        for t, truth in tqdm(enumerate(true_rankings.reshape(newshape).detach().numpy())):
            for g, grid_point in enumerate(cuboid_scores.reshape(newshape_cuboid)):
                cuboid_ndcg[t, g] = call(self.ranking_loss, truth, grid_point)

        # find if there is any better ndcg score for that dataset and how many
        counts = torch.zeros(len(model_ndcg))
        for i, datapoint_ndcg in enumerate(model_ndcg):
            counts[i] = sum(cuboid_ndcg[i] > datapoint_ndcg)

            # Consider: based on this we can actually select the targets for our m-shot

        return counts
