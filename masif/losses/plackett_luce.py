import torch
from torch.nn.modules.loss import _Loss as Loss


class PlackettLuceLoss(Loss):
    def __init__(self, reduction: str = "mean", k=999) -> None:
        super(PlackettLuceLoss, self).__init__(reduction=reduction)
        self.k = k

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""
        Apply Plackett Luce Model to the input and targets and compute their losses
        f is a vector of scores for the respective algorithms # this is

        .. math::
            prod_i^n, prod_j^m * \frac{exp(f(a_{ij})}{sum_{k=j}^m exp(f(a_{i,k})}

        NOTICE the :math:`sum_{k=j}` is dependent on :math:`prod_j`. this is a shift in the softmax normalization,
        that introduces the "without replacement property"

        :param y_hat: torch.Tensor, float. score vector for the algorithms
        :param y: torch.Tensor, float. ordered true score (/ranking) of the algorithms

         NOTE: Carefull, TopkMaxRegret is assuming that the maximum of y_scores is the best.
        """
        assert len(y_hat.shape) == 2
        assert y.shape == y_hat.shape

        y_orders = torch.argsort(y, dim=1)
        yhat_reordered = torch.gather(y_hat, -1, y_orders)

        # TODO remove this:
        # yhat_reordered = yhat_reordered.log_softmax(1)  # FIXME: this is incorrect: we
        # need the "without replacement" option of this. meaning, when we do the
        # softmax, we will have to do a shifted softmax on the reordered tensor.

        return -torch.sum(self.shifted_softmax(yhat_reordered))

    def shifted_softmax(self, ordered_ranking: torch.Tensor):
        """Shifted softmax, to account for the "without replacement property" of the Plackett Luce Model

        Relevant part of the equation is:

        ..:math:
                \frac{exp(f(a_{ij})}{sum_{k=j}^m exp(f(a_{i,k})}

        where we shift the normalization
        """
        shifted = torch.zeros(ordered_ranking.shape)
        for i in range(min(self.k, ordered_ranking.shape[1])):
            a = ordered_ranking[:, i:]

            # TODO analyze impact on optimization procedure
            # Safeguarding NaN and infs with value impuation in torch

            shifted[:, i] = torch.nan_to_num(torch.log(torch.exp(a[:, 0]) / torch.exp(a).sum(axis=1)))

        return shifted


if __name__ == "__main__":
    loss = PlackettLuceLoss(k=3)

    y_hat = torch.tensor(
        [
            [
                -0.9204,
                -0.5594,
                0.7821,
                2.4453,
                0.0966,
                1.1002,
                -0.5674,
                -0.6546,
                -0.4556,
                -0.6019,
                -0.9281,
                0.4240,
                1.2746,
                -0.9137,
                -0.8880,
                -0.6984,
                1.1963,
                -0.3932,
                -0.7912,
                1.0838,
                -0.3330,
                -0.4195,
                -0.4281,
                -0.9152,
                2.2998,
                0.1853,
                -0.6625,
                -0.5602,
                1.3004,
                -1.0301,
                1.3173,
                0.5263,
                -1.0629,
                -0.6789,
                2.2061,
                -0.8885,
                -0.8748,
                -0.6881,
                -0.7391,
                -0.6611,
                -0.7257,
                1.3788,
                -0.8584,
                -0.5710,
                1.7002,
                -1.0215,
                -0.9639,
                2.8822,
                -0.6389,
                -0.8447,
                2.6585,
                -0.5089,
                -0.7349,
                -0.6525,
                0.4345,
                -0.4275,
                1.1999,
                -0.8655,
            ],
            [
                -0.1554,
                -0.5594,
                1.5874,
                -0.7404,
                0.6773,
                -0.8101,
                -0.5674,
                -0.6546,
                0.6953,
                -0.6019,
                0.9033,
                -0.5308,
                -0.7956,
                -0.9137,
                0.0355,
                -0.6984,
                -0.7982,
                -0.1470,
                -0.7912,
                -0.7677,
                -0.3330,
                -0.2476,
                1.8957,
                0.1157,
                -0.8371,
                -0.5991,
                -0.4519,
                -0.5602,
                -0.6773,
                -0.0768,
                -0.4810,
                -1.1383,
                0.5224,
                -0.1323,
                -0.7222,
                -0.1491,
                0.3033,
                0.9129,
                1.4962,
                0.4487,
                0.2175,
                -0.7516,
                1.0876,
                0.0729,
                -0.8813,
                0.5468,
                1.7383,
                -0.0431,
                -0.1576,
                0.5788,
                -0.5805,
                -0.5089,
                -0.7349,
                -0.6525,
                -0.7726,
                -0.4275,
                -0.6766,
                1.4693,
            ],
        ]
    )
    y = torch.tensor(
        [
            [
                78.0764,
                44.8373,
                93.7765,
                57.2843,
                96.3225,
                55.3041,
                48.0905,
                55.3041,
                57.9915,
                44.6959,
                44.6959,
                55.3041,
                47.8076,
                76.3791,
                42.4328,
                56.0113,
                58.9816,
                55.3041,
                57.9915,
                58.4158,
                53.0410,
                49.3635,
                44.6959,
                44.6959,
                93.7765,
                97.8784,
                42.1499,
                93.3522,
                51.0608,
                54.8798,
                42.1499,
                64.0735,
                54.1726,
                58.8402,
                53.0410,
                44.2716,
                55.3041,
                79.7737,
                41.8670,
                57.5672,
                44.1301,
                55.4455,
                53.3239,
                52.8996,
                58.2744,
                58.2744,
                55.3041,
                48.7977,
                51.2023,
                44.6959,
                83.4512,
                44.6959,
                42.9986,
                50.6365,
                50.0707,
                43.4229,
                86.1386,
                49.2221,
            ],
            [
                80.5042,
                65.2596,
                80.2541,
                71.2114,
                80.3841,
                77.7833,
                11.7735,
                11.7735,
                74.1322,
                12.7738,
                14.0442,
                40.9023,
                74.0722,
                81.1543,
                77.5733,
                25.3076,
                70.4911,
                88.2265,
                74.3423,
                74.4423,
                65.3096,
                81.2744,
                11.7735,
                11.7735,
                79.5239,
                82.3347,
                77.5633,
                88.2265,
                64.0392,
                14.5744,
                63.8592,
                81.4344,
                65.5697,
                77.3032,
                64.4193,
                20.6662,
                12.9839,
                79.1938,
                53.4460,
                75.1726,
                65.5697,
                30.4391,
                81.3444,
                48.9047,
                74.3123,
                40.9423,
                52.2257,
                63.8392,
                85.1856,
                31.6795,
                79.6739,
                38.4915,
                87.7163,
                73.7721,
                67.9704,
                65.6597,
                80.8843,
                78.7136,
            ],
        ]
    )

    y_hat = torch.tensor(
        [
            [
                0.1,
                0.2,
                0.3,
                0.4,
            ],
            [
                0.1,
                0.2,
                0.3,
                0.4,
            ],
        ]
    )
    y = torch.tensor([[4, 2, 3, 1], [4, 3, 2, 1]])
    print(loss(y_hat, y))
