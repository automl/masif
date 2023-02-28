from omegaconf import DictConfig

from masif.scheduler.sh_masif import SH_masif
from masif.trainer import BaseTrainer


class SHScheduler(BaseTrainer):
    def __init__(self, model, optimizer, sh_config: DictConfig, **kwargs):
        super().__init__(model, optimizer)

        self._model = model

        # Consider: alternatively pass a partially instantiated model in here
        self.scheduler = SH_masif(model, **sh_config)

    def test(self, test_loader, test_loss_fns, **kwargs):
        """define one epoch of testing"""

        # change the reference to the model to be the scheduler (which has the model
        # as an attribute), to make the scheduler ready for slice evaluation.
        self.model = self.scheduler

        return BaseTrainer.test(self, test_loader, test_loss_fns, **kwargs)
