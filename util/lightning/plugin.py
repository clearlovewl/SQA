from pytorch_lightning.plugins import SingleDevicePlugin
from typing import Any, Optional, Union
import torch
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.training_type.training_type_plugin import TrainingTypePlugin
from pytorch_lightning.utilities import _XLA_AVAILABLE


class MyPlugin(TrainingTypePlugin):
    """Plugin that handles communication on a single device."""

    def __init__(
        self,
        device: torch.device,
        checkpoint_io: Optional[CheckpointIO] = None,
    ):
        super().__init__(checkpoint_io)
        self.device: torch.device = device
        self.global_rank = 0
        self.local_rank = 0
        self.world_size = 1

    @property
    def on_tpu(self) -> bool:
        return self.root_device.type == "xla" and _XLA_AVAILABLE

    @property
    def on_gpu(self) -> bool:
        return self.root_device.type == "cuda" and torch.cuda.is_available()

    def reduce(self, tensor: Union[Any, torch.Tensor], *args: Any, **kwargs: Any) -> Union[Any, torch.Tensor]:
        """Reduces a tensor from several distributed processes to one aggregated tensor. As this plugin only
        operates with a single device, the reduction is simply the identity.

        Args:
            tensor: the tensor to sync and reduce
            *args: ignored
            **kwargs: ignored

        Return:
            the unmodified input as reduction is not needed for single process operation
        """
        return tensor

    def all_gather(self, tensor: torch.Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> torch.Tensor:
        """Perform a all_gather on all processes."""
        return tensor

    @property
    def root_device(self) -> torch.device:
        return self.device

    def model_to_device(self) -> None:
        self._model.to(self.root_device)

    def setup(self) -> None:
        self.model_to_device()

    @property
    def is_global_zero(self) -> bool:
        return True

    def barrier(self, *args, **kwargs) -> None:
        pass

    def broadcast(self, obj: object, src: int = 0) -> object:
        return obj

    def teardown(self) -> None:
        if self.on_gpu:
            # GPU teardown
            self.lightning_module.cpu()
            # clean up memory
            torch.cuda.empty_cache()

    @property
    def lightning_restore_optimizer_and_schedulers(self) -> bool:
        """Override to disable Lightning restoring optimizers/schedulers.

        This is useful for plugins which manage restoring optimizers/schedulers.
        """
        return False