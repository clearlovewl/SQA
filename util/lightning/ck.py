from pytorch_lightning.plugins import TorchCheckpointIO 
from typing import Any, Callable, Dict, Optional
import pytorch_lightning as pl
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.cloud_io import atomic_save, get_filesystem
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.types import _PATH

class CustomCheckpointIO(TorchCheckpointIO):
    def load_checkpoint(
        self, path: _PATH, map_location: Optional[Callable] = lambda storage, loc: storage
    ) -> Dict[str, Any]:
        """Loads checkpoint using :func:`torch.load`, with additional handling for ``fsspec`` remote loading of
        files.

        Args:
            path: Path to checkpoint
            map_location: a function, :class:`torch.device`, string or a dict specifying how to remap storage
            locations.

        Returns: The loaded checkpoint.

        Raises:
            FileNotFoundError: If ``path`` is not found by the ``fsspec`` filesystem
        """

        # Try to read the checkpoint at `path`. If not exist, do not restore checkpoint.
        fs = get_filesystem(path)
        if not fs.exists(path):
            raise FileNotFoundError(f"Checkpoint at {path} not found. Aborting training.")
        a=pl_load(path, map_location=map_location)
        remain=['epoch','global_step','pytorch-lightning_version','state_dict']
        b={}
        for i in remain:
            b[i]=a[i]
        return b