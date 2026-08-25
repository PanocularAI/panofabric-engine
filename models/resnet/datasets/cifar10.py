import numpy as np
import torch

from dataclasses import dataclass
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node

from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.tools.logging import logger


class HuggingFaceImageDataset(IterableDataset, Stateful):
    """A simple HuggingFace image dataset for CIFAR-10.

    This dataset uses the HuggingFace datasets library to load the CIFAR-10 dataset.
    It is designed to work with the ParallelAwareDataloader for distributed training.

    """

    def __init__(self,
                 dataset_name: str,
                 dp_rank: int = 0, 
                 dp_world_size: int = 1, 
                 infinite: bool = False
    ):
        self.dataset_name = dataset_name
        ds = load_dataset(self.dataset_name, split="train")
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self.infinite = infinite

        # Variables for checkpointing
        self._sample_idx = 0

    def __iter__(self):
        while True:
            for item in self._data:
                image = item["img"]
                label = item["label"]

                # Convert PIL Image to numpy array → tensor
                np_img = np.array(image)
                tensor_img = torch.from_numpy(np_img).permute(2, 0, 1)  # (H, W, C) → (C, H, W)

                # Optionally normalize
                tensor_img = tensor_img.float() / 255.0

                yield {"input": tensor_img}, torch.tensor(label)

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)


class CifarDataLoader(ParallelAwareDataloader):
    """Configurable image dataloader for CIFAR-10 using HuggingFace datasets."""

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset: str = "uoft-cs/cifar10"
        infinite: bool = True

    def __init__(
        self,
        config: "CifarDataLoader.Config",
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer=None,
        seq_len: int = 0,
        local_batch_size: int,
        **kwargs,
    ):
        dataset = HuggingFaceImageDataset(
            dataset_name=config.dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
        )
        super().__init__(
            dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            batch_size=local_batch_size,
        )


