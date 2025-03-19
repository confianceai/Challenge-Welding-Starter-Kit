"""
This module contains example codes to create dataloader from challenge datasets
"""

# Import dependencies modules
import urllib.request
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from challenge_welding.user_interface import ChallengeUI

from collections.abc import Callable


class ChallengeWeldingDataset(Dataset):
    """
    This class defines a pytorch dataset that is connected to the challenge repository
    """

    def __init__(
        self, user_ui: ChallengeUI, meta_df, resize=None, transform: Callable = None
    ):
        """
        This class defines a pytorch dataset that is connected to the challenge repository

        Arguments:
            user_ui: challenge_welding.user_interface
                Object to access challenge dataset for the dataset
            meta_df: pd.Dataframe
                Pandas dataframe file containing all your dataset metadata.
            resize : tuplet
                Tuplet containing the desired resizing : (width, height)
            transform : Callable
                Transform fonction call on sample to apply dedicated transformation
        notes :
            at lead one argument between resize dans transform shall be provide
            to make homogenous the image size in dataset for dataloader
        """

        self.meta_df = meta_df
        self.user_ui = user_ui
        self.resize = resize
        self.transform = transform

    def __len__(self):
        """Return the number of sample in the dataset"""
        return len(self.meta_df)

    def __getitem__(self, idx):
        """
        Parameters :
            idx: int
                Index of the sample you want to get in the dataset

        Return : dict
            A dict containing two key:

            "image" : np.array
                Image numpy array
            "meta" : Dict
                Dict containing all metadata associated with the image
        """

        image_array = self.user_ui.open_image(self.meta_df.iloc[idx]["path"])
        sample = {"image": image_array, "meta": self.meta_df.iloc[idx].to_dict()}

        # Apply resizing if it was given as parameters
        if self.resize:
            img = Image.fromarray(sample["image"], "RGB")
            sample["meta"]["resolution"] = list(
                self.resize
            )  # update resolution field of image metadata
            sample["image"] = np.array(img.resize(self.resize))

        # Applying conditional transform
        if self.transform:
            sample = self.transform(sample)

        return sample


def create_pytorch_dataloader(
        dataset: ChallengeWeldingDataset,
        batch_size,
        shuffle=False,
        num_workers=0,
):
    """This method create a pytorch dataloader from the input dataframe

    Args:

    dataset : ChallengeWeldingDataset
       The dataset for which you want to create a dataloader

    batch_size : int
        Size of the batch

    num_worker : int
        Number of workers to be used
    shuffle: bool

    Return :
        A pytorch dataloader browsing dataset covered by your input meta dataframe
    """

    print("Creating dataloader . .")

    # Create a pytorch dataloader from your dataset
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    print("Dataloader created")
    return dataloader
