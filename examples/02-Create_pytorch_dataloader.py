"""
This script show how to use this package to create a pytorch dataloader
It requires the installation of pytorch==2.6.0
"""

import sys
sys.path.insert(0, "..") # For local tests without pkg installation, to make challenge_welding module visible 

from challenge_welding.user_interface import ChallengeUI
import challenge_welding.dataloaders

# Initiate the user interface

my_challenge_UI=ChallengeUI(cache_strategy="local",cache_dir="loadercache")

# Get list of available datasets

ds_list=my_challenge_UI.list_datasets()
print(ds_list)

# In this example we will choose a small dataset

ds_name="example_mini_dataset"
ds_name="welding-detection-challenge-dataset"
# Load all metadata of your dataset

meta_df=my_challenge_UI.get_ds_metadata_dataframe(ds_name)

# Create your dataloader
dataloader=challenge_welding.dataloaders.create_pytorch_dataloader(input_df=meta_df[0:20],
                                                     cache_strategy=my_challenge_UI.cache_strategy,
                                                     cache_dir=my_challenge_UI.cache_dir,
                                                     batch_size=100,
                                                     shuffle=False)
# Test your dataloader       
for i_batch, sample_batched in enumerate(dataloader):
    print("batch number", i_batch)
    print("batch content image",    sample_batched['image'].shape)
    print("batch content meta",sample_batched['meta'])

    # observe 4th batch and stop.
    if i_batch == 3:
        break