"""
This module contains the code used to create a pytorch dataset thatt is connected to the challenge storage repository
"""

# Import dependencies modules
from torch.utils.data import Dataset, DataLoader
import challenge_welding.user_interface
from PIL import Image
import numpy as np


class ChallengeWeldingDataset(Dataset):
    
    def __init__(self, meta_df,resize=None,cache_strategy="remote",cache_dir=".cache"):
        """
        This class defines a pytorch dataset that is connected to the challenge repository 

        Arguments:
            meta_df: pd.Dataframe
                Pandas dataframe file containing all your dataset metadata.
            resize : tuplet
                Tuplet containing the desired resizing : (width, height)
            cache_strategy : str
                String representing cache strategy : "local" or "remote"
            cache_dir : str
                Path to cache directory when cache strategy is "local"
        """
        
        self.meta_df=meta_df
        self.user_ui=challenge_welding.user_interface.ChallengeUI(cache_strategy,cache_dir)
        self.resize=resize
        
    def __len__(self):
        """ Return the number of sample in the dataset"""
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
                
        image_array=self.user_ui.open_image(self.meta_df.iloc[idx]["path"])
        sample = {'image': image_array, 'meta': self.meta_df.iloc[idx].to_dict()}
        
        # Apply resizing if it was given as parameters
        if self.resize:           
            img = Image.fromarray(sample["image"], 'RGB')
            sample["meta"]["resolution"]=list(self.resize) # update resolution field of image metadata
            sample["image"]=np.array(img.resize(self.resize))
        return sample








