"""
This module contains example codes to create dataloader from challenge datasets
"""

# Import dependencies modules
from torch.utils.data import Dataset, DataLoader
import challenge_welding.user_interface
from PIL import Image
import numpy as np
import os 
from tqdm import tqdm
import urllib.request
import pandas as pd

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


def create_pytorch_dataloader(input_df,batch_size,cache_strategy,cache_dir,shuffle=False, num_workers=0,):
           
    """This method create a pytorch dataloader from the input dataframe 
        
        Args:
            
        input_df :pd.DataFrame
            Dataframe containing the metadata of the dataset for which you want to create a dataloader
               
        cache_strategy : str
            String representing cache strategy : "local" or "remote"
        cache_dir : str
            Path to cache directory when cache strategy is "local"
        
        batch_size : int
            Size of the batch
            
        num_worker : int
            Number of workers to be used
        shuffle: bool

        Return :
            A pytorch dataloader browsing dataset covered by your input meta dataframe
    """
    
    # If local cache is activated         
    if  cache_strategy=="local": 
        print("Cache storage has been activated in ",cache_dir)
        
        unique_id=int(input_df.iloc[0]["sample_id"].split("_")[-1])+int(input_df.iloc[-1]["sample_id"].split("_")[-1]) # unique _id associated with input dataframe
        print("cache_metadata_unique_id",unique_id)
        local_meta_path=cache_dir+os.sep+"local_"+str(unique_id)+"_storage_meta.parquet"
        
        # If metadata have already been downloaded in cache directory (we check specific parquet existence file for that)
        if os.path.exists(local_meta_path):  
            
            print("Cache directory has already been built, loading local metadata..")
            input_df=pd.read_parquet(local_meta_path)
            
            print("local metadata loaded !")
            print(input_df["path"])
        else:
            print("Downloading all raw samples in cache storage, please wait . .")
            local_meta_df=input_df.copy()
            
            #Create local cache folder for raw data
            local_meta_df.set_index("sample_id",inplace=True)
            for sp in tqdm(local_meta_df.index): # For each sample in dataset

                # Define target local path and create directory if necessary 
                target_image_name=local_meta_df.loc[sp,"path"]
                output_path=cache_dir+os.sep+target_image_name.replace("/",os.sep)
                
                if not os.path.exists(os.sep.join(output_path.split(os.sep)[:-1])):
                        os.makedirs(os.sep.join(output_path.split(os.sep)[:-1]))
                # Download image in cache folder 
                urllib.request.urlretrieve (local_meta_df.loc[sp,"external_path"],output_path)
            
            # Copy metadataframe to local storage, it will be used to check if all image are already in cache or not
            local_meta_df.reset_index().to_parquet(local_meta_path)
            input_df=local_meta_df.copy()
        
            print("cache directory_built")
  
    print("Creating dataloader . .")
    # Create a pytorch dataset from your dataset meta dataframe. 
    challenge_dataset = ChallengeWeldingDataset(input_df,(540,540),cache_strategy,cache_dir)

    # Create a pytorch dataloader from your dataset
    dataloader = DataLoader(challenge_dataset, batch_size=batch_size,
                        shuffle=shuffle, num_workers=num_workers)    
    print( "Dataloader created")
    return dataloader





