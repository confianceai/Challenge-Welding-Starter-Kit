"""
This module contains a class of useful functions to help the user of the challenge Welding to interact with the datasets
"""

# list all required imports
import pandas as pd
import requests
from io import BytesIO
import hashlib
import yaml
from PIL import Image
import numpy as np
from tqdm import tqdm
import numpy as np
import os
import urllib.request

class ChallengeUI:
    """This class provide the user some useful functions to interact with the datasets of the challenge Welding"""
    
    def __init__(self, 
                cache_strategy="remote",
                cache_dir=".cache", 
                base_url="https://minio-storage.apps.confianceai-public.irtsysx.fr/",
                ds_meta_path="challenge-welding/datasets_list.yml"):
        """
        Inputs:

            cache_strategy:  str  in ["local", "remote"]
                If it is set to "local", all images will be locally stored in a cache directory, when used.
                If if is set on "remote", all image will stay on remote server when used.
            
            cache_dir: str
                Directory that shall be used to store cache data

            base_url : str
                Root url to the challenge storage

            ds_meta_path : str
                Relative path to dataset list in challenge storage

            """
        
        if not cache_strategy in ["local","remote"]:
            raise Exception("Error the cache strategy is not recognized")

        self.cache_strategy=cache_strategy
        self.cache_dir=cache_dir
        self.base_url=base_url
        self.ds_list=self.base_url+ds_meta_path 

    def list_datasets(self):
        """Class method list the names of datasets available in the challenge
        
        Returns:
        
        list : A list containing the names of all datasets present and accessible in the challenge
        
        """
        response = requests.get(self.ds_list, allow_redirects=True)
        if response.status_code==200:
           return yaml.safe_load(response.content.decode("utf-8"))  
        else:
           raise Exception(" Error, the file listing the datasets is missing from the repository, Please contact the challenge support team")
                   

    def get_ds_metadata_dataframe(self,ds_name):
        """This method returns a pandas dataframe containing all metadata of the dataset whose name is given as input
        
        Args:
        
        ds_name : str
            Name of the dataset you want to retrieve the metadata
            
        Returns:

        pandas.DataFrame : A dataframe containing all you dataset metadata
        
        """
        try:
            ds_meta_path=self.base_url+"challenge-welding/datasets/"+ds_name+"/metadata/ds_meta.parquet"
            print(ds_meta_path)
            return pd.read_parquet(ds_meta_path)
        except:
            raise Exception("The dataset", ds_name, "does not exist in the repository")

    def open_image(self,image_url):
        
        """This method return the numpy array of the image whose url is given as input
        
        Args:
        
        image_url : str
            Url of the image that you want to open
               
        Returns: numpy.array : 
            Numpy array representing the tensor of the input image
        
        """
        
        remote_image_url=self.base_url+image_url 
        
        if self.cache_strategy=="local": # If local cache is activated
            local_image_path=self.cache_dir+os.sep+image_url
            local_image_path=local_image_path.replace("/",os.sep)
            
            if os.path.exists(local_image_path):  # If the image is present in cache open it directly 
                
                # print("image found in local cache,direct opening")
                return np.asarray(Image.open(local_image_path)) 
                
            else: # else download it from remote repository . .
                
                # print("image not found in cache , downloading the file . .")
                # print("local_image_path",local_image_path)
                
                response=requests.get(remote_image_url)
                if response.status_code==200:
                    img = Image.open(BytesIO(response.content)) 
                else:
                    raise Exception(" Error , there is no image present on the repository, at this url", remote_image_url)
                    
                # then store it in cache directory
                if not os.path.exists(os.sep.join(local_image_path.split(os.sep)[:-1])):
                    os.makedirs(os.sep.join(local_image_path.split(os.sep)[:-1]))
                img.save(local_image_path)

        else: # Directly download image from remote repository
            response=requests.get(remote_image_url)
            if response.status_code==200:
                img = Image.open(BytesIO(response.content)) 
            else:
                raise Exception(" Error , there is no image present on the repository at this url", image_url)
        
        return np.asarray(img)
        
    def check_integrity(self,ds_name):
        """This method check the integrity of the dataset whose name is passed as input by comparing for each sample the sha256 of file stored
        with those present in the metadata . The list of abnormal sample is stored in a yaml file named anomalies_sample_list.yaml 
        and returned as output of this method too.
        
        Args:
            
        ds_name :str
            Name of the dataset you want to check the integrity
            
        Return :
            A list contaning the informations (idx, sample_id, and path ) of all abnormal samples
        """
        
        # Get metadata of given dataset
        
        ds_meta_path=self.base_url+"challenge-welding/datasets/"+ds_name+"/metadata/ds_meta.parquet"
        meta_df=pd.read_parquet(ds_meta_path) 
        
        anomalous_samples_list=[] # This list will store samples with anomalies 
        
        # For each sample in dataset
        print("Begin checking integrity of dataset : ", ds_name, " . . . ")
        for i in tqdm(meta_df.index):
            
            # Get the sha256 of the image present in the repository at the external_path"
            
            response = requests.get(meta_df.iloc[i]["external_path"])
            datastream=BytesIO(response.content)
            hash = hashlib.file_digest(datastream, "sha256") 
            sha_image = hash.digest()
            
            # If this sha256 is different from those stored in sample metadata
            
            if sha_image !=meta_df.iloc[i]["sha256"]: # 

                # Add this sample to the list of anomalous samples
                anomalous_sample={"index" : i, "sample_id": meta_df.iloc[i]["sample_id"], "full_path": meta_df.iloc[i]["external_path"]}
                anomalous_samples_list.append(anomalous_sample)
                print("\n anomalous image detected : ", anomalous_sample)
        
        
        # Export the final list of anormal samples as yaml file
        with open('anomalous_samples_list.yml', 'w', encoding = "utf-8") as yaml_file:
            dump = yaml.safe_dump(anomalous_samples_list)
            yaml_file.write(dump)
        
        print("integrity checking is complete")
        
        # If there are abnormal samples that were detected send a message to user.
        
        if len(anomalous_samples_list)>0:
            print("Warning : there are ", len(anomalous_samples_list), "samples that are corrupted or missing, check anomalous_samples_list.yml file for the detailed list, please contact support team")
        
        return anomalous_samples_list
      
    def create_pytorch_dataloader(self,input_df,batch_size,shuffle=False, num_workers=0):
           
        """This method create a pytorch dataloader from the input dataframe 
            
            Args:
                
            input_df :pd.DataFrame
                Dataframe containing the metadata of the dataset for which you want to create a dataloader
                           
            batch_size : int
                Size of the batch
                
            num_worker : int
                Number of workers to be used
            shuffle: bool

            Return :
                A pytorch dataloader browsing dataset covered by your input meta dataframe
        """
        # Import pytorch specific requirements for this method

        from torch.utils.data import Dataset, DataLoader  
        from challenge_welding.tools import ChallengeWeldingDataset
         
        # If local cache is activated         
        if self.cache_strategy=="local": 
            print("Cache storage has been activated in ",self.cache_dir)
            
            unique_id=int(input_df.iloc[0]["sample_id"].split("_")[-1])+int(input_df.iloc[-1]["sample_id"].split("_")[-1]) # unique _id associated with input dataframe
            print("cache_metadata_unique_id",unique_id)
            local_meta_path=self.cache_dir+os.sep+"local_"+str(unique_id)+"_storage_meta.parquet"
            
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
                    output_path=self.cache_dir+os.sep+target_image_name.replace("/",os.sep)
                    print(output_path)
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
        challenge_dataset = ChallengeWeldingDataset(input_df,(540,540),self.cache_strategy,self.cache_dir)

        # Create a pytorch dataloader from your dataset
        dataloader = DataLoader(challenge_dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)    
        
        return dataloader
    
    

