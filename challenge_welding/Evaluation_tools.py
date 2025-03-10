"""
This module contains all code needed by the evaluation pipeline
"""

# Import here all required dependencies

import sys
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import challenge_welding.user_interface
from challenge_welding.U_metrics_V2 import U_metrics,curves_calibration_cost,compute_cost_matrix
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import yaml


def plot_confusion_matrix(y_true, y_pred, classes, output_file,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix. It is used  during the computation of operational metrics
    Parameters:
        y_true: list[str]
            List containing ground truth
        y_pred: list[str] or np;array
            List containing predictions
        classes: list[str]
            List of labels to use in confusion matrix plot
        output_file :str
            path to file to store confusion matrix as image
    
    Return numpy.array
        Numpy array containing the confusion amtrix
    """
   
    # Compute confusion matrix using sklearn
    cm = confusion_matrix(y_true, y_pred)
    print("confusion matrix")
    print(cm)
    
    # Plot confusion confusino as matplotlib figure
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    # Export confusionn matrix plot to png image
    plt.savefig(output_file)
    plt.close()
    return cm
  
#=============================================================================================  
class EvaluationPipeline:
    """
    This class instanciate an evaluation pipeline. With it, you can use the AI component in inference and generate different quality metrics.
    For now,  only 2 quality metrics are integrated for demonstration : operationnal metrics, and uncertainty metrics.
    Parameters:
        proposed_solution_path : str 
            Path to the local dirictory containing the python package of the AI component to evaluate 
        meta_root_path : str
            Path to local directory where all results file generated by pipeline are stored
        cache_strategy: str in ["local","remote"]
            If on "local" : all image used in pipeline for evaluation are locally downloaded in cache in the first time. If on "remote" no cache is used and images are always downloaded when used
        cache_dir : str
            Path of the cache directory used when it is activated with cache_strategy
    """  
    def __init__(self,proposed_solution_path,meta_root_path,cache_strategy="remote",cache_dir=".cache_pipeline"):
        
        self.meta_root_path=meta_root_path
        self.AI_component_path=proposed_solution_path   
        self.cache_strategy=cache_strategy
        self.cache_dir=cache_dir

        # The two next line are just for debugging the component execution without having to install its package

        # if self.AI_component_path!=None:
            # sys.path.insert(0, proposed_solution_path) # Add AI component path to python path thus the AI_component class is accessible without pkg installation 
                        
    def load_proposed_solution(self):
        """
        This method load the AI component as member of this class in field self.AIComponent. This process is divided into two main steps :
        - Installation of the AI comp package in the evaluation python environement
        - Call of the load_model() method of the AI component interface.
        """
        
        import subprocess
        
        req_file_path=self.AI_component_path+"\\requirements.txt"
        
        # Save the pipeline active directory
        pipeline_dir=os.getcwd()
        
        # Installation of AI component package in the active virtual environement
        
        subprocess.check_call(["pip", "install", self.AI_component_path])
        
        # Set python interpreter at the root directory of AI component (in case of Ai component load_model() method need to access local files)
        os.chdir(self.AI_component_path)
        from challenge_solution.AIComponent import MyAIComponent
        
        # Init the AI component via its interface
        self.AIComponent=MyAIComponent()
             
        # Load the AI component in pipeline class calling its interface load method .
        self.AIComponent.load_model()
        print("AI component loaded")
        
        # Move back python interpreter active directory into pipeline directory
        
        os.chdir(pipeline_dir)
      
    def perform_grouped_inference(self,evaluation_dataset,results_inference_path,batch_size=10):
        """
        This function perform inference of each sample of the input dataset. It adds results fields ["predicted_state", "score_OK", score KO"] 
        to the evaluation dataset metainfo dataframe given as input and store the obtained new dataframe in a parquet file.
  
        Parameters: 
            evaluation_dataset: pandas.DataFrame
                DatraFrame representing metadata of your evaluation dataset
            result_inference_path : str
                Path of inference results parquet file
            batch_size: int
                Number of grouped samples you pass as input when calling the AI component

        Return : pandas.DataFrame
            Pandas dataframe containing evaluation dataset metadata dataframe augmented with inference results
    """
    
        #Load metadata of eval dataset and copy them in a new dataframe for working .

        output_df=evaluation_dataset.copy()
        output_df.set_index("sample_id",inplace=True)
        total_deb=time.time()
        
        # Initialize connector to challenge data
        user_ui=challenge_welding.user_interface.ChallengeUI(cache_strategy=self.cache_strategy,cache_dir=self.cache_dir)
           
        # Prepare working variable for inference computation (number of batch, and bound indexes of batch in datasets metadataframe
        nb_batch= (len(output_df)//batch_size)
        
        if len(output_df) % batch_size!=0:
            nb_batch+=1
        
        # Init current batch bound indexes in metadata dataframe
        batch_start=0
        batch_end=min(batch_size,len(output_df)) # to ensure good work with case where batch_size> len(output_df)
        
        print("Number of  batch to process for inference : ", nb_batch, " , start processing..")
        
        # Process each batch ..
        for batch_idx in tqdm(range(0,nb_batch)):
            batch_sample_ids=[] 
            batch_images_data=[] 
            batch_images_meta=[] 
            
            for sp_idx in range(batch_start,batch_end): # Iterate on each sample in the current batch 
                
                # Get metadata of your image
        
                sample_id=output_df.index[sp_idx] 
                image_meta=output_df.loc[sample_id]
       
                # Get  numpy array of the image
                image_data=user_ui.open_image(image_meta["path"])
                
                # Add them to batch list storages
                batch_sample_ids.append(sample_id)
                batch_images_meta.append(image_meta)
                batch_images_data.append(image_data)
        
            # Call AI component and pass the batch content as input
            
            inference_start_time = time.time()           
            results=self.AIComponent.predict(batch_images_data,batch_images_meta)
            inference_end_time=time.time()
            
            # Update meta dataframe with inference results for all samples in the current batch
            
            output_df.loc[batch_sample_ids,"predicted_state"]=results["predictions"]
            output_df.loc[batch_sample_ids,"scores KO"]=[x[0] for x in results["probabilities"]]
            output_df.loc[batch_sample_ids,"scores OK"]=[x[1] for x in results["probabilities"]]
            output_df.loc[batch_sample_ids,"scores OOD"]=results["OOD_scores"]
            output_df.loc[batch_sample_ids,"compute_time"]=(inference_end_time-inference_start_time)/len(batch_sample_ids)
            
            # Update batch bound index for next iteration
            
            batch_start+=batch_size
            batch_end+=batch_size
            if batch_end >= len(output_df): # cut last batch size if necessary
                batch_end=len(output_df)
        
        # Export inference results as parquet file 
        
        output_path=results_inference_path
        
        # Create results output directory  if it does not exists
        if not output_path.split("/")[0] in ["", output_path]: 
                if not os.path.exists("/".join(output_path.split("/")[:-1])):
                    os.makedirs("/".join(output_path.split("/")[:-1]))

        # Export metadataframe to parquet file              
        output_df.to_parquet(output_path) 
        total_fin=time.time()
        print("cumulated inference time ",total_fin-total_deb)

        return output_df
    
      
    def compute_operationnal_metrics(self,AIcomp_name,res_inference_path):
        """
        This methoid generate operationnal metrics used to evaluate solution quality  
        It use only internal result inference parquet file of inference process as input.
        The computed metrics are stored in a yaml file named "operationnal_scores.yaml" and stored in the pipeline results folder.
        The  confusion matrix generated during computations is also stored as png image in the pipeline results folder.

        Parameters:
            AIcomp_name: str
                 Name that will be just used as name for metric results output folder 
            res_inference_path : str
                Path to parquet file containing inference results of your AI component
        """
        
        # Read inference file
        res_df=pd.read_parquet(res_inference_path)       
        output_path=self.meta_root_path+"/operationnal_cm.png"
        
        # Create outupt metric results directory if it does not exists
        if not output_path.split("/")[0] in ["", output_path]: # Create meta folder if it does not exists
                if not os.path.exists("/".join(output_path.split("/")[:-1])):
                    os.makedirs("/".join(output_path.split("/")[:-1]))
        
        # Compute the confusion matrix
        cm=plot_confusion_matrix(res_df["class"],
                          res_df["predicted_state"],
                          ["KO","OK","UNKNOWN"],
                          output_path,
                          title='Confusion matrix on all seams',
                          cmap=plt.cm.Blues)
        
        print("confusion matrix OK")
        
        #==================================================
        # Computation of operational costs part
        #==================================================
        
        # Get working variables 
        if cm.shape[0]==2: # If there is no UNKNOWN in predictions
            nb_pred_unknown=0
        else:
            nb_pred_unknown=cm[:,2].sum() # Number of weldings predicted as Unknown
        nb_pred_KO=cm[:,0].sum() # Number of weldings predicted as KO
        nb_FN=cm[0,1] # Number of false negative
        nb_operator_controlled=nb_pred_unknown+nb_pred_KO # Number of image that will be operationnaly controlled by operator (meaning weldings predicted Unknown or KO by AI comp)
        total_nb_welding=cm.sum() # Total number of welding to be processed
                    
        print("Number of operator controlled image", nb_operator_controlled)
        
        # Cost of single analysis 
        cost_operator_analyze=0 # unit cost of qualify a weld image by human
        cost_IA_analyze=0 # unit cost of qualify a weld image by AI
       
        # Cost if all welding would have been controlled by human (those formulas could be changed)
       
        cost_human_only= total_nb_welding * (4.05 + cost_operator_analyze) # If N is total_nb_welding to control , Cost = N[3/1000* 1000€] + N[7/1000 * 150€]--> N*4.05 

        # Cost with human + AIcomp collaboration
        cost_operator= nb_operator_controlled*(4.05 + cost_operator_analyze) # cost due to image thatt come back to operator for control
        cost_AI=nb_FN*1000 + total_nb_welding*cost_IA_analyze # 
        cost_human_AI= cost_AI+cost_operator
        
        #==================================================
        # Computation of inference time metric
        #==================================================   
        inf_condition=res_df["compute_time"]<0.083 # 0.083 is that maximal time operationnaly allowed for a image state prediction
        inference_score=inf_condition.sum()/len(res_df)*100 # proportion of sample satisfying inference time requirement
      
        #==================================================
        # Final export
        #==================================================  
      
        # Store all metrics in a dict
        final_results={"human_only_cost": int(cost_human_only), 
                        "human+AI_cost" : int(cost_human_AI),
                        "gain in Euros" : int(cost_human_only-cost_human_AI),
                        "inference_score": int(inference_score)}
        
        print(final_results)
        
        # Export dict as yaml file
        
        # Set metrics file path 
        output_path=self.meta_root_path+"/operationnal_scores.yaml"
        
        # Create directory if it doesnot exists
        if not output_path.split("/")[0] in ["", output_path]: # Create meta folder if it does not exists
                if not os.path.exists("/".join(output_path.split("/")[:-1])):
                    os.makedirs("/".join(output_path.split("/")[:-1]))
        
        # Write the yaml file at output path
        with open(output_path, 'w') as outfile:
                yaml.dump(final_results, outfile, default_flow_style=False)
       
        return None
        
        
    def transform_class(self,label):
        """
        Transformation function to encodes the predictions column from strings to integers. This function is necessary to be compatible with some sklearn 
        functions used in the uncertainty metrics computation function

        Parameters :
            label : str
                Input label to transform 

        Return : int
            Integer representing the converted label

        """   
        if label=="OK":
            return 1
        elif label=="KO":
            return 0
        else:
            return 2
    
    def compute_uncertainty_metrics(self,res_inference_path,AIcomp_name):
        """
        This method generates operationnal metrics used to evaluate the Ai component quality  
        It use only internal result_parquet file as input.
        The computed uncertainty metrics are stored in a yaml file named "uncertainty_scores.yaml" stored in the pipeline results folder.

        Parameters:
            AIComp_name: str
                Name that will be used as output folder name
            res_inference_path : str
                Path to parquet file containing inference results of your AI component 
        """
        
        # Read inference file
        res_df=pd.read_parquet(res_inference_path)
           
        # Transform predictions format to match with those required by uncertainty evaluation functions
        res_df["class_one_hot"]=res_df["class"].apply(lambda x: self.transform_class(x))
        res_df["predicted_one_hot"]=res_df["predicted_state"].apply(lambda x: self.transform_class(x))
        
        # Start uncertainty metrics computation (score u_gain and score_calib)
       
        score_u_gain = U_metrics(res_df["class_one_hot"].values,(res_df["predicted_one_hot"].values,res_df["scores OK"].values),epsilon=1e-10,n_class=2,with_unknown=True,
        mode='diff_gap_prob_quad',cost_matrix=compute_cost_matrix())

        score_calib = curves_calibration_cost(res_df["class_one_hot"].values,(res_df["predicted_one_hot"].values,res_df["scores OK"].values),epsilon=1e-10,n_class=2,with_unknown=True,
        mode='quantile',n_bins=20)        
                                 
        # Store all uncertainty all metrics in dict                         
        final_results={"score_u_gain": -1*float(score_u_gain), 
                       "score_calib" : float(score_calib)
                       }
        
        print(final_results)
        
        # Export dict as yaml file
        
        # Set metrics file path 
        output_path=self.meta_root_path+"/uncertainty_scores.yaml"
        
        # Create directory if it doesnot exists
        if not output_path.split("/")[0] in ["", output_path]: # Create meta folder if it does not exists
                if not os.path.exists("/".join(output_path.split("/")[:-1])):
                    os.makedirs("/".join(output_path.split("/")[:-1]))
        
        # Write the yaml file at output path
        with open(output_path, 'w') as outfile:
                yaml.dump(final_results, outfile, default_flow_style=False)
        
        return res_df,final_results