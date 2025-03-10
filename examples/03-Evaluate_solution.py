# This script show how an AI component can be evaluated 

import sys
# sys.path.insert(0, "..") # Uncomment this line For local tests without pkg installation, to make challenge_welding module visible 
from challenge_welding.user_interface import ChallengeUI
from challenge_welding.Evaluation_tools import EvaluationPipeline

#1) Load a dataset to test 

# In this example we will choose a small dataset

ds_name="example_mini_dataset"

# Load all metadata of your dataset as a pandas dataframe, (you can point to a local cache metafile instead of original one pointing on remote repository)

my_challenge_UI=ChallengeUI()
evaluation_ds_meta_df=my_challenge_UI.get_ds_metadata_dataframe(ds_name)

# Define path of AI component to test
AI_comp_path= "..\\reference-solutions\\Solution-1"

# Initialize test pipeline
myPipeline=EvaluationPipeline(proposed_solution_path=AI_comp_path, # Set here the AI component path you want to evaluate
                              meta_root_path="starter_kit_pipeline_results", # set name of output directory that will store the pipeline results
                              cache_strategy="local", # In this on local, all image used for evaluation , will be locally stored in a cache directory
                              cache_dir="evaluation_cache") # chosen directory for cache
                                                         
# Load the AI component in evaluation pipeline
myPipeline.load_proposed_solution()

# Perform inference on AI component on evaluation dataset, inference results are stored as parquet file and as df returned by the function too.

result_df=myPipeline.perform_grouped_inference(evaluation_dataset=evaluation_ds_meta_df,
                                       results_inference_path=myPipeline.meta_root_path+"/res_inference.parquet",
                                       batch_size=100) 


# Compute operationnal metrics
myPipeline.compute_operationnal_metrics(AIcomp_name="sol_0",
                                        res_inference_path=myPipeline.meta_root_path+"/res_inference.parquet"
                                        )

#Compute uncertainty metrics

res_df,final_results=myPipeline.compute_uncertainty_metrics(res_inference_path=myPipeline.meta_root_path+"/res_inference.parquet",
                                                            AIcomp_name="sol_0" # this field is just used for naming the results files
                                                            )                                                                                  

print("pipeline finished")
