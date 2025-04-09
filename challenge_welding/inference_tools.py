"""
This module contains all code needed by the evaluation pipeline
"""

# Import here all required dependencies

import os
import sys
import time
import subprocess
from tqdm import tqdm
import challenge_welding.user_interface


# =============================================================================================
class TestAIComponent:
    """
    This class is made to  test if your AI component is compatible with the evaluation pipeline that will be used to
    evaluate your solution.
    It integrates the load process the AI component in the pipeline, and the inference call on an input dataset.
    Score computation functions are not provided here, but they all rely only the inference results as input.
    Thus if the inference process is workin it ensures that the scores computations on your Ai component will work too.


    Parameters:
        proposed_solution_path : str
            Path to the local dirictory containing the python package of the AI component to evaluate
        meta_root_path : str
            Path to local directory where all results file generated by pipeline are stored
        cache_strategy: str in ["local","remote"]
            If on "local" : all image used in pipeline for evaluation are locally downloaded in cache in the first time.
            If on "remote" no cache is used and images are always downloaded when used
        cache_dir : str
            Path of the cache directory used when it is activated with cache_strategy
    """

    def __init__(
        self,
        proposed_solution_path,
        meta_root_path,
        cache_strategy="remote",
        cache_dir=".cache_pipeline",
    ):
        self.meta_root_path = meta_root_path
        self.ai_component_path = proposed_solution_path
        self.ai_component = None
        self.cache_strategy = cache_strategy
        self.cache_dir = cache_dir

        # The two next line are just for debugging the component execution without having to install its package

        # if self.ai_component_path!=None:
        # sys.path.insert(0, proposed_solution_path) # Add AI component path to python path thus the AI_component class
        # is accessible without pkg installation

    def load_proposed_solution(self):
        """
        This method load the AI component as member of this class in field self.AIComponent.
        This process is divided into two main steps :
        - Installation of the AI comp package in the evaluation python environement
        - Call of the load_model() method of the AI component interface.
        """

        #if "http" in self.ai_component_path:
            #if the path in an url on git repository
            #self.ai_component_path = "git+" + self.ai_component_path

        # Installation of AI component package in the active test virtual environement
        #subprocess.check_call(["pip", "install", self.ai_component_path])

        sys.path.insert(0, self.ai_component_path)  # Ajout du chemin du composant à PYTHONPATH

        # Import your AI component in the test environnement
        from challenge_solution.AIComponent import MyAIComponent
        # Init the AI component via its interface
        self.ai_component = MyAIComponent()

        # Load the AI component in pipeline class calling its interface load method .
        self.ai_component.load_model()
        print("AI component loaded")

    def perform_grouped_inference(
        self, evaluation_dataset, results_inference_path, batch_size=10
    ):
        """
        This function perform inference of each sample of the input dataset.
        It adds results fields ["predicted_state", "score_OK", score KO"] to the evaluation dataset metainfo dataframe
        given as input and store the obtained new dataframe in a parquet file.

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

        # Load metadata of eval dataset and copy them in a new dataframe for working .

        output_df = evaluation_dataset.copy()
        output_df.set_index("sample_id", inplace=True)
        total_deb = time.time()

        # Initialize connector to challenge data
        user_ui = challenge_welding.user_interface.ChallengeUI(
            cache_strategy=self.cache_strategy, cache_dir=self.cache_dir
        )

        # Prepare working variable for inference computation (number of batch, and bound indexes of batch in datasets
        # metadataframe
        nb_batch = len(output_df) // batch_size

        if len(output_df) % batch_size != 0:
            nb_batch += 1

        # Init current batch bound indexes in metadata dataframe
        batch_start = 0
        batch_end = min(
            batch_size, len(output_df)
        )  # to ensure good work with case where batch_size> len(output_df)

        print(
            "Number of  batch to process for inference : ",
            nb_batch,
            " , start processing..",
        )

        # Process each batch ..
        for batch_idx in tqdm(range(0, nb_batch)):
            batch_sample_ids = []
            batch_images_data = []
            batch_images_meta = []
            print(batch_idx)
            for sp_idx in range(
                batch_start, batch_end
            ):  # Iterate on each sample in the current batch
                # Get metadata of your image

                sample_id = output_df.index[sp_idx]
                image_meta = output_df.loc[sample_id]

                # Get  numpy array of the image
                image_data = user_ui.open_image(image_meta["path"])

                # Add them to batch list storages
                batch_sample_ids.append(sample_id)
                batch_images_meta.append(image_meta)
                batch_images_data.append(image_data)

            # Call AI component predict method  and pass the batch content as input

            inference_start_time = time.time()
            results = self.ai_component.predict(batch_images_data, batch_images_meta, device='cuda')
            inference_end_time = time.time()

            # Update meta dataframe with inference results for all samples in the current batch

            output_df.loc[batch_sample_ids, "predicted_state"] = results["predictions"]
            output_df.loc[batch_sample_ids, "scores KO"] = [
                x[0] for x in results["probabilities"]
            ]
            output_df.loc[batch_sample_ids, "scores OK"] = [
                x[1] for x in results["probabilities"]
            ]
            output_df.loc[batch_sample_ids, "scores UNKNOWN"] = [
                x[2] for x in results["probabilities"]
            ]
            output_df.loc[batch_sample_ids, "scores OOD"] = results["OOD_scores"]
            output_df.loc[batch_sample_ids, "compute_time"] = (
                inference_end_time - inference_start_time
            ) / len(batch_sample_ids)

            # Update batch bound index for next iteration

            batch_start += batch_size
            batch_end += batch_size
            if batch_end >= len(output_df):  # cut last batch size if necessary
                batch_end = len(output_df)

        # Export inference results as parquet file

        output_path = results_inference_path

        # Create results output directory  if it does not exists
        if output_path.split("/")[0] not in ["", output_path]:
            if not os.path.exists("/".join(output_path.split("/")[:-1])):
                os.makedirs("/".join(output_path.split("/")[:-1]))

        # Export metadataframe to parquet file
        output_df.to_parquet(output_path)
        total_fin = time.time()
        print("cumulated inference time ", total_fin - total_deb)

        return output_df
