# This class satisfy the AI component interface required for your solution be accepted as challenge candidate.
# It contains as requested a class named "myAiComponent" containing the 2 requested methods (load_model(), and predict()).
# Of course you are free to add any other methods you may need. This is the case here .

from abc import ABC, abstractmethod
from challenge_welding.AIComponent_interface import AbstractAIComponent
from pathlib import Path


# You will need this variable ROOT_PATH to access from this code to any local file you added the directory "challenge_solution".
# For example , in this python file , if you want to access to a file named  "my_model.h5" . You shall acces it with this path ROOT_PATH/my_model.h5)

ROOT_PATH = Path(__file__).parent.resolve() # This point to the path of the challenge_solution folder of the installed pkg in the evaluation virtual env

class MyAIComponent(AbstractAIComponent):
    def __init__():
        """
         Init a ML-component 
        """ 
    
    )
            
    def predict(self,input_images, images_meta_informations):
        """
        Perform a prediction using the appropriate model.
        Parameters:
            input_images: A list of NumPy arrays representing the list of images where you need to make predictions.
            image_meta_informations: List of Metadata dictionaries corresponding to metadata of input images
        Returns:
            A dict containing 4 keys "predictions", "probabilities", "OOD_scores"(optional),"explainability"(optional). 
                predictions : A list of the predictions given by the AI component among 3 possible values [KO, OK UNKNOWN"]
                probabilities : A list of 3-values lists containing predicted scores for each sample in this order [proba KO, proba OK, proba UNKNOWN]. sum of proba shall be 1 for each lists  
                OOD_scores : A list of  OOD score predicted by the AI component for each sample. An ood score is a real positive number. The image is considered OOD when this score is >=1
                explainabilities  : a list of expllainabilities for each sample . An explainability , is an intensity matrix (a numpy array contaning only real numbers between 0 and 1)
        """

        return {"predictions" : None , "probabilities": None , "OOD_scores": None,  "explainabilities": None}

    
    def load_model(self):
        return None
    

