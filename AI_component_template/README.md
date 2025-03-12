# challenge-welding-reference-solution-1

This repository contains a template to create your own AI component for the challenge-welding-reference-solution-1

# Step to build a compatible AI component

Starting from this template:

- Complete the myAIComponent class in file challenge/solution/AIcomponent.py by filling load_model() method and predict() methods that is required so that the evaluation process can interact with your compoent
- Complete the setup.py with your informations
- Fill the requirements.txt with your own dependencies
- Fill the MANIFEST.in file with all additional files you added in the challenge_solution folder to make your AI component working 


# Additional informations :

You are free to add as many file you want in the ```challenge_solution``` folder to make your predict and load_model method working. 
But in this case make sure to those added files in ```MANIFEST.in``` file to ensure that those files will be integrated in the package python of your AI component.

To access to these added files from your code in AIcomponent.py use the ROOT_PATH variable to ensure to point on the challenge_solution installed directory in the evaluation environnement


