# Welding quality detection challenge starter-kit

This repository contain a python package containing useful functions to help the challenge users to interact with the datasets.
We consider in the following sections a dataset as a collection of samples. In our challenge, a "sample" is a single image. 

We recall the website of this challenge: https://confianceai.github.io/Welding-Quality-Detection-Challenge/

For any issues or technical support : please contact challenge.confiance@irt-systemx.fr

All code present in this repository has been tested with python 3.12.7.

# Preparing your environnement

## Create and activate your virtual environnement
To create a virtual environnement, you can use many different tools as python virtual environments (venv), [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [uv](https://github.com/astral-sh/uv). Conda and uv has the advantage of making possible to choose the python version  you want to use in your virtual env by setting X.Y in the commands below:

If using python integrated venv , to create a new virtual environnement, type in a terminal : 

 ```commandline
 python -m venv path_to_your_env
 ``` 

If using conda : 

```commandline
conda create -n path_to_your_env python=X.Y 
```

If using uv :

```commandline
uv venv path_to_your_env --python=X.Y
```

## Activate your virtual env

On Windows power shell 
```commandline
./path_to_your_env/Scripts/activate
```

On Linux: 
```commandline
source path_to_your_env/bin/activate
```

## Installation of the ChallengeWelding package
To install the package Challenge Welding and its dependencies, from the root directory of this repository type:  
```commandline 
pip install .
```

# Starter kit content
The scripts and Jupyter notebooks are provided in [example](examples) directory to guide the participants through the different utilities developed to facilitate the dataset manipulation and AI component development and evaluation:

- ```01-Tutorial.py``` : This code shows how to use main user functions present in this package. Wihtin, there are examples about how to listing available datasets, explore metadata, and draw basic statistics on contextual variables.

- ```02-Create_pytorch_dataloader.py``` : This code shows how to use this package to create a Pytorch dataloader.

- ```03-Test_AIComponent.py``` : This script shows how to load an AI component and evaluate it by generating operational and uncertainty metrics. 

These scripts are also available as jupyter-notebooks, where we provide more information concerning the usecase, the dataset and the AI component to be developed and evaluated: 

- ```01-Tutorial.ipynb``` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/confianceai/Challenge-Welding-Starter-Kit/blob/main/examples/01-Tutorial.ipynb) 

- ```02-Create_pytorch_dataloader.ipynb``` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/confianceai/Challenge-Welding-Starter-Kit/blob/main/examples/02-Create_pytorch_dataloader.ipynb) 

- ```03-Test_AIComponent.ipynb``` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/confianceai/Challenge-Welding-Starter-Kit/blob/main/examples/03-Test_AIComponent.ipynb) 

# Dataset Storage organisation

The list of available datasets is present in a yaml file at the following url

```https://minio-storage.apps.confianceai-public.irtsysx.fr/challenge-welding/datasets_list.yml```

Each dataset has a parquet file that contains the metadata of all samples present in the dataset.
This file is accessible for a dataset named : "YOUR_DS_NAME" at :  

```https://minio-storage.apps.confianceai-public.irtsysx.fr/challenge-welding/datasets/YOUR_DS_NAME/metadata/ds_meta.parquet```

Look the [Dataset informations](docs/Dataset_description.md) for more informations about metadescription contents

# Expected AI solution

Look the [Solution requirements and evaluation process](docs/Requirements_and_Evaluation_process.md) to get informations about the expected solution and the way it will be evaluated

An exemple of AI Component is provided within this repository but only to show an example of the expected architecture. The model has not been trained to be performant and efficient for this challenge.













- ```01-Tutorial.py``` : This code shows how to use main user functions present in this package. Wihtin, there are examples about how to listing available datasets, explore metadata, and draw basic statistics on contextual variables.
- ```02-Create_pytorch_dataloader.py``` : This code show how to use this package to create a Pytorch dataloader. Run this
example will require to install an additionnal dependency torch==2.6.0
- ```03-Test_AIComponent.py``` : This script describe how you can build your own AI component and test it to check if it is compatible with the evaluation pipeline

You have quivalent examples as jupyter-notebooks.
- ```01-Tutorial.ipynb``` : 
- ```02-Create_pytorch_dataloader.ipynb```
- ```03-Test_AIComponent.ipynb```

# Dataset Storage organisation

The list of available datasets is present in a yaml file at the following url

```https://minio-storage.apps.confianceai-public.irtsysx.fr/challenge-welding/datasets_list.yml```

Each dataset has a parquet file that contains the metadata of all samples present in the dataset.
This file is accessible for a dataset named : "YOUR_DS_NAME" at :  

```https://minio-storage.apps.confianceai-public.irtsysx.fr/challenge-welding/datasets/YOUR_DS_NAME/metadata/ds_meta.parquet```

Look the [Dataset informations](docs/Dataset_description.md) for more informations about metadescription contents

# Expected AI solution

Look the [Solution requirements and evaluation process](docs/Requirements_and_Evaluation_process.md) to get informations about the expected solution and the way it will be evaluated

An exemple of AI Component is provided within this repository but only to show an example of the expected architecture. The model has not been trained to be performant and efficient for this challenge.





