# Energy efficiency measurement in optimization and inference of ML models

**Author: Alec Lagarde Teixidó**

**Director: Silverio Martínez-Fernández**

**Co-director: Matias Martinez**

The research of this TFG consists of understanding how existing ML optimization frameworks impact the energy efficiency 
and accuracy of ML models. We will study the following aspects of ML models deployment:

1. Energy consumed measurement when applying model optimization (e.g., quantization, pruning).
2. Energy consumed and correctness measurement when applying model inference.

This project was developed from January to June 2023 as a Bachelor's Degree thesis for the Data Science and Engineering 
Degree in the Universitat Politècnica de Catalunya.

----------

## Environment
To replicate this projects environment, simply run ``conda create --name tfg python=3.10``. This will create the 
environment (named **tfg**) with Python 3.10.

Once the environment is created, activate the environment with ``conda activate tfg``, after that, you can install all 
necessary packages with ``pip install -r requirements.txt``. It is necessary to install the packages with pip instead of 
conda because conda does not work for some packages.

----------

## Structure and usage

It is structured as follows:

```
tfg
├── README.md <- The top-level README for this project.
├── requirements.txt <- The requirements file to reproduce the project environment.
│
│
├── analysis
│ └── analysis.ipynb <- The notebook that performs the results analysis and answers the RQs.
│
│
├── data
│ ├── image_dataset <- Dataset for the Computer Vision models' inference.
│ ├── code_dataset.txt <- Dataset for the Code Generation models' inference.
│ ├── imagenet1000_idx_to_labels.txt <- Dictionary that translates the ImageNet labels from int to string.
│ └── text_dataset.txt <- Dataset for the Text Generation models' inference.
│
│
├── results
│ ├── inference_results.csv <- Results from the inference phase.
│ └── optimization_results.csv <- Results from the optimization phase.
│
│
└── src
  ├── models
  │ ├── saved <- Directory that stores the models.
  │ ├── get_model_objects.py <- Auxiliary script to access the objects of the desired model.
  │ ├── optimize.py <- Optimization script.
  │ └── optimize_utils.py <- Auxiliary optimization functions.
  │
  ├── inference.py <- Inference script.
  ├── inference_functions.py <- Auxiliary script with functions that perform the inference.
  └── inference_utils.py <- Auxiliary inference functions.
```

The first step is optimizing and saving the models. This is done by executing the ``optimize.py`` script. Once the
execution of the script ends, the results will be saved in the **results** directory as ``optimization_results.csv``.

Then, with all the models saved, we can execute the ``inference.py`` script, which performs inference with all the 
models saved in the optimization phase. As before, when the execution ends the results are stored in the **results**
directory as ``inference_results.csv``.

Finally, all that it is left to do is to analyze the two results datasets. The code used in this project is in the 
notebook ``analysis.ipynb``.
