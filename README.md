# Energy Efficiency Measurement in Pruned and Quantized ML Models

This in progress research consists of understanding how existing ML optimization frameworks impact the 
energy efficiency and accuracy of ML models. We will study the energy consumed and correctness measurement when applying 
model inference.

----------

## Environment
To replicate this projects environment, simply run ``conda create --name <environment name> python=3.10``. This will 
create the environment with Python 3.10.

Once the environment is created, activate the environment with ``conda activate <environment name>``, after that, you 
can install all necessary packages with ``pip install -r requirements.txt``. It is necessary to install the packages
with pip instead of conda because conda does not work for some packages.

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
