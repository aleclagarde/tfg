"""
Inference script
================

.. module:: inference
   :platform: Linux
   :synopsis: Inference script.

.. module_author:: Alec Lagarde Teixidó <aleclagarde@gmail.com>

This script performs the whole process of inference. That is, loading the models and infer it for ``data_size`` data
points, computing the mean of the correctness of these data points and save it for the results' dataset.
"""

# Dataframe
import pandas as pd

# Remove automatically generated emissions file
import os

# Inference functions
from inference_functions import inference
from inference_utils import download_datasets, add_measurements


# Define the models, number of measurements and number of data points
models = ['gpt2', 'opt', 'resnet', 'regnet', 'codeparrot', 'codegpt']
number_of_measurements = 30
data_size = 50

# Initialize results dataframe
df = pd.DataFrame(columns=['timestamp', 'project_name', 'run_id', 'duration', 'emissions', 'emissions_rate',
                           'cpu_power', 'gpu_power', 'ram_power', 'cpu_energy', 'gpu_energy', 'ram_energy',
                           'energy_consumed', 'country_name', 'country_iso_code', 'region', 'cloud_provider',
                           'cloud_region', 'os', 'python_version', 'cpu_count', 'cpu_model', 'gpu_count', 'gpu_model',
                           'longitude', 'latitude', 'ram_total_size', 'tracking_mode', 'on_cloud'])

# Download datasets and save them in disk
print('DOWNLOADING DATASETS...')
download_datasets(data_size=data_size)
print('DATASETS DOWNLOADED')

# For each model take every version
for model_short_name in models:
    models_suffix = ['-torch-baseline', '-torch-pruned', '-torch-quantized.pth', '-tf-baseline', '-tf-pruned',
                     '-tf-quantized']
    for suf in models_suffix:
        model_name = model_short_name + suf
        model_correctness = []
        # For every model version repeat number_of_measurements times the inference
        for i in range(number_of_measurements):
            correctness = inference(model_name, model_short_name)
            model_correctness.append(correctness)
        # Add the new measurements to the dataframe
        df = add_measurements(df, number_of_measurements=number_of_measurements, model_name=model_name,
                              correctness=model_correctness)

# Save dataframe in disk
df.to_csv('../results/inference_results.csv')

# Remove the emissions file
os.remove('emissions.csv')
