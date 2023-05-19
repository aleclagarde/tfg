import pandas as pd
import os

from inference_functions import inference
from inference_utils import download_datasets, add_measurements


models = ['gpt2', 'opt', 'resnet', 'regnet', 'codeparrot', 'codegpt']
number_of_measurements = 30
data_size = 100

df = pd.DataFrame(columns=['timestamp', 'project_name', 'run_id', 'duration', 'emissions', 'emissions_rate',
                           'cpu_power', 'gpu_power', 'ram_power', 'cpu_energy', 'gpu_energy', 'ram_energy',
                           'energy_consumed', 'country_name', 'country_iso_code', 'region', 'cloud_provider',
                           'cloud_region', 'os', 'python_version', 'cpu_count', 'cpu_model', 'gpu_count', 'gpu_model',
                           'longitude', 'latitude', 'ram_total_size', 'tracking_mode', 'on_cloud'])

print('DOWNLOADING DATASETS...')
download_datasets(data_size=data_size)
print('DATASETS DOWNLOADED')


for model_short_name in models:
    models_suffix = ['-torch-baseline', '-torch-pruned', '-torch-quantized.pth', '-tf-baseline', '-tf-pruned',
                     '-tf-quantized']
    for suf in models_suffix:
        model_name = model_short_name + suf
        model_correctness = []
        for i in range(number_of_measurements):
            correctness = inference(model_name, model_short_name)
            model_correctness.append(correctness)
        df = add_measurements(df, number_of_measurements=number_of_measurements, model_name=model_name,
                              correctness=model_correctness)

df.to_csv('inference_results.csv')

# Remove the emissions file
os.remove('emissions.csv')
