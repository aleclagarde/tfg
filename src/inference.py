import pandas as pd
import os

from inference_functions import bert, t5
from inference_utils import load_model


models = ['bert', 't5']
data_size = 1

df = pd.DataFrame(columns=['timestamp', 'project_name', 'run_id', 'duration', 'emissions', 'emissions_rate',
                           'cpu_power', 'gpu_power', 'ram_power', 'cpu_energy', 'gpu_energy', 'ram_energy',
                           'energy_consumed', 'country_name', 'country_iso_code', 'region', 'cloud_provider',
                           'cloud_region', 'os', 'python_version', 'cpu_count', 'cpu_model', 'gpu_count', 'gpu_model',
                           'longitude', 'latitude', 'ram_total_size', 'tracking_mode', 'on_cloud', 'domain', 'model',
                           'data_number', 'correct'])

for model_short_name in models:
    models_suffix = ['-torch-baseline', '-torch-pruned', '-torch-quantized.pth', '-tf-baseline', '-tf-pruned',
                     '-tf-quantized']
    for suf in models_suffix:
        model_name = model_short_name + suf

        print("#############################################################################################")
        print(f'Inference for {model_name}')
        print("#############################################################################################")

        model, framework = load_model(model_short_name, 'models/saved/'+model_name)

        if model_short_name == 'bert':
            bert(data_size)
        elif model_short_name == 't5':
            df = t5(model_name=model_name, model=model, framework=framework, data_size=data_size, df=df)


df.to_csv('inference_results.csv')

# Remove the emissions file
os.remove('emissions.csv')
