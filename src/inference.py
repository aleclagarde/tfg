import pandas as pd
import os
import time
from inference_utils import load_model, add_measurements

from models.inference.t5 import infer_t5


models = ['models/saved/t5-torch-baseline', 'models/saved/t5-torch-pruned', 'models/saved/t5-torch-quantized.pth',
          'models/saved/t5-tf-baseline', 'models/saved/t5-tf-pruned', 'models/saved/t5-tf-quantized']

df = pd.DataFrame(columns=['timestamp', 'project_name', 'run_id', 'duration', 'emissions', 'emissions_rate',
                           'cpu_power', 'gpu_power', 'ram_power', 'cpu_energy', 'gpu_energy', 'ram_energy',
                           'energy_consumed', 'country_name', 'country_iso_code', 'region', 'cloud_provider',
                           'cloud_region', 'os', 'python_version', 'cpu_count', 'cpu_model', 'gpu_count', 'gpu_model',
                           'longitude', 'latitude', 'ram_total_size', 'tracking_mode', 'on_cloud', 'domain', 'model',
                           'data_number'])

for model_name in models:
    print("#############################################################################################")
    print(f'Inference for {model_name}')
    print("#############################################################################################")

    model, framework = load_model(model_name)

    if 'quantized' in model_name:
        quantized = True
    else:
        quantized = False

    with open("data/sentences.txt") as my_file:
        sentences_to_post = my_file.read().splitlines()
    with open("data/target_sentences.txt") as my_file:
        target_sentences = my_file.read().splitlines()

    i = 0
    for sentence in sentences_to_post:
        # Translate text
        start_time = time.time()
        output = infer_t5(model, framework, 'fr : ' + sentence, quantized)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("#############################################################################################")
        print(f'Translated_text : {output} Time taken: {elapsed_time} seconds')
        df = add_measurements(df, number_of_measurements=1, model_name=model_name, data_number=i + 1)
        i = i + 1

df.to_csv('inference_results.csv')

# Remove the emissions file
os.remove('emissions.csv')
