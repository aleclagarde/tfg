import torch
import tensorflow as tf
import pandas as pd
import pickle
import time
import os
from transformers import T5ForConditionalGeneration, TFT5ForConditionalGeneration, T5Config

from models.inference.t5 import infer_t5
from models.optimize_utils import add_measurements


models = ['models/saved/t5-torch-baseline', 'models/saved/t5-torch-pruned', 'models/saved/t5-tf-baseline',
          'models/saved/t5-tf-pruned', 'models/saved/t5-torch-quantized.pth', 'models/saved/t5-tf-quantized.pkl']

df = pd.DataFrame(columns=['timestamp', 'project_name', 'run_id', 'duration', 'emissions', 'emissions_rate',
                           'cpu_power', 'gpu_power', 'ram_power', 'cpu_energy', 'gpu_energy', 'ram_energy',
                           'energy_consumed', 'country_name', 'country_iso_code', 'region', 'cloud_provider',
                           'cloud_region', 'os', 'python_version', 'cpu_count', 'cpu_model', 'gpu_count', 'gpu_model',
                           'longitude', 'latitude', 'ram_total_size', 'tracking_mode', 'on_cloud', 'information'])
# Remove the emissions file
os.remove('emissions.csv')


def load_model(model_name):
    if 'torch' in model_name:
        if 'quantized' in model_name:
            # Load the configuration of T5ForConditionalGeneration
            config = T5Config.from_pretrained('t5-base')

            # Create an instance of T5ForConditionalGeneration using the configuration object
            model = T5ForConditionalGeneration(config)

            # Load the quantized state dictionary
            quantized_state_dict = torch.load(model_name, map_location='cpu')

            # Load the quantized state dictionary into the model with strict=False
            model.load_state_dict(quantized_state_dict, strict=False)
        else:
            model = T5ForConditionalGeneration.from_pretrained(model_name)
        framework = 'pt'
    else:
        if 'quantized' in model_name:
            with open(model_name, "rb") as f:
                state_dict = pickle.load(f)
            model = tf.lite.Interpreter(model_content=state_dict)
            model.allocate_tensors()
        else:
            model = TFT5ForConditionalGeneration.from_pretrained(model_name)
        framework = 'tf'

    return model, framework


def infer(model_name, df):
    model, framework = load_model(model_name)

    with open("data/sentences.txt") as my_file:
        sentences_to_post = my_file.read().splitlines()
    with open("data/target_sentences.txt") as my_file:
        target_sentences = my_file.read().splitlines()

    i = 0
    for sentence in sentences_to_post:
        # Translate text
        start_time = time.time()
        output = infer_t5(model, framework, sentence)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("#############################################################################################")
        print(f'Translated_text : {output}. Time taken: {elapsed_time} seconds')
        df = add_measurements(df, 1, f'{model_name}-sentence-{i}')
        i = i + 1
    return df


for mod in models:
    print("#############################################################################################")
    print(f'Inference for {mod}')
    print("#############################################################################################")
    df = infer(mod, df)

df.to_csv('inference_results.csv')
