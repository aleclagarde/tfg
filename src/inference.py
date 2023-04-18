import torch
import tensorflow as tf
import pandas as pd
import time
import os
from transformers import T5ForConditionalGeneration, TFT5ForConditionalGeneration

from models.inference.t5 import infer_t5


models = ['models/saved/t5-torch-baseline', 'models/saved/t5-torch-pruned', 'models/saved/t5-torch-quantized.pth',
          'models/saved/t5-tf-baseline', 'models/saved/t5-tf-pruned', 'models/saved/t5-tf-quantized']

df = pd.DataFrame(columns=['timestamp', 'project_name', 'run_id', 'duration', 'emissions', 'emissions_rate',
                           'cpu_power', 'gpu_power', 'ram_power', 'cpu_energy', 'gpu_energy', 'ram_energy',
                           'energy_consumed', 'country_name', 'country_iso_code', 'region', 'cloud_provider',
                           'cloud_region', 'os', 'python_version', 'cpu_count', 'cpu_model', 'gpu_count', 'gpu_model',
                           'longitude', 'latitude', 'ram_total_size', 'tracking_mode', 'on_cloud', 'domain', 'model',
                           'data_number'])


def add_measurements(dataframe, number_of_measurements, model_name, data_number):
    """
    This function takes a Pandas DataFrame and adds *number_of_measurements* rows with the string *information* in the
    'information' column, along with the measurement number in the 'iteration' column. This is used to add emissions
    measurements to the results table.

    :param dataframe: Pandas DataFrame to add rows.
    :param number_of_measurements: Number of measurements (rows) to add.
    :param model_name: String to store in the new rows' model column. It is also used to get the domain.
    :param data_number: String to store in the new rows' data_number column.
    :return: Pandas DataFrame with the added measurements.
    """
    new_measurements = pd.read_csv('emissions.csv').tail(number_of_measurements)

    if model_name in ['bert', 'gpt2', 't5']:
        domain = 'NLP'
    elif model_name in ['vit', 'clip', 'segformer']:
        domain = 'Computer Vision'
    else:
        domain = 'Code'

    new_measurements['domain'] = domain
    new_measurements['model'] = model_name
    new_measurements['data_number'] = data_number
    return pd.concat([dataframe, new_measurements], axis=0)


def load_model(model_name):
    if 'torch' in model_name:
        framework = 'pt'
        if 'quantized' in model_name:
            model = torch.load(model_name)
        else:
            model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        framework = 'tf'
        if 'quantized' in model_name:
            model = tf.keras.models.load_model(model_name)
        else:
            model = TFT5ForConditionalGeneration.from_pretrained(model_name)

    return model, framework


def infer(model_name, df):
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
        output = infer_t5(model, framework, sentence, quantized)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("#############################################################################################")
        print(f'Translated_text : {output} Time taken: {elapsed_time} seconds')
        df = add_measurements(df, number_of_measurements=1, model_name=model_name, data_number=i+1)
        i = i + 1
    return df


for mod in models:
    print("#############################################################################################")
    print(f'Inference for {mod}')
    print("#############################################################################################")
    df = infer(mod, df)

df.to_csv('inference_results.csv')

# Remove the emissions file
os.remove('emissions.csv')
