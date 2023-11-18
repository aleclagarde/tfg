"""
Inference utils
===============

.. module:: inference_utils
   :platform: Linux
   :synopsis: Inference auxiliary functions.

This script contains auxiliary functions for the inference part of the project that compute the correctness, downloads
the datasets, adds measurements to the results dataframe and loads the models.

.. autosummary::
   :toctree: generated/

   language_model_score
   pep8_score
   download_datasets
   add_measurements
   load_model
"""

# Dataframe transformation
import pandas as pd

# Model loading
from models.get_model_objects import get_model_objects
import torch
from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForImageClassification

# Datasets
from datasets import load_dataset
import itertools

# Score
import requests
import pep8

# File management
import os
import shutil
import tempfile


def language_model_score(text: str) -> float:
    """
    Computes the Language Model score for a string of text.

    :param text: Text to be used.
    :return: Language Model Score.
    """
    # Define url and parameters
    api_url = 'https://languagetool.org/api/v2/check'
    params = {
        'text': text,
        'language': 'en-US',
        'disabledRules': 'WHITESPACE_RULE'  # Exclude whitespace rule from analysis
    }

    try:
        # Get the APIs output
        response = requests.get(api_url, params=params)
    except:
        # Exception for possible connection error
        return language_model_score(text)
    try:
        # Process the output and compute score
        data = response.json()
        matches = data['matches']
        score = 1 - (len(matches) / len(text.split()))
    except:
        # Exception for possible error in the computation
        score = 0

    return score


def pep8_score(code: str) -> float:
    """
    Computes the PEP8 score for a code string.

    :param code: Code string to use.
    :return: PEP8 score.
    """
    # Create a temporary file with the code string
    with tempfile.NamedTemporaryFile(suffix='.py') as temp_file:
        temp_file.write(code.encode())
        temp_file.flush()

        # Create a StyleGuide object
        pep8style = pep8.StyleGuide()

        # Check the temporary file for PEP 8 compliance
        report = pep8style.check_files([temp_file.name])

        # Calculate the PEP 8 score
        total_errors = report.get_count()
        max_score = len(code.splitlines())
        score = max(0, max_score - total_errors) / max_score

    return score


def download_datasets(data_size: int):
    """
    Downloads the three datasets and saves them in disk.

    :param data_size: Number of data points to download.
    """
    # TEXT DATASET
    text_dataset = load_dataset('bookcorpus', split='train', streaming=True)
    text_dataset = itertools.islice(text_dataset, data_size)
    with open('../data/text_dataset.txt', 'w') as file:
        for data_item in text_dataset:
            # Convert the data item to a string representation
            data_str = str(data_item)

            # Write the data item to the file
            file.write(data_str)
            file.write('\n')

    # IMAGE DATASET
    # Need to log in to huggingface (huggingface-cli login)
    image_dataset = load_dataset('imagenet-1k', split='validation', streaming=True)
    image_dataset = itertools.islice(image_dataset, data_size)
    directory_path = '../data/image_dataset/'
    # Remove the directory if it exists
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)

    # Create the directory
    os.makedirs(directory_path)

    # Create a file to store the mapping of image file names to labels
    mapping_file_path = os.path.join(directory_path, 'mapping.txt')
    mapping_file = open(mapping_file_path, 'w')

    for i, data_item in enumerate(image_dataset):
        # Get the image and label from the data_item
        image = data_item['image']
        label = data_item['label']

        # Save the image to the directory
        image_path = os.path.join(directory_path, f'image_{i}.jpg')
        image.save(image_path)

        # Write the mapping of image file name to label in the mapping file
        mapping_file.write(f'image_{i}.jpg\t{label}\n')
    mapping_file.close()

    # CODE DATASET
    code_dataset = load_dataset('code_search_net', 'python', split='test', streaming=True)
    code_dataset = itertools.islice(code_dataset, data_size)
    with open('../data/code_dataset.txt', 'w') as file:
        for data_item in code_dataset:
            # Extract only the function header
            funct = data_item['whole_func_string'].split(':')[0] + ':'
            # Convert the data item to a string representation
            data_str = str(funct)

            # Write the data item to the file
            file.write(data_str)
            file.write('\n')


def add_measurements(dataframe: pd.DataFrame, number_of_measurements: int, model_name: str,
                     correctness) -> pd.DataFrame:
    """
    This function takes a Pandas DataFrame and adds *number_of_measurements* rows with metrics that we want to analyse.

    :param dataframe: Pandas DataFrame to add rows.
    :param number_of_measurements: Number of measurements (rows) to add.
    :param model_name: String to store in the new rows' model column. It is also used to get the domain.
    :param correctness: Whether the output is correct or not.
    :return: Pandas DataFrame with the added measurements.
    """
    # Get new measurements generated by CodeCarbon
    new_measurements = pd.read_csv(filepath_or_buffer='emissions.csv').tail(n=number_of_measurements)

    model_short_name = model_name.split('-')[0]
    if model_short_name in ['gpt2', 'opt', 'gptj']:
        domain = 'NLP'
    elif model_short_name in ['resnet', 'vit', 'regnet']:
        domain = 'Computer Vision'
    else:
        domain = 'Code'

    model = model_name.split('-')[0]
    framework = model_name.split('-')[1]
    version = model_name.split('-')[2]

    # Assign the values to the new measurements
    new_measurements['domain'] = domain
    new_measurements['model'] = model
    new_measurements['framework'] = framework
    new_measurements['version'] = version
    new_measurements['iteration'] = [x for x in range(1, number_of_measurements+1)]
    new_measurements['correctness'] = correctness
    # Add the new measurements to the dataframe
    return pd.concat([dataframe, new_measurements], axis=0)


def load_model(model_short_name: str, path: str):
    """
    This function loads the corresponding model and returns it along with the framework identifier.

    :param model_short_name: Name of the model to load.
    :param path: Path of the model to load.
    :return: Loaded model, tokenizer and framework identifier.
    """
    objects = get_model_objects(model_name=model_short_name)

    constructor_torch = objects['constructor_torch']
    constructor_tf = objects['constructor_tf']
    tokenizer = objects['constructor_tokenizer'].from_pretrained(objects['full_name'])

    if 'torch' in path:
        framework = 'pt'
        if 'quantized' in path:
            model = torch.load(path)
        else:
            model = constructor_torch.from_pretrained(path)
    else:
        framework = 'tf'
        if 'quantized' in path:
            if model_short_name in ['resnet', 'vit', 'regnet']:
                model = ORTModelForImageClassification.from_pretrained(path)
            else:
                model = ORTModelForCausalLM.from_pretrained(path)
        else:
            model = constructor_tf.from_pretrained(path)

    return model, tokenizer, framework
