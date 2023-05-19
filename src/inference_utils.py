"""
Inference utils
==================

.. module:: inference_utils
   :platform: Linux
   :synopsis: Inference auxiliary functions.

.. module_author:: Alec Lagarde Teixidó <aleclagarde@gmail.com>

This script contains auxiliary functions for the inference part of the project...

.. autosummary::
   :toctree: generated/

   add_measurements
   load_model
"""

import pandas as pd
import torch
from datasets import load_dataset
import itertools
import os
import shutil
from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForImageClassification
import language_tool_python
from models.get_model_objects import get_model_objects
import nltk
nltk.download('punkt')


def language_model_score(text: str):
    tool = language_tool_python.LanguageTool('en-US')  # Load the language model
    matches = tool.check(text)  # Check the text for grammar and syntax errors
    score = 1 - (len(matches) / len(text.split()))  # Calculate the language model score
    return score


def download_datasets(data_size: int):
    text_dataset = load_dataset('bookcorpus', split='train', streaming=True)
    text_dataset = itertools.islice(text_dataset, data_size)
    with open('../data/text_dataset.txt', 'w') as file:
        for data_item in text_dataset:
            # Convert the data item to a string representation
            data_str = str(data_item)

            # Write the data item to the file
            file.write(data_str)
            file.write('\n')

    # Need to login to huggingface (huggingface-cli login)
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

    code_dataset = load_dataset('code_search_net', split='test', streaming=True)
    code_dataset = itertools.islice(code_dataset, data_size)
    with open('../data/code_dataset.txt', 'w') as file:
        for data_item in code_dataset:
            # Convert the data item to a string representation
            data_str = str(data_item)

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

    new_measurements['domain'] = domain
    new_measurements['model'] = model
    new_measurements['framework'] = framework
    new_measurements['version'] = version
    new_measurements['iteration'] = [x for x in range(1, number_of_measurements+1)]
    new_measurements['correctness'] = correctness
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
