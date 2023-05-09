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
import tensorflow as tf
import nltk
nltk.download('punkt')

from models.get_model_objects import get_model_objects


def bleu_score(reference_str: str, generated_str: str):
    # Preprocess the reference string
    reference_tokens = nltk.word_tokenize(reference_str)
    # Preprocess the generated string
    generated_tokens = nltk.word_tokenize(generated_str)

    weights = [(1.0/len(reference_tokens)) for _ in range(len(reference_tokens))]
    return nltk.translate.bleu_score.sentence_bleu([reference_tokens], generated_tokens, weights=weights)


def self_bleu_score(generated_str: str, n: int = 4):
    if len(generated_str) < 2:
        return 0
    # Preprocess the generated string
    generated_tokens = nltk.word_tokenize(generated_str)

    bleu_scores = []
    for i in range(len(generated_tokens) - n + 1):
        reference_tokens = generated_tokens[i:i+n]
        weights = [(1.0/len(reference_tokens)) for _ in range(len(reference_tokens))]
        bleu_score = nltk.translate.bleu_score.sentence_bleu([generated_tokens], reference_tokens, weights=weights)
        bleu_scores.append(bleu_score)
    return sum(bleu_scores) / len(bleu_scores)


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
    if model_short_name in ['bert', 'gpt2', 't5']:
        domain = 'NLP'
    elif model_short_name in ['resnet', 'vit', 'convnext']:
        domain = 'Computer Vision'
    else:
        domain = 'Code'

    new_measurements['domain'] = domain
    new_measurements['model'] = model_name
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
            model = tf.keras.models.load_model(path)
        else:
            model = constructor_tf.from_pretrained(path)

    return model, tokenizer, framework
