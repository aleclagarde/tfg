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

from models.get_model_objects import get_model_objects


def add_measurements(dataframe: pd.DataFrame, number_of_measurements: int, model_name: str,
                     data_number: int) -> pd.DataFrame:
    """
    This function takes a Pandas DataFrame and adds *number_of_measurements* rows with the string *information* in the
    'information' column, along with the measurement number in the 'iteration' column. This is used to add emissions
    measurements to the results table.

    :param dataframe: Pandas DataFrame to add rows.
    :param number_of_measurements: Number of measurements (rows) to add.
    :param model_name: String to store in the new rows' model column. It is also used to get the domain.
    :param data_number: Integer to store in the new rows' data_number column.
    :return: Pandas DataFrame with the added measurements.
    """
    new_measurements = pd.read_csv(filepath_or_buffer='emissions.csv').tail(n=number_of_measurements)

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


def load_model(model_name: str):
    """
    This function loads the corresponding model and returns it along with the framework identifier.

    :param model_name: Path of the model to load.
    :return: Loaded model and framework identifier.
    """
    constructor_torch = get_model_objects(model_name=model_name)['constructor_torch']
    constructor_tf = get_model_objects(model_name=model_name)['constructor_tf']

    if 'torch' in model_name:
        framework = 'pt'
        if 'quantized' in model_name:
            model = torch.load(model_name)
        else:
            model = constructor_torch.from_pretrained(model_name)
    else:
        framework = 'tf'
        if 'quantized' in model_name:
            model = tf.keras.models.load_model(model_name)
        else:
            model = constructor_tf.from_pretrained(model_name)

    return model, framework
