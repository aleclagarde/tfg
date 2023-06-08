"""
Inference functions
===================

.. module:: inference_fuctions
   :platform: Linux
   :synopsis: Functions that performs the inference process.

.. module_author:: Alec Lagarde Teixidó <aleclagarde@gmail.com>

This script contains the functions that perform inference and manage the loading of the models, the datasets and the
computation of the correctness.

.. autosummary::
   :toctree: generated/

   inference
   text_generation
   image_classification
   code_generation
   infer_text_generation
   infer_image_classification
"""

# Frameworks
import torch
import tensorflow as tf

# Emissions and energy consumption
from codecarbon import track_emissions

# Inference pipeline
from transformers import pipeline

# Data transformation
import ast

# Time calculation
import time

# File modification
import os

# Image loading
from PIL import Image

# Auxiliary functions
from inference_utils import load_model, language_model_score, pep8_score


@track_emissions
def inference(model_name: str, model_short_name: str) -> float:
    """
    Divides the models by domains, calls the respective function and returns the average correctness.

    :param model_name: Model name.
    :param model_short_name: Short model name (eg. 'gpt2').
    :return: Average correctness.
    """
    print("#############################################################################################")
    print(f'Inference for {model_name}')
    print("#############################################################################################")

    # Model loading
    model, tokenizer, framework = load_model(model_short_name, 'models/saved/' + model_name)

    # Call inference functions depending on domain
    if model_short_name in ['gpt2', 'opt', 'gptj']:
        correctness = text_generation(model_name=model_name, model=model, tokenizer=tokenizer, framework=framework)
    elif model_short_name in ['resnet', 'vit', 'regnet']:
        # Read the labels file
        with open('../data/imagenet1000_idx_to_labels.txt', 'r') as f:
            labels = f.read()

        imagenet_labels = ast.literal_eval(labels)
        correctness = image_classification(model_name=model_name, model=model, processor=tokenizer, framework=framework,
                                           labels=imagenet_labels)
    else:
        correctness = code_generation(model_name=model_name, model=model, tokenizer=tokenizer, framework=framework)
    print(f'Correctness {model_name}: {correctness}')
    # Return the average correctness
    return sum(correctness) / len(correctness)


def text_generation(model_name: str, model, tokenizer, framework: str) -> list[float]:
    """
    Infers the text generation models and returns the list of correctness.

    :param model_name: Model name.
    :param model: Model to infer.
    :param tokenizer: Model tokenizer.
    :param framework: ML framework.
    :return: List of correctness for each data point.
    """
    i = 0
    score = []
    with open('../data/text_dataset.txt', 'r') as dataset:
        # Infer each data point
        for data in dataset:
            # Safely evaluate each line as a dictionary object
            data = ast.literal_eval(data)
            start_time = time.time()
            # Call inference function
            output = infer_text_generation(text=data['text'], model=model, tokenizer=tokenizer, framework=framework,
                                           quantized='quantized' in model_name)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("#############################################################################################")
            print(f'Iteration: {i + 1} Output: {output} Time taken: {elapsed_time} s')
            print("#############################################################################################")
            # Compute score
            lms = language_model_score(output)
            score.append(lms)
            i = i + 1
    return score


def image_classification(model_name: str, model, processor, framework: str, labels: dict) -> list[float]:
    """
    Infers the image classification models and returns the correctness.

    :param model_name: Model name.
    :param model: Model to infer.
    :param processor: Model processor.
    :param framework: ML framework.
    :param labels: ImageNet labels.
    :return: List of correctness for the data points.
    """
    i = 0
    accuracy = []
    directory_path = '../data/image_dataset/'
    # Read the mapping file to get the image file names and their labels
    mapping_file_path = os.path.join(directory_path, 'mapping.txt')
    with open(mapping_file_path, 'r') as mapping_file:
        # Infer each data point
        for line in mapping_file:
            image_file_name, label = line.strip().split('\t')
            image_path = os.path.join(directory_path, image_file_name)
            # Read the image file
            image = Image.open(image_path)
            start_time = time.time()
            # Call inference function
            output = infer_image_classification(image=image, model=model, processor=processor, framework=framework,
                                                quantized='quantized' in model_name)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("#############################################################################################")
            print(f'Iteration: {i+1} Output: {output} Target: {labels[int(label)]} Time taken: {elapsed_time} s')
            print("#############################################################################################")
            # Compute accuracy
            accuracy.append(output == labels[int(label)])
            i = i + 1
    return accuracy


def code_generation(model_name: str, model, tokenizer, framework: str) -> list[float]:
    """
    Infers the code generation models and returns the correctness.

    :param model_name: Model name.
    :param model: Model to infer.
    :param tokenizer: Model tokenizer.
    :param framework: ML framework.
    :return: List of correctness for the data points.
    """
    i = 0
    code_quality = []
    with open('../data/code_dataset.txt', 'r') as dataset:
        # Infer each data point
        for data in dataset:
            start_time = time.time()
            # Call inference function
            output = infer_text_generation(text=data, model=model, tokenizer=tokenizer, framework=framework,
                                           quantized='quantized' in model_name)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("#############################################################################################")
            print(f'Iteration: {i + 1} Output: {output} Time taken: {elapsed_time} s')
            print("#############################################################################################")
            # Compute score
            pep_score = pep8_score(output)
            lms = language_model_score(output)
            score = 0.5*pep_score + 0.5*lms
            print(pep_score, lms, score)
            code_quality.append(score)
            i = i + 1
    return code_quality


def infer_text_generation(text: str, model, tokenizer, framework: str, quantized: bool, length: int = 50) -> str:
    """
    Infers a string of text for a given model and returns the generated text.

    :param text: Text to infer.
    :param model: Model to infer.
    :param tokenizer: Model tokenizer.
    :param framework: ML framework.
    :param quantized: Whether it is the quantized version of the model.
    :param length: Length of the generated text.
    :return: Generated text.
    """
    if framework == 'pt':
        # PyTorch inference
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

        output = generator(text, max_length=length)

        return output[0]["generated_text"]
    else:
        # TensorFlow inference
        if quantized:
            generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
            return generator(text)[0]['generated_text']
        else:
            inputs = tokenizer(text, return_tensors="tf")
            outputs = model(inputs)
            logits = outputs.logits

            # Get the index of the highest probability token
            predicted_token_indexes = tf.argmax(logits, axis=-1)[0]

            # Decode the generated text
            output_text = tokenizer.decode(predicted_token_indexes)

            return text + output_text


def infer_image_classification(image: Image.Image, model, processor, framework: str, quantized: bool) -> str:
    """
    Infers an image for a given model and returns the output label.

    :param image: Image to infer.
    :param model: Model to infer.
    :param processor: Model processor.
    :param framework: ML framework.
    :param quantized: Whether it is the quantized version of the model.
    :return: Image label.
    """
    inputs = processor(image, return_tensors=framework)

    if framework == 'pt':
        # PyTorch inference
        with torch.no_grad():
            logits = model(**inputs).logits

        # model predicts one of the 1000 ImageNet classes
        predicted_label = logits.argmax(-1).item()
        return model.config.id2label[predicted_label]
    else:
        # TensorFlow inference
        if quantized:
            classifier = pipeline("image-classification", model=model, image_processor=processor)
            return classifier(image)[0]['label']
        else:
            # Run inference on the input image
            logits = model(inputs)[0]

            # Model predicts one of the 1000 ImageNet classes
            predicted_label = tf.argmax(logits, axis=-1).numpy()[0]
            return model.config.id2label[predicted_label]
