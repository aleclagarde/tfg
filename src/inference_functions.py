import torch
import tensorflow as tf
from codecarbon import track_emissions
from transformers import pipeline
import time
import os
from PIL import Image
from inference_utils import load_model, language_model_score, pep8_score
import ast


@track_emissions
def inference(model_name: str, model_short_name: str):
    print("#############################################################################################")
    print(f'Inference for {model_name}')
    print("#############################################################################################")

    model, tokenizer, framework = load_model(model_short_name, 'models/saved/' + model_name)

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
    return sum(correctness) / len(correctness)


def text_generation(model_name: str, model, tokenizer, framework: str):
    i = 0
    lms = []
    with open('../data/text_dataset.txt', 'r') as dataset:
        for data in dataset:
            # Safely evaluate each line as a dictionary object
            data = ast.literal_eval(data)
            start_time = time.time()
            output = infer_text_generation(text=data['text'], model=model, tokenizer=tokenizer, framework=framework,
                                           quantized='quantized' in model_name)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("#############################################################################################")
            print(f'Iteration: {i + 1} Output: {output} Time taken: {elapsed_time} s')
            print("#############################################################################################")
            score = language_model_score(output)
            lms.append(score)
            i = i + 1
    return lms


def image_classification(model_name: str, model, processor, framework: str, labels: dict):
    i = 0
    accuracy = []
    directory_path = '../data/image_dataset/'
    # Read the mapping file to get the image file names and their labels
    mapping_file_path = os.path.join(directory_path, 'mapping.txt')
    with open(mapping_file_path, 'r') as mapping_file:
        for line in mapping_file:
            image_file_name, label = line.strip().split('\t')
            image_path = os.path.join(directory_path, image_file_name)
            # Read the image file
            image = Image.open(image_path)
            start_time = time.time()
            output = infer_image_classification(image=image, model=model, processor=processor, framework=framework,
                                                quantized='quantized' in model_name)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("#############################################################################################")
            print(f'Iteration: {i+1} Output: {output} Target: {labels[int(label)]} Time taken: {elapsed_time} s')
            print("#############################################################################################")
            accuracy.append(output == labels[int(label)])
            i = i + 1
    return accuracy


def code_generation(model_name: str, model, tokenizer, framework: str):
    i = 0
    code_quality = []
    with open('../data/code_dataset.txt', 'r') as dataset:
        for data in dataset:
            start_time = time.time()
            output = infer_text_generation(text=data, model=model, tokenizer=tokenizer, framework=framework,
                                           quantized='quantized' in model_name)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("#############################################################################################")
            print(f'Iteration: {i + 1} Output: {output} Time taken: {elapsed_time} s')
            print("#############################################################################################")
            pep_score = pep8_score(output)
            lms = language_model_score(output)
            score = 0.5*pep_score + 0.5*lms
            print(pep_score, lms, score)
            code_quality.append(score)
            i = i + 1
    return code_quality


def infer_text_generation(text: str, model, tokenizer, framework: str, quantized: bool, length: int = 50) -> str:
    if framework == 'pt':
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

        output = generator(text, max_length=length)

        return output[0]["generated_text"]
    else:
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


def infer_image_classification(image, model, processor, framework, quantized):
    inputs = processor(image, return_tensors=framework)

    if framework == 'pt':
        with torch.no_grad():
            logits = model(**inputs).logits

        # model predicts one of the 1000 ImageNet classes
        predicted_label = logits.argmax(-1).item()
        return model.config.id2label[predicted_label]
    else:
        if quantized:
            classifier = pipeline("image-classification", model=model, image_processor=processor)
            return classifier(image)[0]['label']
        else:
            # Run inference on the input image
            logits = model(inputs)[0]

            # Model predicts one of the 1000 ImageNet classes
            predicted_label = tf.argmax(logits, axis=-1).numpy()[0]
            return model.config.id2label[predicted_label]
