import numpy as np
import torch
import tensorflow as tf
from datasets import load_dataset
from codecarbon import track_emissions
from transformers import pipeline
import time
import itertools
from inference_utils import load_model, self_bleu_score
import ast


@track_emissions
def inference(model_name: str, model_short_name: str, data_size: int):
    print("#############################################################################################")
    print(f'Inference for {model_name}')
    print("#############################################################################################")

    model, tokenizer, framework = load_model(model_short_name, 'models/saved/' + model_name)

    if model_short_name in ['gpt2', 'opt', 'xlnet']:
        correctness = text_generation(model_name=model_name, model=model, tokenizer=tokenizer, framework=framework,
                                      data_size=data_size)
    elif model_short_name in ['resnet', 'vit', 'convnext']:
        # Read the labels file
        with open('imagenet1000_idx_to_labels.txt', 'r') as f:
            my_dict_str = f.read()

        imagenet_labels = ast.literal_eval(my_dict_str)
        correctness = image_classification(model_name=model_name, model=model, processor=tokenizer, framework=framework,
                                           data_size=data_size, labels=imagenet_labels)
    else:
        correctness = code_generation(model_name=model_name, model=model, tokenizer=tokenizer, framework=framework,
                                      data_size=data_size)
    return sum(correctness) / len(correctness)


def text_generation(model_name: str, model, tokenizer, framework: str, data_size: int):
    dataset = load_dataset('ptb_text_only', split='test', streaming=True)
    dataset = itertools.islice(dataset, data_size)
    # dataset = [{'sentence': "April is the fourth month"}]

    i = 0
    bleu = []
    for data in dataset:
        start_time = time.time()
        output = infer_text_generation(text=data['sentence'], model=model, tokenizer=tokenizer, framework=framework,
                                       quantized='quantized' in model_name)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("#############################################################################################")
        print(f'({i + 1}/{data_size}) Output: {output} Time taken: {elapsed_time} seconds')
        print("#############################################################################################")
        bleu.append(self_bleu_score(output))
        i = i + 1
    return bleu


def image_classification(model_name: str, model, processor, framework: str, data_size: int, labels: dict):
    # Need to login to huggingface (huggingface-cli login)
    dataset = load_dataset('imagenet-1k', split='validation', streaming=True)
    dataset = itertools.islice(dataset, data_size)

    i = 0
    accuracy = []
    for data in dataset:
        print(data)
        start_time = time.time()
        output = infer_image_classification(image=data['image'], model=model, processor=processor, framework=framework,
                                            quantized='quantized' in model_name)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("#############################################################################################")
        print(f'({i+1}/{data_size}) Output: {output} Target: {labels[data["label"]]} Time taken: {elapsed_time} s')
        print("#############################################################################################")
        accuracy.append(output == labels[data['label']])
        i = i + 1
    return accuracy


def code_generation(model_name: str, model, tokenizer, framework: str, data_size: int):
    dataset = [{"text": "def hello_world():"}]
    i = 0
    bleu = []
    for data in dataset:
        text = data['text']
        start_time = time.time()
        output = infer_text_generation(text=text, model=model, tokenizer=tokenizer, framework=framework,
                                       quantized='quantized' in model_name)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("#############################################################################################")
        print(f'({i + 1}/{data_size}) Output: {output} Time taken: {elapsed_time} seconds')
        print("#############################################################################################")
        bleu.append(self_bleu_score(output))
        i = i + 1
    return bleu


def infer_text_generation(text: str, model, tokenizer, framework: str, quantized: bool, length: int = 30,
                          temperature: float = 0.7) -> str:
    if framework == 'pt':
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

        output = generator(text, max_length=length, do_sample=True, temperature=temperature)

        return output[0]["generated_text"]
    else:
        if quantized:
            onnx_clx = pipeline("text-generation", model=model, tokenizer=tokenizer)
            return onnx_clx(text)[0]['generated_text']
        else:
            prompt_tokens = tokenizer.encode(text)
            input_seq = tf.constant([prompt_tokens])

            output_seq = input_seq
            for i in range(length):
                logits = model({'input_ids': output_seq})['logits'][:, -1, :]
                logits /= temperature
                probs = tf.nn.softmax(logits).numpy()[0]
                selected_token = np.random.choice(tokenizer.vocab_size, p=probs)
                output_seq = tf.concat([output_seq, [[selected_token]]], axis=-1)

            output_text = tokenizer.decode(output_seq.numpy()[0])

            return output_text.strip()


def infer_image_classification(image, model, processor, framework, quantized):
    inputs = processor(image, return_tensors=framework)

    if framework == 'pt':
        with torch.no_grad():
            logits = model(**inputs).logits

        # model predicts one of the 1000 ImageNet classes
        predicted_label = logits.argmax(-1).item()
        return model.config.id2label[predicted_label]
    else:
        # Run inference on the input image
        logits = model(inputs)[0]

        # Model predicts one of the 1000 ImageNet classes
        predicted_label = tf.argmax(logits, axis=-1).numpy()[0]
        return model.config.id2label[predicted_label]
