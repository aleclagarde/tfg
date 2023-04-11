import torch
import tensorflow as tf
import pickle
import time
from transformers import T5ForConditionalGeneration, TFT5ForConditionalGeneration, T5Config

from models.inference.t5 import infer_t5


models = ['models/saved/t5-torch-baseline', 'models/saved/t5-torch-pruned-0.2', 'models/saved/t5-tf-baseline',
          'models/saved/t5-tf-pruned-0.2', 'models/saved/t5-torch-quantized.pth', 'models/saved/t5-tf-quantized.pkl']


def load_model(model_name):
    if 'torch' in model_name:
        if 'quantized' in model_name:
            config = T5Config.from_pretrained('t5-base')

            # Create an instance of T5ForConditionalGeneration using the configuration object
            model = T5ForConditionalGeneration(config)

            # Load the model weights from a checkpoint
            state_dict = torch.load(model_name)
            model.load_state_dict(state_dict)
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


def infer(model_name):
    model, framework = load_model(model_name)

    with open("data/sentences.txt") as my_file:
        sentences_to_post = my_file.read().splitlines()
    with open("data/target_sentences.txt") as my_file:
        target_sentences = my_file.read().splitlines()

    for sentence in sentences_to_post:
        # Translate text
        start_time = time.time()
        output = infer_t5(model, framework, sentence)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Translated_text : {output}. Time taken: {elapsed_time} seconds')
        return output


for mod in models:
    infer(mod)
