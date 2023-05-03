import numpy as np
import torch
import tensorflow as tf
from datasets import load_dataset
from codecarbon import track_emissions
from transformers import pipeline, GPT2Tokenizer, T5Tokenizer
import time
import itertools
from inference_utils import load_model


@track_emissions
def inference(model_name: str, model_short_name: str, number_of_measurements: int):
    print("#############################################################################################")
    print(f'Inference for {model_name}')
    print("#############################################################################################")

    model, framework = load_model(model_short_name, 'models/saved/' + model_name)

    correct = []
    if model_short_name == 'gpt2':
        correct = gpt2(model_name=model_name, model=model, framework=framework,
                       number_of_measurements=number_of_measurements)
    elif model_short_name == 't5':
        correct = t5(model_name=model_name, model=model, framework=framework,
                     number_of_measurements=number_of_measurements)
    return correct


def infer_gpt2(text: str, model, tokenizer, framework: str, quantized: bool, length: int = 30,
               temperature: float = 0.7) -> str:
    if framework == 'pt':
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

        output = generator(text, max_length=length, do_sample=True, temperature=temperature)

        return output[0]["generated_text"]
    else:
        if quantized:
            input_ids = tokenizer.encode(text, add_special_tokens=True)
            input_ids = np.array(input_ids)[np.newaxis, :]
            # Perform inference on the preprocessed input
            output = model.predict(input_ids)
            output_ids = np.argmax(output, axis=-1)
            output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return text + output_str
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


def gpt2(model_name: str, model, framework: str, number_of_measurements: int):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # gpt2_dataset = load_dataset("openwebtext", split='test', streaming=True)
    # gpt2_dataset = itertools.islice(gpt2_dataset, data_size)
    gpt2_dataset = [{'text': "April is the fourth month"}]

    i = 0
    correct = []
    for data in gpt2_dataset:
        text = data['text']
        start_time = time.time()
        output = infer_gpt2(text=text, model=model, tokenizer=tokenizer, framework=framework,
                            quantized='quantized' in model_name)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("#############################################################################################")
        print(f'({i + 1}/{number_of_measurements}) Output : {output} Time taken: {elapsed_time} seconds')
        correct.append(False)
        i = i + 1
    return correct


def infer_t5(text: str, model, tokenizer, framework: str, quantized: bool) -> str:
    if quantized:
        if framework == 'pt':
            # Encode the input text using the T5 tokenizer
            inputs = tokenizer.encode(text, return_tensors='pt')

            # Extract the input_ids and attention_mask tensors
            input_ids = inputs[:, :512]
            attention_mask = torch.ones_like(input_ids)

            # Generate output text from the quantized model
            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=100,
                    num_return_sequences=1,
                )

            # Decode the generated output text using the T5 tokenizer
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)

            return output_text
        else:

            # Encode the input text using the tokenizer
            input_ids = tokenizer.encode(text, return_tensors='tf')

            # Generate translations using the T5 model
            outputs = model(inputs=input_ids, training=False)

            # Decode the output text using the tokenizer
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            return output_text
    else:
        input_ids = tokenizer.encode(text, return_tensors=framework)
        outputs = model.generate(input_ids)

        # Decode and return translated text
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text.strip()


def t5(model_name: str, model, framework: str, number_of_measurements: int):
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    t5_dataset = load_dataset("opus100", "de-en", split="test", streaming=True)
    t5_dataset = itertools.islice(t5_dataset, number_of_measurements)

    i = 0
    correct = []
    for data in t5_dataset:
        sentence = data['translation']['en']
        target = data['translation']['de']
        # Translate text
        prefix = "translate English to German: "
        start_time = time.time()
        output = infer_t5(text=prefix+sentence, model=model, tokenizer=tokenizer, framework=framework,
                          quantized='quantized' in model_name)
        end_time = time.time()
        elapsed_time = end_time - start_time
        output = output.replace('"', '')
        target = target.replace('"', '')
        print("#############################################################################################")
        print(f'({i + 1}/{number_of_measurements}) Translated_text : {output} Target: {target} '
              f'Time taken: {elapsed_time} seconds')

        correct.append(output == target)
        i = i + 1
    return correct
