import pandas as pd
import torch
import tensorflow as tf
from datasets import load_dataset
from transformers import T5Tokenizer
from codecarbon import track_emissions
from inference_utils import add_measurements
import time

tokenizer = T5Tokenizer.from_pretrained("t5-base")


def bert(data_size):
    bert_dataset = load_dataset("openwebtext", split='test').select(range(data_size))
    print(bert_dataset)


@track_emissions
def infer_t5(model, framework, text, quantized):
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
            input_ids = input_ids.numpy()

            # Perform inference on the Sequential model
            output_ids = model.predict(input_ids)

            # Convert output_ids to a tf.Tensor
            output_ids = tf.constant(output_ids)[0]

            # Decode the output text using the tokenizer
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

            return output_text
    else:
        input_ids = tokenizer.encode(text, return_tensors=framework)
        outputs = model.generate(input_ids)

        # Decode and return translated text
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text.strip()


def t5(model_name: str, model, framework: str, data_size: int, df: pd.DataFrame):
    t5_dataset = load_dataset("wmt16", "de-en", split="test").select(range(data_size))

    i = 0
    for data in t5_dataset:
        sentence = data['translation']['en']
        target = data['translation']['de']
        # Translate text
        start_time = time.time()
        prefix = "translate English to German: "
        output = infer_t5(model=model, framework=framework, text=prefix+sentence, quantized='quantized' in model_name)
        end_time = time.time()
        elapsed_time = end_time - start_time
        output = output.replace('"', '')
        target = target.replace('"', '')
        print("#############################################################################################")
        print(f'({i + 1}/{data_size})Translated_text : {output} Target: {target} Time taken: {elapsed_time} seconds')

        correct = output == target
        df = add_measurements(df, number_of_measurements=1, model_name=model_name, data_number=i + 1, correct=correct)
        i = i + 1
