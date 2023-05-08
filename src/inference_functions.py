import numpy as np
import torch
import tensorflow as tf
from datasets import load_dataset
from codecarbon import track_emissions
from transformers import pipeline, BertTokenizer, GPT2Tokenizer, T5Tokenizer, AutoTokenizer
import time
import itertools
from inference_utils import load_model, self_bleu_score


@track_emissions
def inference(model_name: str, model_short_name: str, data_size: int):
    print("#############################################################################################")
    print(f'Inference for {model_name}')
    print("#############################################################################################")

    model, framework = load_model(model_short_name, 'models/saved/' + model_name)

    if model_short_name == 'bert':
        bleu = bert(model_name=model_name, model=model, framework=framework, data_size=data_size)
    elif model_short_name == 'gpt2':
        bleu = gpt2(model_name=model_name, model=model, framework=framework, data_size=data_size)
    elif model_short_name == 't5':
        bleu = t5(model_name=model_name, model=model, framework=framework, data_size=data_size)
    else:
        bleu = codegen(model_name=model_name, model=model, framework=framework, data_size=data_size)
    return sum(bleu) / len(bleu)


def infer_bert(text: str, model, tokenizer, framework: str, quantized: bool):
    text = 'April is the [MASK] month'
    if framework == 'pt':
        # Tokenize input string and convert to tensor
        input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)

        # Find the index of the masked token
        mask_token_id = tokenizer.mask_token_id
        masked_index = torch.where(torch.eq(input_ids, mask_token_id))[1][0]

        # Forward pass through the model to get predictions
        outputs = model(input_ids)
        predictions = outputs[0]

        # Get the predicted token index and decode the predicted word
        predicted_index = torch.argmax(predictions[0, masked_index]).item()
        predicted_word = tokenizer.decode([predicted_index])

        # Replace the masked token with the predicted word
        output = text.replace('<mask>', predicted_word)
    else:
        if quantized:
            input_ids = tokenizer.encode(text, add_special_tokens=True)
            input_ids = np.array(input_ids)[np.newaxis, :]
            # Perform inference on the preprocessed input
            output = model.predict(input_ids)
            output_ids = np.argmax(output, axis=-1)
            output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return text.replace('[MASK]', output_str)
        else:
            # Tokenize input string and convert to tensor
            input_ids = tf.constant(tokenizer.encode(text, add_special_tokens=True))[None, :]

            # Forward pass through the model to get predictions
            outputs = model(input_ids)
            predictions = outputs.logits

            # Find the index of the masked token and get the predicted word
            mask_token_id = tokenizer.mask_token_id
            masked_index = tf.where(tf.equal(input_ids, mask_token_id))[0, 1]
            predicted_index = tf.argmax(predictions[0, masked_index]).numpy()
            predicted_word = tokenizer.decode([predicted_index])

            # Replace the masked token with the predicted word
            output = text.replace('<mask>', predicted_word)
    return output


def bert(model_name: str, model, framework: str, data_size: int):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # bert_dataset = load_dataset('rcds/wikipedia-persons-masked', split='test', streaming=True)
    # bert_dataset = itertools.islice(bert_dataset, data_size)

    i = 0
    bleu = []
    for data in bert_dataset:
        print(data['text'])
        start_time = time.time()
        output = infer_bert(text=data['text'], model=model, tokenizer=tokenizer, framework=framework,
                            quantized='quantized' in model_name)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("#############################################################################################")
        print(f'({i + 1}/{data_size}) Output : {output} Time taken: {elapsed_time} seconds')
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
            input_ids = tokenizer.encode(text, add_special_tokens=True)
            input_ids = np.array(input_ids)[np.newaxis, :]
            # Perform inference on the preprocessed input
            output = model.predict(input_ids)
            output_ids = np.argmax(output, axis=-1)
            output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return output_str
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


def gpt2(model_name: str, model, framework: str, data_size: int):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # gpt2_dataset = load_dataset("openwebtext", split='test', streaming=True)
    # gpt2_dataset = itertools.islice(gpt2_dataset, data_size)
    gpt2_dataset = [{'text': "April is the fourth month"}]

    i = 0
    bleu = []
    for data in gpt2_dataset:
        text = data['text']
        start_time = time.time()
        output = infer_text_generation(text=text, model=model, tokenizer=tokenizer, framework=framework,
                                       quantized='quantized' in model_name)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("#############################################################################################")
        print(f'({i + 1}/{data_size}) Output : {output} Time taken: {elapsed_time} seconds')
        print("#############################################################################################")
        bleu.append(self_bleu_score(output))
        i = i + 1
    return bleu


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
            input_ids = tokenizer.encode(text, add_special_tokens=True)
            input_ids = np.array(input_ids)[np.newaxis, :]
            # Perform inference on the preprocessed input
            output = model.predict(input_ids)
            output_ids = np.argmax(output, axis=-1)
            output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return output_str
    else:
        input_ids = tokenizer.encode(text, return_tensors=framework)
        outputs = model.generate(input_ids)

        # Decode and return translated text
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text.strip()


def t5(model_name: str, model, framework: str, data_size: int):
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # t5_dataset = load_dataset("opus100", "de-en", split="test", streaming=True)
    # t5_dataset = itertools.islice(t5_dataset, data_size)
    t5_dataset = [{"de": "DIE HOHEN VERTRAGSPARTEIEN, ", "en": "THE HIGH CONTRACTING PARTIES, "}]

    i = 0
    bleu = []
    for data in t5_dataset:
        sentence = data['en']
        target = data['de']
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
        print(f'({i + 1}/{data_size}) Translated_text : {output} Target: {target} Time taken: {elapsed_time} seconds')
        print("#############################################################################################")

        bleu.append(output == target)
        i = i + 1
    return bleu


def codegen(model_name: str, model, framework: str, data_size: int):
    if 'parrot' in model_name:
        tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot-small")
    else:
        tokenizer = GPT2Tokenizer.from_pretrained("microsoft/CodeGPT-small-py")

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
        print(f'({i + 1}/{data_size}) Output : {output} Time taken: {elapsed_time} seconds')
        print("#############################################################################################")
        bleu.append(self_bleu_score(output))
        i = i + 1
    return bleu
