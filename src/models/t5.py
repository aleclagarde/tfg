from transformers import T5Tokenizer
from codecarbon import track_emissions

tokenizer = T5Tokenizer.from_pretrained("t5-base")


@track_emissions
def infer_t5_torch(model, text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


@track_emissions
def infer_t5_tf(model, text):
    inputs = tokenizer.encode(text, return_tensors='tf')
    outputs = model.generate(inputs)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
