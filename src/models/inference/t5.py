from transformers import T5Tokenizer
from codecarbon import track_emissions

tokenizer = T5Tokenizer.from_pretrained("t5-base")


@track_emissions
def infer_t5(model, framework, text):
    inputs = tokenizer.encode(text, return_tensors=framework)
    outputs = model.generate(inputs)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
