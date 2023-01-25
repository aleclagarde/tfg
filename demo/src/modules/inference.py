from torch.nn.utils.prune import l1_unstructured
from transformers import T5Tokenizer, T5ForConditionalGeneration
from codecarbon import track_emissions

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small", low_cpu_mem_usage=True)


@track_emissions
def infer_t5(text, prune_pct):
    # Prune the model
    if prune_pct > 0:
        for name, param in model.named_parameters():
            if "embedding" in name:
                l1_unstructured(param, name='weight', amount=prune_pct)

    input_ids = tokenizer(text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
