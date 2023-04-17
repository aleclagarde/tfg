from transformers import T5Tokenizer
from codecarbon import track_emissions
import tensorflow as tf
import torch

tokenizer = T5Tokenizer.from_pretrained("t5-base")


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
        text = "fr : " + text
        input_ids = tokenizer.encode(text, return_tensors=framework)
        outputs = model.generate(input_ids)

        # Decode and return translated text
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text.strip()
