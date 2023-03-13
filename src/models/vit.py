from transformers import ViTFeatureExtractor, ViTForImageClassification
from codecarbon import track_emissions
import torch

tokenizer = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")


@track_emissions
def infer_vit(image_path):
    image = tokenizer(images=image_path, return_tensors="pt").pixel_values
    outputs = model(image)

    return torch.argmax(outputs.logits).item()
