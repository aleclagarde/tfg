from transformers import ViTFeatureExtractor, ViTForImageClassification
from codecarbon import track_emissions
import requests
from PIL import Image

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")


@track_emissions
def infer_vit(image_path):
    url = "https://images.unsplash.com/photo-1611042777789-9ac8774e8694"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Get the predicted class index and score
    predicted_class = outputs.logits.argmax().item()
    predicted_score = outputs.logits.softmax(dim=-1)[0, predicted_class].item()

    # Print the predicted class index and score
    print(f"Predicted class: {predicted_class}")
    print(f"Predicted score: {predicted_score}")
