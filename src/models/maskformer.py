from transformers import SwinTransformerFeatureExtractor, SwinTransformerForSemanticSegmentation
from codecarbon import track_emissions

tokenizer = SwinTransformerFeatureExtractor.from_pretrained("facebook/maskformer-swin-small-coco")
model = SwinTransformerForSemanticSegmentation.from_pretrained("facebook/maskformer-swin-small-coco")


@track_emissions
def infer_maskformer(image_path):
    image = tokenizer(images=image_path, return_tensors="pt").pixel_values
    outputs = model(image)

    return outputs.logits

