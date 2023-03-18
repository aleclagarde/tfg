from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from codecarbon import track_emissions
from PIL import Image
import requests

feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-small-coco")
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-small-coco")


@track_emissions
def infer_maskformer(image_path):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = feature_extractor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    # model predicts class_queries_logits of shape `(batch_size, num_queries)`
    # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits

    # you can pass them to feature_extractor for postprocessing
    result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
    predicted_panoptic_map = result["segmentation"]

