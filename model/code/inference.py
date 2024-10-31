import onnxruntime as ort
from PIL import Image
from tempfile import TemporaryDirectory
import os
import numpy as np
import json


def model_fn(model_dir):
    return ort.InferenceSession(model_dir + "/" + "model.onnx")


def input_fn(request_body, request_content_type):
    with TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, 'image.jpg')
        with open(file_path, 'wb') as f:
            f.write(request_body)
        image = Image.open(file_path).convert("RGB")
        image = image.resize((384, 216))
        input_data = np.array(image).astype(np.float32).transpose(2, 0, 1) / 255.0
        return np.expand_dims(input_data, axis=0)


def predict_fn(input_data, model):
    return model.run(None, {"input": input_data})


def output_fn(prediction, response_content_type):
    with open("coco-labels-paper.txt") as f:
        target_labels = f.read().splitlines()

    boxes = prediction[0]
    labels = prediction[1]
    scores = prediction[2]

    response_boxes = []
    response_labels = []
    response_scores = []

    for i, score in enumerate(scores):
        if score > 0.7:
            response_boxes.append(boxes[i].tolist())
            response_labels.append(target_labels[labels[i] - 1])
            response_scores.append(score)

    return json.dumps({"boxes": response_boxes, "labels": response_labels, "scores": response_scores}), response_content_type
