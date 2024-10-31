import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import requests


def scale_bbox(bbox, original_size, resized_size):
    """
    バウンディングボックスを元画像の座標系にスケール変換する関数。

    Args:
        bbox (list): [x1, y1, x2, y2] 形式のバウンディングボックス
        original_size (tuple): 元の画像サイズ (width, height)
        resized_size (tuple): リサイズ後の画像サイズ (width, height)

    Returns:
        list: 元の画像に対応するスケール変換後のバウンディングボックス
    """
    orig_w, orig_h = original_size
    resized_w, resized_h = resized_size

    # スケールファクタの計算
    x_scale = orig_w / resized_w
    y_scale = orig_h / resized_h

    # 各座標をスケーリング
    x1 = int(bbox[0] * x_scale)
    y1 = int(bbox[1] * y_scale)
    x2 = int(bbox[2] * x_scale)
    y2 = int(bbox[3] * y_scale)

    return np.array([x1, y1, x2, y2])


session = ort.InferenceSession("model.onnx")

image_path = "../images/image_4.png"
image = Image.open(image_path).convert("RGB")

image = image.resize((384, 216))

input_data = np.array(image).astype(np.float32).transpose(2, 0, 1) / 255.0
input_data = np.expand_dims(input_data, axis=0)

url = "https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-paper.txt"
response = requests.get(url)
target_labels = response.text.splitlines()

# 推論の実行
boxes, labels, scores = session.run(None, {"input": input_data})

boxes = [scale_bbox(b, (3840, 2160), (384, 216)) for b in boxes]
image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)  # OpenCVで画像読み込み
for i, (box, score) in enumerate(zip(boxes, scores)):
    if score > 0.5:  # スコアが0.5以上のものを表示
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Class: {target_labels[labels[i]-1]}, Score: {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

# 結果を表示
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("off")
plt.savefig("result.png")
