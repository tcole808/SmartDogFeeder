from flask import Flask, request, jsonify
from PIL import Image
import io
import os
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import models, transforms

app = Flask(__name__)

# ==========================================
# CONFIG
# ==========================================
# Resolve paths relative to this file so running the server from any working
# directory still finds the model weights.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "..",
    "dogBowlApp",
    "animal_model_out",
    "best_model.pt",
)
IMG_SIZE = 224
DOG_CONFIDENCE_THRESHOLD = 0.65

SAVE_DIR = "detections"
os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_NAMES = [
    "butterfly",
    "cat",
    "chicken",
    "cow",
    "dog",
    "elephant",
    "horse",
    "human",
    "sheep",
    "spider",
    "squirrel",
]

DOG_CLASS_NAME = "dog"

# ==========================================
# TRANSFORMS
# ==========================================
eval_tfs = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def build_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model():
    model = build_model(len(CLASS_NAMES))
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        cleaned_state_dict[new_key] = v

    model.load_state_dict(cleaned_state_dict)
    model.eval()

    return model


MODEL = load_model()


def preprocess_image(image):
    image = image.convert("RGB")
    tensor = eval_tfs(image).unsqueeze(0)
    return tensor


def predict(image):

    x = preprocess_image(image)

    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    pred_idx = pred_idx.item()
    confidence = conf.item()

    predicted_label = CLASS_NAMES[pred_idx]

    dog_detected = (
        predicted_label == DOG_CLASS_NAME and
        confidence >= DOG_CONFIDENCE_THRESHOLD
    )

    return predicted_label, confidence, dog_detected


@app.route("/predict", methods=["POST"])
def predict_route():

    if "image" not in request.files:
        return jsonify({"error": "no image uploaded"}), 400

    file = request.files["image"]

    try:

        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        predicted_label, confidence, dog_detected = predict(image)

        # =====================================
        # SAVE IMAGE WITH TIMESTAMP
        # =====================================

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        filename = f"{timestamp}_{predicted_label}_{confidence:.2f}.jpg"

        save_path = os.path.join(SAVE_DIR, filename)

        image.save(save_path)

        print(f"Saved image: {save_path}")

        return jsonify({
            "predicted_label": predicted_label,
            "confidence": confidence,
            "dog_detected": dog_detected,
            "saved_image": filename
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)