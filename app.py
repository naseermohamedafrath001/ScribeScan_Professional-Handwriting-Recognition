import os, json
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import uuid

app = Flask(__name__)

# Load Model and Labels
MODEL_PATH = "model.h5"
LABELS_PATH = "labels.json"

if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    raise FileNotFoundError("model.h5 or labels.json not found!")

model = load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)
mapping = {i: name for i, name in enumerate(labels)}

# --- Inference Helpers (Extracted from run.py) ---

def grayscale_conversion(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def locate_text_lines(gray_img):
    h = gray_img.shape[0]
    _, th = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    cleaned = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)
    projection = np.sum(cleaned, axis=1)
    if projection.max() == 0: return [(0, h)]
    limit = max(1, int(0.03 * projection.max()))
    coord_ranges = []
    is_active = False
    s_ptr = 0
    for y, val in enumerate(projection):
        if val > limit and not is_active:
            is_active, s_ptr = True, y
        elif val <= limit and is_active:
            e_ptr = y
            is_active = False
            if e_ptr - s_ptr >= 6:
                coord_ranges.append((max(0, s_ptr - 2), min(h, e_ptr + 2)))
    if is_active: coord_ranges.append((s_ptr, h))
    return coord_ranges

def isolate_words(line_snippet):
    gray = grayscale_conversion(line_snippet)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    expanded = cv2.dilate(th, k, iterations=1)
    blobs, _ = cv2.findContours(expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for b in blobs:
        x, y, w, h = cv2.boundingRect(b)
        if w < 8 or h < 8: continue
        boxes.append((x, y, w, h))
    boxes = sorted(boxes, key=lambda i: i[0])
    return [line_snippet[y:y+h, x:x+w] for (x, y, w, h) in boxes]

def extract_characters(word_snippet, min_w=6):
    gray = grayscale_conversion(word_snippet)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    col_density = np.sum(th, axis=0)
    limit = max(1, int(0.05 * col_density.max()))
    whitespace = col_density <= limit
    parts = []
    active = False
    start = 0
    for i, is_empty in enumerate(whitespace):
        if not is_empty and not active:
            active, start = True, i
        elif is_empty and active:
            end = i
            active = False
            if end - start >= min_w:
                parts.append(word_snippet[:, start:end])
    if active:
        end = len(whitespace)
        if end - start >= min_w:
            parts.append(word_snippet[:, start:end])
    return parts

def prepare_input_image(char_img, h_target=64, w_target=64):
    fixed = cv2.cvtColor(char_img, cv2.COLOR_GRAY2RGB) if len(char_img.shape) == 2 else char_img.copy()
    h1, w1 = fixed.shape[:2]
    multiplier = min(max(1e-6, w_target / w1), max(1e-6, h_target / h1))
    nw, nh = max(1, int(w1 * multiplier)), max(1, int(h1 * multiplier))
    resized = cv2.resize(fixed, (nw, nh), interpolation=cv2.INTER_AREA)
    dx, dy = (w_target - nw) // 2, (h_target - nh) // 2
    out_frame = 255 * np.ones((h_target, w_target, 3), dtype=np.uint8)
    out_frame[dy : dy + nh, dx : dx + nw, :] = resized
    return out_frame.astype(np.float32) / 255.0

# --- Flask Routes ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Read image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_raw is None:
        return jsonify({"error": "Invalid image format"}), 400

    # Processing (Voting Logic)
    mono = grayscale_conversion(img_raw)
    lines = locate_text_lines(mono)
    
    char_stack = []
    for sy, ey in lines:
        line_focus = img_raw[sy:ey, :]
        words = isolate_words(line_focus)
        if not words: words = [line_focus]
        for w in words:
            chars = extract_characters(w)
            if not chars:
                char_stack.append(prepare_input_image(w))
            else:
                for c in chars:
                    char_stack.append(prepare_input_image(c))

    if not char_stack:
        return jsonify({"label": "NONE", "confidence": 0.0, "message": "No characters detected"})

    array_input = np.array(char_stack)
    # Slight noise for realism as per project constraints
    array_input = np.clip(array_input + np.random.normal(0, 0.02, array_input.shape), 0, 1)
    
    raw_preds = model.predict(array_input, verbose=0)
    predicted_indices = np.argmax(raw_preds, axis=1)
    
    # Majority Voting
    final_idx = np.bincount(predicted_indices).argmax()
    final_prediction = mapping[final_idx]
    
    # Calculate confidence as percentage of votes
    votes = np.count_nonzero(predicted_indices == final_idx)
    confidence = (votes / len(predicted_indices)) * 100

    return jsonify({
        "label": final_prediction,
        "confidence": round(confidence, 2),
        "num_segments": len(char_stack)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
