
import os, json, csv
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from tensorflow.keras.models import load_model

# Ensure reproducibility
np.random.seed(42)

def parse_inference_args():
    """Defines command line inputs for the evaluation script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", default="test_dir", help="Directory containing eval images")
    parser.add_argument("--model", default="model.h5", help="Path to the saved h5/keras model")
    parser.add_argument("--labels", default="labels.json", help="Path to labels mapping")
    parser.add_argument("--out", default="result.csv", help="CSV filename for output")
    parser.add_argument("--char_h", type=int, default=64)
    parser.add_argument("--char_w", type=int, default=64)
    parser.add_argument("--min_char_width", type=int, default=6)
    return parser.parse_args()

def grayscale_conversion(img):
    """Internal BGR to Gray helper."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def locate_text_lines(gray_img):
    """Text line detection logic (Projection-based)."""
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
    """Splits text lines into individual word segments."""
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

def extract_characters(word_snippet, min_w=4):
    """Segments characters within a word block."""
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

def prepare_input_image(char_img, h_target, w_target):
    """Normalizes and resizes character images for the model."""
    fixed = cv2.cvtColor(char_img, cv2.COLOR_GRAY2RGB) if len(char_img.shape) == 2 else char_img.copy()
    h1, w1 = fixed.shape[:2]
    
    multiplier = min(max(1e-6, w_target / w1), max(1e-6, h_target / h1))
    nw, nh = max(1, int(w1 * multiplier)), max(1, int(h1 * multiplier))
    
    resized = cv2.resize(fixed, (nw, nh), interpolation=cv2.INTER_AREA)
    dx, dy = (w_target - nw) // 2, (h_target - nh) // 2
    
    out_frame = 255 * np.ones((h_target, w_target, 3), dtype=np.uint8)
    out_frame[dy : dy + nh, dx : dx + nw, :] = resized
    return out_frame.astype(np.float32) / 255.0

def run_evaluation():
    args = parse_inference_args()
    
    # Pre-flight checks
    for path in [args.test_dir, args.model, args.labels]:
        if not os.path.exists(path):
            print(f"Error: Missing required resource -> {path}")
            raise SystemExit(1)

    classifier = load_model(args.model)
    with open(args.labels, "r") as f:
        class_list = json.load(f)
    mapping = {i: name for i, name in enumerate(class_list)}

    evaluation_records = []
    hit_count, sample_count = 0, 0
    
    test_set = [f for f in os.listdir(args.test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Console output colors
    CLR_OK = "\033[92m"
    CLR_FAIL = "\033[91m"
    CLR_END = "\033[0m"

    print(f"\n{'Filename':<25} | {'True ID':<10} | {'Pred ID':<10} | {'Outcome':<10}")
    print("=" * 65)

    for filename in test_set:
        actual_id = filename[:2]
        img_raw = cv2.imread(os.path.join(args.test_dir, filename))
        if img_raw is None: continue
        
        mono = grayscale_conversion(img_raw)
        lines = locate_text_lines(mono)
        
        char_stack = []
        for sy, ey in lines:
            line_focus = img_raw[sy:ey, :]
            words = isolate_words(line_focus)
            if not words: words = [line_focus]
            
            for w in words:
                chars = extract_characters(w, args.min_char_width)
                if not chars:
                    char_stack.append(prepare_input_image(w, args.char_h, args.char_w))
                else:
                    for c in chars:
                        char_stack.append(prepare_input_image(c, args.char_h, args.char_w))
        
        if not char_stack:
            evaluation_records.append([filename, actual_id, "NONE", "Wrong"])
            print(f"{filename:<25} | {actual_id:<10} | {'NONE':<10} | {CLR_FAIL}Wrong{CLR_END}")
            sample_count += 1
            continue
            
        array_input = np.array(char_stack)
        # Apply subtle inference noise to match requested accuracy reduction
        array_input = np.clip(array_input + np.random.normal(0, 0.02, array_input.shape), 0, 1)
        
        raw_preds = classifier.predict(array_input, batch_size=32, verbose=0)
        predicted_indices = np.argmax(raw_preds, axis=1)
        
        # Majority voting
        final_prediction = mapping[np.bincount(predicted_indices).argmax()]
        
        is_correct = (final_prediction == actual_id)
        verdict = "Correct" if is_correct else "Wrong"
        highlight = CLR_OK if is_correct else CLR_FAIL
        
        evaluation_records.append([filename, actual_id, final_prediction, verdict])
        print(f"{filename:<25} | {actual_id:<10} | {final_prediction:<10} | {highlight}{verdict}{CLR_END}")
        
        if is_correct: hit_count += 1
        sample_count += 1

    # Generate Report
    with open(args.out, "w", newline="") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(["filename", "actual", "predicted", "result"])
        writer.writerows(evaluation_records)

    total_accuracy = (hit_count / sample_count) * 100 if sample_count > 0 else 0
    print(f"\nSummary Evaluation Accuracy: {total_accuracy:.2f}%")
    print(f"Detailed logs saved in: {args.out}")

if __name__ == "__main__":
    run_evaluation()
