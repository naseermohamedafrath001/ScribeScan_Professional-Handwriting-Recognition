
import os, json, random
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import argparse

def get_config():
    """Setup command line arguments."""
    p = argparse.ArgumentParser(description="Handwriting Model Training Script")
    p.add_argument("--train_dir", type=str, default="train_dir", help="Source directory for training data")
    p.add_argument("--model_out", type=str, default="model.h5", help="Output path for the trained model")
    p.add_argument("--labels_out", type=str, default="labels.json", help="Output path for the class labels")
    p.add_argument("--char_h", type=int, default=64)
    p.add_argument("--char_w", type=int, default=64)
    p.add_argument("--min_char_width", type=int, default=6)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--val_fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def init_random_seeds(val):
    """Sets seeds for reproducibility."""
    random.seed(val)
    np.random.seed(val)
    tf.random.set_seed(val)

def fetch_image_list(base_path):
    """Recursively collects all image paths."""
    directory = Path(base_path)
    allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    return sorted([str(f) for f in directory.rglob("*") if f.suffix.lower() in allowed_extensions])

def parse_label(file_path):
    """Extracts label from the filename prefix."""
    return Path(file_path).name[:2]

def grayscale_conversion(image):
    """Converts a BGR image to grayscale."""
    if image is None: return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

def locate_text_lines(gray_img):
    """Finds vertical boundaries of text lines using projection."""
    height = gray_img.shape[0]
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
    
    horizontal_projection = np.sum(cleaned, axis=1)
    if horizontal_projection.max() == 0: return [(0, height)]
    
    noise_threshold = max(1, int(0.03 * horizontal_projection.max()))
    line_indices = []
    active = False
    start_y = 0
    
    for y, intensity in enumerate(horizontal_projection):
        if intensity > noise_threshold and not active:
            active, start_y = True, y
        elif intensity <= noise_threshold and active:
            end_y = y
            active = False
            if end_y - start_y >= 6:
                line_indices.append((max(0, start_y - 2), min(height, end_y + 2)))
    
    if active: line_indices.append((start_y, height))
    return line_indices

def isolate_words(line_snippet):
    """Segments a line image into word blocks."""
    gray = grayscale_conversion(line_snippet)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    stretched = cv2.dilate(binary, dilation_kernel, iterations=1)
    shapes, _ = cv2.findContours(stretched, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for s in shapes:
        x, y, w, h = cv2.boundingRect(s)
        if w < 8 or h < 8: continue
        regions.append((x, y, w, h))
    
    regions = sorted(regions, key=lambda r: r[0])
    return [line_snippet[y:y+h, x:x+w] for (x, y, w, h) in regions]

def extract_characters(word_snippet, min_w=4):
    """Splits a word into individual characters."""
    gray = grayscale_conversion(word_snippet)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    vertical_projection = np.sum(binary, axis=0)
    gap_threshold = max(1, int(0.05 * vertical_projection.max()))
    
    splits = vertical_projection <= gap_threshold
    segments = []
    active = False
    start_x = 0
    
    for x, is_gap in enumerate(splits):
        if not is_gap and not active:
            active, start_x = True, x
        elif is_gap and active:
            end_x = x
            active = False
            if end_x - start_x >= min_w:
                segments.append(word_snippet[:, start_x:end_x])
    
    if active:
        end_x = len(splits)
        if end_x - start_x >= min_w:
            segments.append(word_snippet[:, start_x:end_x])
    return segments

def prepare_input_image(char_img, h_target, w_target):
    """Resizes and pads character image to target dimensions."""
    working_char = cv2.cvtColor(char_img, cv2.COLOR_GRAY2RGB) if len(char_img.shape) == 2 else char_img.copy()
    curr_h, curr_w = working_char.shape[:2]
    
    ratio = min(max(1e-6, w_target / curr_w), max(1e-6, h_target / curr_h))
    new_w, new_h = max(1, int(curr_w * ratio)), max(1, int(curr_h * ratio))
    
    resized = cv2.resize(working_char, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x_offset, y_offset = (w_target - new_w) // 2, (h_target - new_h) // 2
    
    canvas = 255 * np.ones((h_target, w_target, 3), dtype=np.uint8)
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w, :] = resized
    return canvas.astype(np.float32) / 255.0

# Define data augmentation pipeline
input_augmenter = tf.keras.Sequential([
    layers.RandomRotation(0.20),
    layers.RandomTranslation(0.10, 0.10),
    layers.RandomZoom(0.15, 0.15),
    layers.RandomContrast(0.20),
    layers.GaussianNoise(0.02),
], name="spatial_augmentation")

@tf.function
def run_tf_augmentation(batch):
    return input_augmenter(batch)

def apply_image_noise(img_data):
    """Wraps TF augmentation for numpy inputs."""
    normalized = img_data.astype(np.float32) / 255.0
    tensor = tf.convert_to_tensor(normalized[None, ...], dtype=tf.float32)
    processed = run_tf_augmentation(tensor)
    result_np = processed[0].numpy()
    return np.clip(result_np * 255.0, 0, 255).astype(np.uint8)

def generate_network(input_dims, class_count):
    """Constructs the Deep CNN architecture."""
    input_layer = layers.Input(shape=input_dims)
    # Inline character-level jitter
    hid = layers.RandomRotation(0.05)(input_layer)
    hid = layers.RandomTranslation(0.03, 0.03)(hid)
    
    for f_size in [32, 64, 128, 256, 256]:
        hid = layers.Conv2D(f_size, 3, padding="same", activation="relu")(hid)
        hid = layers.BatchNormalization()(hid)
        hid = layers.MaxPool2D()(hid)
        hid = layers.Dropout(0.25)(hid)
    
    hid = layers.GlobalAveragePooling2D()(hid)
    hid = layers.Dense(1024, activation="relu")(hid)
    hid = layers.BatchNormalization()(hid)
    hid = layers.Dropout(0.5)(hid)
    final_output = layers.Dense(class_count, activation="softmax")(hid)
    
    return models.Model(input_layer, final_output)

def main():
    cfg = get_config()
    init_random_seeds(cfg.seed)
    
    data_files = fetch_image_list(cfg.train_dir)
    if not data_files: raise SystemError("Directory is empty")
    
    unique_labels = sorted({parse_label(f) for f in data_files})
    label_map = {name: idx for idx, name in enumerate(unique_labels)}
    
    feature_set, target_set = [], []
    
    for path in tqdm(data_files, desc="Processing Samples"):
        original = cv2.imread(path)
        if original is None: continue
        
        _, full_w = original.shape[:2]
        step = full_w // 3
        sub_patches = [original[:, :step], original[:, step : 2*step], original[:, 2*step:]]
        
        # Increase dataset variety
        multiplied_patches = []
        for sp in sub_patches:
            multiplied_patches.append(sp)
            try:
                multiplied_patches.append(apply_image_noise(sp))
            except Exception:
                multiplied_patches.append(sp.copy())
        
        for patch in multiplied_patches:
            mono = grayscale_conversion(patch)
            found_lines = locate_text_lines(mono)
            for y_start, y_end in found_lines:
                line_img = patch[y_start:y_end, :]
                found_words = isolate_words(line_img)
                if not found_words: found_words = [line_img]
                
                for w_img in found_words:
                    chars = extract_characters(w_img, cfg.min_char_width)
                    if not chars:
                        feature_set.append(prepare_input_image(w_img, cfg.char_h, cfg.char_w))
                        target_set.append(label_map[parse_label(path)])
                    else:
                        for c_img in chars:
                            feature_set.append(prepare_input_image(c_img, cfg.char_h, cfg.char_w))
                            target_set.append(label_map[parse_label(path)])
                            
    features = np.array(feature_set, dtype=np.float32)
    targets = np.array(target_set, dtype=np.int32)
    
    # Shuffle for training
    order = np.random.permutation(len(features))
    features, targets = features[order], targets[order]
    targets_ohe = to_categorical(targets, num_classes=len(unique_labels))
    
    # Data partitioning
    split_idx = max(1, int(cfg.val_fraction * len(features)))
    val_x, val_y = features[:split_idx], targets_ohe[:split_idx]
    train_x, train_y = features[split_idx:], targets_ohe[split_idx:]
    
    # Balance losses
    distribution = Counter(targets.tolist())
    weights = {i: (len(targets) / (len(distribution) * distribution[i])) for i in distribution}
    
    # Model instantiation
    network = generate_network((cfg.char_h, cfg.char_w, 3), len(unique_labels))
    network.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Training constraints
    monitor_callbacks = [
        callbacks.ModelCheckpoint(cfg.model_out, monitor="val_accuracy", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1)
    ]
    
    print("\nStarting Model Training Phase...")
    network.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        shuffle=True,
        class_weight=weights,
        callbacks=monitor_callbacks
    )
    
    # Finalization
    network.save(cfg.model_out)
    with open(cfg.labels_out, "w") as f_out:
        json.dump(unique_labels, f_out, indent=2)
    print(f"\nProcessing Complete. Model: {cfg.model_out}, Mapping: {cfg.labels_out}")

if __name__ == "__main__":
    main()
