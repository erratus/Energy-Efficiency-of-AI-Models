import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from cuml.cluster import KMeans


# Load SAM model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

# YOLO Class Mapping
CLASS_MAPPING = {
    0: "home",      # White circles
    1: "opponent",  # Black triangles
    2: "ball"       # "+" sign
}

def detect_ball(image):
    """Detects the ball (+ symbol) using template matching."""
    ball_template = cv2.imread("ball_template.jpg", cv2.IMREAD_GRAYSCALE)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(image_gray, ball_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val > 0.7:  # Confidence threshold
        x, y = max_loc
        return (x, y, ball_template.shape[1], ball_template.shape[0])  # x, y, w, h
    return None

def cluster_objects(masks, image):
    """Clusters detected objects into 'home' and 'opponent' using color features."""
    features = []
    bounding_boxes = []

    for mask in masks:
        x, y, w, h = cv2.boundingRect(mask["segmentation"])
        roi = image[y:y+h, x:x+w]
        avg_color = np.mean(roi, axis=(0, 1))  # Extract average color
        features.append(avg_color)
        bounding_boxes.append((x, y, w, h))

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(features)
    labels = kmeans.labels_

    return labels, bounding_boxes

def annotate_image(image_path, output_dir):
    """Generates YOLO annotations for a single image."""
    image = cv2.imread(image_path)
    masks = mask_generator.generate(image)

    if not masks:
        return  # No objects detected

    labels, bounding_boxes = cluster_objects(masks, image)

    yolo_annotations = []
    for (x, y, w, h), label in zip(bounding_boxes, labels):
        # Convert to YOLO format
        img_h, img_w = image.shape[:2]
        x_center, y_center = (x + w / 2) / img_w, (y + h / 2) / img_h
        w, h = w / img_w, h / img_h

        yolo_annotations.append(f"{label} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    # Detect and add ball (+)
    ball_bbox = detect_ball(image)
    if ball_bbox:
        x, y, w, h = ball_bbox
        img_h, img_w = image.shape[:2]
        x_center, y_center = (x + w / 2) / img_w, (y + h / 2) / img_h
        w, h = w / img_w, h / img_h
        yolo_annotations.append(f"2 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")  # Class 2 is ball

    # Save annotations
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(output_dir, f"{image_name}.txt")

    with open(label_path, "w") as f:
        f.write("\n".join(yolo_annotations))

def process_dataset(image_dir, output_label_dir):
    """Processes all images in the dataset."""
    os.makedirs(output_label_dir, exist_ok=True)

    for img_file in os.listdir(image_dir):
        if img_file.endswith(('.jpg', '.png')):
            annotate_image(os.path.join(image_dir, img_file), output_label_dir)

    print("Annotation complete!")

# Run on dataset
image_dir = "data/keyframes/final"
output_label_dir = "dataset/labels"
process_dataset(image_dir, output_label_dir)
