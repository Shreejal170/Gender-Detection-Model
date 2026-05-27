#!/usr/bin/env python3
"""
Gender Detection Script (Final Version)
---------------------------------------
This script performs gender detection on a face within an image.
It extracts 4 crop versions of the face:
  1. Default InsightFace bounding box
  2. Highly Cropped (standard tight face)
  3. Super Highly Cropped (eyes, nose, mouth only)
  4. Expanded (hair visible)

It runs inference on each crop level using a trained EfficientNet-B0 model,
averages the probability distributions across all 4 crop results, and reports
both individual and average results.

Usage:
    python gender_detector.py --image <path_to_image> [--model <path_to_weights>]
"""

import os
import sys
import argparse
import logging
from enum import Enum
from typing import Dict, Tuple, Any
# from datetime import time
import time

# Configure robust and clear logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("gender_detector")

# Attempt to import dependencies with clear warnings if missing
try:
    import cv2
    import numpy as np
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image
    from insightface.app import FaceAnalysis
except ImportError as e:
    logger.error(
        f"Missing required dependency: {e.name}. "
        "Please ensure PyTorch, Torchvision, OpenCV, InsightFace, Pillow, and NumPy are installed."
    )
    sys.exit(1)


class GenderEnum(Enum):
    """
    Enum representing the outcomes of the gender detection process.
    """
    NO_FACE_DETECTED = "No Face Detected"
    MALE = "Male"
    FEMALE = "Female"


class GenderDetector:
    """
    A class wrapper to handle face detection, multi-crop generation,
    and ensemble gender classification.
    """

    def __init__(self, model_weights_path: str, device: str = None):
        """
        Initializes the FaceAnalysis app (InsightFace) and loading PyTorch classification model.

        Args:
            model_weights_path: Path to the trained EfficientNet-B0 weight file (.pth).
            device: Computing device ('cuda', 'cpu', etc.). If None, automatically selected.
        """
        # 1. Setup device acceleration
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # 2. Initialize and prepare InsightFace Detector
        logger.info("Initializing InsightFace detector...")
        # ctx_id=-1 forces CPU execution locally for detector reliability
        self.face_app = FaceAnalysis(allowed_modules=['detection'])
        self.face_app.prepare(ctx_id=-1, det_size=(640, 640))

        # 3. Re-create training architecture (EfficientNet-B0)
        logger.info("Building classifier architecture (EfficientNet-B0)...")
        self.model = models.efficientnet_b0(weights=None)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_ftrs, 2)
        )

        # 4. Load trained weights
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"Model weight file not found at: {model_weights_path}")
        
        logger.info(f"Loading weights from {model_weights_path}...")
        self.model.load_state_dict(
            torch.load(model_weights_path, map_location=self.device)
        )
        self.model = self.model.to(self.device)
        self.model.eval()  # Set classifier to evaluation mode
        logger.info("Model loaded and initialized successfully!")

        # 5. Define evaluation image preprocessing pipeline
        self.infer_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_crops(self, img_bgr: np.ndarray, face: Any) -> Dict[str, Image.Image]:
        """
        Extracts 4 crop styles of a single detected face bounding box.

        Args:
            img_bgr: Loaded BGR image as a NumPy array.
            face: The detected face object containing a bbox field.

        Returns:
            A dictionary containing the 4 crop PIL Image objects.
        """
        h, w, _ = img_bgr.shape
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        
        box_height = y2 - y1
        box_width = x2 - x1

        face_crops = {}

        # 1. Super Highly Cropped (Eyes, Nose, Mouth only)
        shc_y1 = max(0, y1 + int(box_height * 0.30))
        shc_y2 = min(h, y2 - int(box_height * 0.15))
        shc_x1 = max(0, x1 + int(box_width * 0.20))
        shc_x2 = min(w, x2 - int(box_width * 0.20))
        face_crops["Highly Cropped (Super)"] = Image.fromarray(
            cv2.cvtColor(img_bgr[shc_y1:shc_y2, shc_x1:shc_x2], cv2.COLOR_BGR2RGB)
        )

        # 2. Highly Cropped (Standard Tight Face)
        hc_y1 = max(0, y1 + int(box_height * 0.12))
        hc_y2 = min(h, y2 - int(box_height * 0.05))
        hc_x1 = max(0, x1 + int(box_width * 0.05))
        hc_x2 = min(w, x2 - int(box_width * 0.05))
        face_crops["Cropped (Highly)"] = Image.fromarray(
            cv2.cvtColor(img_bgr[hc_y1:hc_y2, hc_x1:hc_x2], cv2.COLOR_BGR2RGB)
        )

        # 3. Default InsightFace (Raw bounding box)
        df_y1 = max(0, y1)
        df_y2 = min(h, y2)
        df_x1 = max(0, x1)
        df_x2 = min(w, x2)
        face_crops["Default"] = Image.fromarray(
            cv2.cvtColor(img_bgr[df_y1:df_y2, df_x1:df_x2], cv2.COLOR_BGR2RGB)
        )

        # 4. Expanded (Hair Visible)
        ex_y1 = max(0, y1 - int(box_height * 0.45))
        ex_y2 = min(h, y2 + int(box_height * 0.25))
        ex_x1 = max(0, x1 - int(box_width * 0.40))
        ex_x2 = min(w, x2 + int(box_width * 0.40))
        face_crops["Hair Visible"] = Image.fromarray(
            cv2.cvtColor(img_bgr[ex_y1:ex_y2, ex_x1:ex_x2], cv2.COLOR_BGR2RGB)
        )

        return face_crops

    def get_prediction_probs(self, pil_image: Image.Image) -> np.ndarray:
        """
        Runs inference on a PIL Image and returns the softmax class probabilities.

        Args:
            pil_image: Pre-cropped PIL face image.

        Returns:
            NumPy array of shape (2,) representing probabilities [Female, Male].
        """
        start_time = time.time()
        tensor = self.infer_transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
        end_time = time.time()
        logger.info(f'Total time taken: {end_time - start_time}')
        return probabilities.squeeze().cpu().numpy()


    def analyze_image(self, image_path: str) -> Tuple[GenderEnum, Dict[str, Dict[str, Any]]]:
        """
        Processes a single image file to detect the primary face, extract the 4 crops,
        run gender inference, and compute the ensemble consensus average.

        Args:
            image_path: Absolute or relative path to the target image file.

        Returns:
            Tuple of (GenderEnum consensus result, dictionary of detailed results).
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Target image file not found at: {image_path}")

        # 1. Load image in BGR format
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Failed to read image at: {image_path}. Check if file is corrupted.")

        # 2. Get face detection outputs
        faces = self.face_app.get(img_bgr)

        # 3. Handle 'No Face Detected' case
        if not faces:
            logger.error("no face detected")
            return GenderEnum.NO_FACE_DETECTED, {}

        # Use the primary (first detected) face
        primary_face = faces[0]

        # 4. Extract 4 crop styles
        crops = self.extract_crops(img_bgr, primary_face)

        # 5. Run inference on each crop and collect predictions/probabilities
        crop_results = {}
        all_probs = []

        for crop_name, crop_img in crops.items():
            probs = self.get_prediction_probs(crop_img)  # [Female Prob, Male Prob]
            all_probs.append(probs)

            # Class index 0 = Female, Class index 1 = Male
            predicted_class = np.argmax(probs)
            confidence = probs[predicted_class] * 100.0
            label = "Male" if predicted_class == 1 else "Female"

            crop_results[crop_name] = {
                "label": label,
                "confidence": confidence,
                "probabilities": probs
            }

        # 6. Calculate Average Probability over the 4 results
        average_probs = np.mean(all_probs, axis=0)
        avg_class = np.argmax(average_probs)
        avg_confidence = average_probs[avg_class] * 100.0
        avg_label = "Male" if avg_class == 1 else "Female"

        consensus_enum = GenderEnum.MALE if avg_class == 1 else GenderEnum.FEMALE

        detailed_results = {
            "crops": crop_results,
            "average": {
                "label": avg_label,
                "confidence": avg_confidence,
                "probabilities": average_probs
            }
        }

        return consensus_enum, detailed_results


def main():
    """
    Main function providing Command-Line Interface (CLI) execution capability.
    """
    default_model_path = r"C:\Users\shree\Documents\Veel\GenderDetection\model_download_gender\kaggle\working\best_gender_model.pth"

    parser = argparse.ArgumentParser(
        description="Run clean, 4-tier crop averaged gender detection on a face image."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the image file to analyze."
    )
    parser.add_argument(
        "--model",
        default=default_model_path,
        help="Path to the trained PyTorch EfficientNet model weights."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Target hardware device for inference (cpu, cuda, etc.)."
    )

    args = parser.parse_args()

    try:
        # Initialize detector
        detector = GenderDetector(model_weights_path=args.model, device=args.device)

        # Analyze image
        result_enum, details = detector.analyze_image(args.image)

        # Handle no face detection gracefully according to requirement
        if result_enum == GenderEnum.NO_FACE_DETECTED:
            # Note: logging.error has already logged "no face detected" inside analyze_image
            # Stop executing and exit with error code
            sys.exit(1)

        # Display results in a beautifully formatted table
        print("\n" + "=" * 55)
        print(f"       GENDER ANALYSIS RESULTS: {os.path.basename(args.image)}")
        print("=" * 55)
        print(f"{'Crop Strategy':<25} | {'Predicted':<10} | {'Confidence':<10}")
        print("-" * 55)

        for crop_name, info in details["crops"].items():
            print(f"{crop_name:<25} | {info['label']:<10} | {info['confidence']:.2f}%")

        print("-" * 55)
        
        # Display the ensemble average results
        avg_info = details["average"]
        print(f"{'ENSEMBLE AVERAGE (4 Crops)':<25} | {avg_info['label']:<10} | {avg_info['confidence']:.2f}%")
        print("=" * 55)
        print(f"Final Enum Value: {result_enum}")
        print("=" * 55 + "\n")

        # Exit successfully
        sys.exit(0)

    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
