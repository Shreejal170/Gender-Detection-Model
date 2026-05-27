# Gender Detection Tool 🕵️‍♂️👩

A robust Python script utilizing **InsightFace** and **EfficientNet-B0** to detect the gender of a person in an image. To ensure maximum accuracy and stability, the script captures **four distinct crop variations** of the tracked face and takes the ensemble average of their probability scores.

## How It Works

Instead of relying on a single crop, this tool creates a balanced consensus by analyzing:
1. **Super Highly Cropped**: Eyes, nose, and mouth only (no ears/context).
2. **Highly Cropped**: Standard tight face framing.
3. **Default InsightFace**: The standard raw bounding box returned by the detector.
4. **Expanded (Hair Visible)**: Extra padding to include the head curve and hair.

The probabilities from all four versions are evaluated and unified to reach an `ENSEMBLE AVERAGE` prediction.

---

## Installation & Requirements

Ensure you have Python 3 installed. Install the necessary dependencies via pip:

```bash
pip install torch torchvision opencv-python pillow numpy insightface onnxruntime
```

*Note: The script will automatically fall back to CPU if a CUDA-enabled PyTorch environment is not found, though GPU acceleration requires the correct CUDA version of PyTorch.*

---

## Usage

You can execute the gender detector directly from the command line by passing the path to the target image.

### Basic Command
Navigate to the directory containing `gender_detector.py` and run:

```bash
python gender_detector.py --image "C:\path\to\your\image.jpg"
```

### Specifying the Model Weights (Optional)
If your `best_gender_model.pth` file is located somewhere else, you can point to it explicitly using the `--model` flag:

```bash
python gender_detector.py --image "C:\path\to\your\image.jpg" --model "C:\path\to\weights\best_gender_model.pth"
```

*(By default, the script will look for the model at: `C:\Users\shree\Documents\Veel\GenderDetection\model_download_gender\kaggle\working\best_gender_model.pth`)*

### Choosing the Computation Device (Optional)
Specify `--device` to force the classifier part to run on CPU or GPU (`cpu` or `cuda`). Note that face detection explicitly defaults to CPU for stable execution.

```bash
python gender_detector.py --image "C:\path\to\your\image.jpg" --device cpu
```

---

## Output Example

When a face is detected, the script runs the multi-crop calculations and logs a neat table:

```text
=======================================================
       GENDER ANALYSIS RESULTS: Aak4.jpeg
=======================================================
Crop Strategy             | Predicted  | Confidence
-------------------------------------------------------
Highly Cropped (Super)    | Male       | 97.40%
Cropped (Highly)          | Male       | 98.12%
Default                   | Male       | 96.55%
Hair Visible              | Male       | 99.10%
-------------------------------------------------------
ENSEMBLE AVERAGE (4 Crops)| Male       | 97.79%
=======================================================
Final Enum Value: GenderEnum.MALE
```

## Error Handling

- **No Face Detected**: If InsightFace does not find a face in the provided image, the script will log an error message (`no face detected`) and exit with status code `1`.
- **Missing File**: The script verifies that both the image and the model configuration are valid files before launching the heavy components.
