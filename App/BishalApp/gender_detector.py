import cv2
import time
import torch
import torch.nn as nn
import numpy as np
from insightface.app import FaceAnalysis

# 1. Redefine the exact model architecture
class GenderMLP(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gender_head = nn.Linear(hidden_dim, 2) # NUM_GENDER = 2
        
    def forward(self, x):
        x = self.shared(x)
        return self.gender_head(x)

# 2. Setup Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 3. Initialize and Load the Saved Model
# IMPORTANT: Use the exact hidden_dim and dropout that you used when saving the model
model = GenderMLP(hidden_dim=256, dropout=0.3879) 
model.load_state_dict(torch.load(r"C:\Users\shree\Documents\Veel\GenderDetection\EditedBishalGenderPredictionModel\BishalGenderPrediction.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval() # Set to evaluation mode!

# 4. Initialize InsightFace for embedding extraction
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0 if DEVICE == "cuda" else -1)

# 5. Prediction Function
def predict_loaded_model(img_path):
    # Load and resize image 
    img = cv2.imread(img_path)
    if img is None:
        return {"error": "Image could not be loaded. Check the file path."}
    
    # Resize to match your training preprocessing stability
    img = cv2.resize(img, (640, 640))
    
    # Extract faces
    faces = app.get(img)
    if len(faces) == 0:
        return {"error": "No face detected in the image."}
        
    # Grab the face with the highest detection score
    faces = sorted(faces, key=lambda x: x.det_score, reverse=True)
    embedding = faces[0].embedding
    
    # Convert embedding to a PyTorch tensor and send to device
    x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # Run the model forward pass
    with torch.no_grad():
        start_time = time.time()
        gender_logits = model(x)
        gender_probs = torch.softmax(gender_logits, dim=1)
        end_time = time.time()
        print(f"Total time taken: {(end_time - start_time):.7f}")
        confidence, pred_idx = gender_probs.max(1)
        
    # Convert results back to Python scalars
    pred_idx = pred_idx.item()
    confidence = confidence.item()
    
    # Format the final output
    return {
        "status": "SUCCESS",
        "prediction": "Male" if pred_idx == 1 else "Female",
        "confidence": round(confidence, 4)
    }


import os
import math
import matplotlib.pyplot as plt

def process_folder(folder_path):
    # Get all valid image files
    valid_extensions = ('.png', '.jpg', '.jpeg', '.avif', '.webp')
    image_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(valid_extensions)
    ]
    
    if not image_paths:
        print(f"No valid images found in {folder_path}")
        return
        
    cols = 3
    rows = math.ceil(len(image_paths) / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, img_path in enumerate(image_paths):
        ax = axes[idx]
        
        img = cv2.imread(img_path)
        
        if img is None:
            ax.set_title("LOAD FAILED", fontsize=10)
            ax.axis("off")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        result = predict_loaded_model(img_path)
        
        ax.imshow(img)
        ax.axis("off")
        
        if "error" in result:
            ax.set_title(result["error"], fontsize=10)
            continue
            
        gender = result["prediction"]
        conf = result["confidence"]
        
        ax.set_title(
            f"{gender} ({conf:.2f})",
            fontsize=12
        )
        
    # hide empty plots
    for i in range(len(image_paths), len(axes)):
        axes[i].axis("off")
        
    plt.tight_layout()
    
    # save output
    out_path = os.path.join(os.getcwd(), "gender_predictions_bishal.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    
    plt.show()
    
    print("Saved to:", out_path)

if __name__ == "__main__":
    # Specify the target directory containing images
    target_folder = r"C:\Users\shree\Documents\Veel\GenderDetection\test_images"
    process_folder(target_folder)
