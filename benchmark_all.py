import os
import sys
import time
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from insightface.app import FaceAnalysis

# =====================================================================
# 1. Model Architectures
# =====================================================================

# Unedited Bishal Model (Multi-Task: Age + Gender)
class MultiTaskMLP(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.3): # Using hidden_dim=256 to match the weights!
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
        self.age_head = nn.Linear(hidden_dim, 7) # NUM_AGE = 7
        self.gender_head = nn.Linear(hidden_dim, 2) # NUM_GENDER = 2

    def forward(self, x):
        x = self.shared(x)
        age_logits = self.age_head(x)
        gender_logits = self.gender_head(x)
        return age_logits, gender_logits

# Register MultiTaskMLP in __main__ namespace for PyTorch pickling safety
import __main__
__main__.MultiTaskMLP = MultiTaskMLP

# Edited Bishal Model (Gender Only)
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
        self.gender_head = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        x = self.shared(x)
        return self.gender_head(x)

# Helper function for crops extraction
def extract_crops_local(img_bgr, face):
    h, w, _ = img_bgr.shape
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    box_height, box_width = y2 - y1, x2 - x1
    crops = []
    
    # 1. Super Highly Cropped
    shc_y1, shc_y2 = max(0, y1 + int(box_height * 0.30)), min(h, y2 - int(box_height * 0.15))
    shc_x1, shc_x2 = max(0, x1 + int(box_width * 0.20)), min(w, x2 - int(box_width * 0.20))
    crops.append(Image.fromarray(cv2.cvtColor(img_bgr[shc_y1:shc_y2, shc_x1:shc_x2], cv2.COLOR_BGR2RGB)))
    
    # 2. Highly Cropped
    hc_y1, hc_y2 = max(0, y1 + int(box_height * 0.12)), min(h, y2 - int(box_height * 0.05))
    hc_x1, hc_x2 = max(0, x1 + int(box_width * 0.05)), min(w, x2 - int(box_width * 0.05))
    crops.append(Image.fromarray(cv2.cvtColor(img_bgr[hc_y1:hc_y2, hc_x1:hc_x2], cv2.COLOR_BGR2RGB)))
    
    # 3. Default
    df_y1, df_y2 = max(0, y1), min(h, y2)
    df_x1, df_x2 = max(0, x1), min(w, x2)
    crops.append(Image.fromarray(cv2.cvtColor(img_bgr[df_y1:df_y2, df_x1:df_x2], cv2.COLOR_BGR2RGB)))
    
    # 4. Expanded (Hair Visible)
    ex_y1, ex_y2 = max(0, y1 - int(box_height * 0.45)), min(h, y2 + int(box_height * 0.25))
    ex_x1, ex_x2 = max(0, x1 - int(box_width * 0.40)), min(w, x2 + int(box_width * 0.40))
    crops.append(Image.fromarray(cv2.cvtColor(img_bgr[ex_y1:ex_y2, ex_x1:ex_x2], cv2.COLOR_BGR2RGB)))
    
    return crops

# =====================================================================
# 2. Main Benchmarking Logic
# =====================================================================

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on DEVICE: {DEVICE}")
    
    workspace = r"c:\Users\shree\Documents\Veel\GenderDetection"
    
    # Paths
    edited_weights = os.path.join(workspace, "PredictionModel", "EditedBishalGenderPredictionModel", "BishalGenderPrediction.pth")
    shreejal_weights = os.path.join(workspace, "PredictionModel", "ShreejalGenderPredictionModel", "gender_prediction_model.pth")
    test_images_dir = os.path.join(workspace, "test_images")
    
    # 1. Load Shreejal's model
    print("\nLoading Shreejal's model...")
    shreejal_model = models.efficientnet_b0(weights=None)
    num_ftrs = shreejal_model.classifier[1].in_features
    shreejal_model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_ftrs, 2)
    )
    shreejal_model.load_state_dict(torch.load(shreejal_weights, map_location=DEVICE))
    shreejal_model.to(DEVICE)
    shreejal_model.eval()
    
    shreejal_app = FaceAnalysis(allowed_modules=['detection'])
    shreejal_app.prepare(ctx_id=-1, det_size=(640, 640))
    
    shreejal_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Load Edited Bishal's model
    print("Loading Edited Bishal's model...")
    edited_model = GenderMLP(hidden_dim=256, dropout=0.3879)
    edited_model.load_state_dict(torch.load(edited_weights, map_location=DEVICE))
    edited_model.to(DEVICE)
    edited_model.eval()
    
    # Shared FaceAnalysis buffalo_l for Bishal models
    bishal_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    bishal_app.prepare(ctx_id=-1)
    
    # 3. Load Unedited Bishal's model using the same weights file (strict=False)
    print("Loading Unedited Bishal's multitask model (struct only, strict=False)...")
    unedited_model = MultiTaskMLP(hidden_dim=256, dropout=0.3879)
    unedited_model.load_state_dict(torch.load(edited_weights, map_location=DEVICE), strict=False)
    unedited_model.to(DEVICE)
    unedited_model.eval()
    
    models_to_run = ["shreejal", "edited_bishal", "unedited_bishal"]
    
    # 4. Collect and parse test files
    valid_extensions = ('.png', '.jpg', '.jpeg', '.avif', '.webp')
    image_files = sorted([
        f for f in os.listdir(test_images_dir)
        if f.lower().endswith(valid_extensions)
    ])
    
    dataset = []
    for f in image_files:
        path = os.path.join(test_images_dir, f)
        if "confusing" in f.lower():
            true_lbl, true_gender_str, is_confusing = 1, "Male", True
        else:
            name_part = os.path.splitext(f)[0]
            parts = name_part.split('_')
            if len(parts) >= 3:
                gender_code = parts[-1]
                if gender_code == '0':
                    true_lbl, true_gender_str, is_confusing = 0, "Female", False
                elif gender_code == '1':
                    true_lbl, true_gender_str, is_confusing = 1, "Male", False
                else:
                    continue
            else:
                continue
        dataset.append({
            "filename": f,
            "path": path,
            "true_label": true_lbl,
            "true_gender": true_gender_str,
            "is_confusing": is_confusing
        })
        
    print(f"Found {len(dataset)} images in test_images folder.")
    
    # Warmup
    print("Performing warmup...")
    warmup_img = dataset[0]["path"]
    for _ in range(5):
        # Shreejal warmup
        img_b = cv2.imread(warmup_img)
        faces_s = shreejal_app.get(img_b)
        if faces_s:
            crops = extract_crops_local(img_b, faces_s[0])
            for c_img in crops:
                tensor = shreejal_transform(c_img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    shreejal_model(tensor)
                    
        # Bishal warmup
        img_b = cv2.resize(img_b, (640, 640))
        faces_b = bishal_app.get(img_b)
        if faces_b:
            emb = faces_b[0].embedding
            x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                edited_model(x)
                unedited_model(x)
                    
    # Benchmarking parameters
    N_ITERATIONS = 15
    print(f"Running benchmark over {len(dataset)} images for {N_ITERATIONS} iterations (total {len(dataset) * N_ITERATIONS} inferences per model)...")
    
    results = {m: {item["filename"]: [] for item in dataset} for m in models_to_run}
    
    for iteration in range(N_ITERATIONS):
        print(f"Iteration {iteration + 1}/{N_ITERATIONS}...")
        for item in dataset:
            f = item["filename"]
            img_path = item["path"]
            
            # 1. Shreejal Model
            t_start_pipe = time.time()
            img_bgr = cv2.imread(img_path)
            t_start_det = time.time()
            faces_s = shreejal_app.get(img_bgr)
            t_det = time.time() - t_start_det
            
            if faces_s:
                primary_face = faces_s[0]
                crops = extract_crops_local(img_bgr, primary_face)
                all_probs = []
                t_model = 0.0
                for c_img in crops:
                    t_start_m = time.time()
                    tensor = shreejal_transform(c_img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        outputs = shreejal_model(tensor)
                        probs = torch.softmax(outputs, dim=1)
                    t_model += (time.time() - t_start_m)
                    all_probs.append(probs.squeeze().cpu().numpy())
                avg_probs = np.mean(all_probs, axis=0)
                pred_idx = np.argmax(avg_probs)
                conf = avg_probs[pred_idx]
                results["shreejal"][f].append({
                    "prediction": int(pred_idx),
                    "confidence": float(conf),
                    "pipeline_time": time.time() - t_start_pipe,
                    "det_time": t_det,
                    "model_time": t_model
                })
            else:
                results["shreejal"][f].append({
                    "error": "No face detected",
                    "pipeline_time": time.time() - t_start_pipe,
                    "det_time": t_det,
                    "model_time": 0.0
                })
                
            # 2. Edited Bishal & Unedited Bishal (share face detection)
            t_start_pipe = time.time()
            img_resized = cv2.resize(img_bgr, (640, 640))
            t_start_det = time.time()
            faces_b = bishal_app.get(img_resized)
            t_det = time.time() - t_start_det
            
            if faces_b:
                faces_b = sorted(faces_b, key=lambda x: x.det_score, reverse=True)
                embedding = faces_b[0].embedding
                x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                # Edited inference
                t_start_m = time.time()
                with torch.no_grad():
                    logits_e = edited_model(x)
                    probs_e = torch.softmax(logits_e, dim=1)
                    conf_e, pred_e = probs_e.max(1)
                t_model_e = time.time() - t_start_m
                results["edited_bishal"][f].append({
                    "prediction": pred_e.item(),
                    "confidence": conf_e.item(),
                    "pipeline_time": (time.time() - t_start_pipe),
                    "det_time": t_det,
                    "model_time": t_model_e
                })
                
                # Unedited inference (MultiTaskMLP)
                t_start_m = time.time()
                with torch.no_grad():
                    age_logits, gender_logits = unedited_model(x)
                    probs_u = torch.softmax(gender_logits, dim=1)
                    conf_u, pred_u = probs_u.max(1)
                t_model_u = time.time() - t_start_m
                results["unedited_bishal"][f].append({
                    "prediction": pred_u.item(),
                    "confidence": conf_u.item(),
                    "pipeline_time": (time.time() - t_start_pipe) + t_model_u - t_model_e, # adjust pipeline time
                    "det_time": t_det,
                    "model_time": t_model_u
                })
            else:
                err_data = {
                    "error": "No face detected",
                    "pipeline_time": time.time() - t_start_pipe,
                    "det_time": t_det,
                    "model_time": 0.0
                }
                results["edited_bishal"][f].append(err_data)
                results["unedited_bishal"][f].append(err_data)
                    
    # 5. Analyze Results
    print("\nBenchmark Complete! Analyzing metrics...")
    
    metrics = {}
    for m_key in models_to_run:
        correct_count = 0
        total_count = 0
        female_correct = 0
        female_total = 0
        male_correct = 0
        male_total = 0
        
        all_pipeline_times = []
        all_det_times = []
        all_model_times = []
        
        model_detailed_info = {}
        
        for item in dataset:
            f = item["filename"]
            true_lbl = item["true_label"]
            is_confusing = item["is_confusing"]
            
            run_data = results[m_key][f]
            if not run_data:
                continue
                
            preds = [r["prediction"] for r in run_data if "error" not in r]
            confs = [r["confidence"] for r in run_data if "error" not in r]
            p_times = [r["pipeline_time"] for r in run_data]
            d_times = [r["det_time"] for r in run_data]
            m_times = [r["model_time"] for r in run_data]
            
            if not preds:
                pred_label = -1
                avg_conf = 0.0
            else:
                pred_label = max(set(preds), key=preds.count)
                avg_conf = np.mean(confs)
                
            avg_p_time = np.mean(p_times)
            avg_d_time = np.mean(d_times)
            avg_m_time = np.mean(m_times)
            
            all_pipeline_times.extend(p_times)
            all_det_times.extend(d_times)
            all_model_times.extend(m_times)
            
            is_correct = (pred_label == true_lbl)
            
            model_detailed_info[f] = {
                "true_label": true_lbl,
                "predicted_label": pred_label,
                "confidence": avg_conf,
                "is_correct": is_correct,
                "avg_pipeline_time": avg_p_time,
                "avg_det_time": avg_d_time,
                "avg_model_time": avg_m_time,
                "is_confusing": is_confusing
            }
            
            if not is_confusing:
                total_count += 1
                if is_correct:
                    correct_count += 1
                if true_lbl == 0:
                    female_total += 1
                    if is_correct:
                        female_correct += 1
                else:
                    male_total += 1
                    if is_correct:
                        male_correct += 1
            else:
                metrics[m_key + "_confusing"] = {
                    "filename": f,
                    "true_label": true_lbl,
                    "predicted_label": pred_label,
                    "predicted_str": "Male" if pred_label == 1 else "Female" if pred_label == 0 else "No Face",
                    "confidence": avg_conf,
                    "is_correct": is_correct,
                    "avg_pipeline_time": avg_p_time
                }
                
        # Overall standard metrics
        accuracy = correct_count / total_count if total_count > 0 else 0
        female_acc = female_correct / female_total if female_total > 0 else 0
        male_acc = male_correct / male_total if male_total > 0 else 0
        
        tp = sum(1 for f, d in model_detailed_info.items() if not d["is_confusing"] and d["true_label"] == 1 and d["predicted_label"] == 1)
        fp = sum(1 for f, d in model_detailed_info.items() if not d["is_confusing"] and d["true_label"] == 0 and d["predicted_label"] == 1)
        fn = sum(1 for f, d in model_detailed_info.items() if not d["is_confusing"] and d["true_label"] == 1 and d["predicted_label"] == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Model efficiency
        if m_key == "shreejal":
            model_obj = shreejal_model
            num_params = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)
            disk_size = os.path.getsize(shreejal_weights) / (1024 * 1024)
        elif m_key == "edited_bishal":
            model_obj = edited_model
            num_params = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)
            disk_size = os.path.getsize(edited_weights) / (1024 * 1024)
        else:
            model_obj = unedited_model
            num_params = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)
            disk_size = os.path.getsize(edited_weights) / (1024 * 1024) # unedited loads the same file
            
        metrics[m_key] = {
            "overall": {
                "accuracy": accuracy,
                "female_accuracy": female_acc,
                "male_accuracy": male_acc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "avg_pipeline_time_ms": np.mean(all_pipeline_times) * 1000,
                "avg_det_time_ms": np.mean(all_det_times) * 1000,
                "avg_model_time_ms": np.mean(all_model_times) * 1000,
                "num_parameters": num_params,
                "disk_size_mb": disk_size
            },
            "detailed": model_detailed_info
        }
        
    # Write to a JSON file
    with open("benchmark_results_all.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\n" + "="*90)
    print("                    THREE-MODEL BENCHMARK SUMMARY REPORT")
    print("="*90)
    print(f"{'Metric':<32} | {'Shreejal (EffNet)':<17} | {'Bishal (Edited)':<17} | {'Bishal (Unedited)':<17}")
    print("-"*90)
    
    s_o = metrics["shreejal"]["overall"]
    e_o = metrics["edited_bishal"]["overall"]
    u_o = metrics["unedited_bishal"]["overall"]
    
    print(f"{'Accuracy (excl. confusing)':<32} | {s_o['accuracy']*100:.2f}%           | {e_o['accuracy']*100:.2f}%           | {u_o['accuracy']*100:.2f}%")
    print(f"{'Female Accuracy':<32} | {s_o['female_accuracy']*100:.2f}%           | {e_o['female_accuracy']*100:.2f}%           | {u_o['female_accuracy']*100:.2f}%")
    print(f"{'Male Accuracy':<32} | {s_o['male_accuracy']*100:.2f}%           | {e_o['male_accuracy']*100:.2f}%           | {u_o['male_accuracy']*100:.2f}%")
    print(f"{'Precision (Male)':<32} | {s_o['precision']:.4f}            | {e_o['precision']:.4f}            | {u_o['precision']:.4f}")
    print(f"{'Recall (Male)':<32} | {s_o['recall']:.4f}            | {e_o['recall']:.4f}            | {u_o['recall']:.4f}")
    print(f"{'F1 Score (Male)':<32} | {s_o['f1_score']:.4f}            | {e_o['f1_score']:.4f}            | {u_o['f1_score']:.4f}")
    print("-"*90)
    
    s_c = metrics["shreejal_confusing"]
    e_c = metrics["edited_bishal_confusing"]
    u_c = metrics["unedited_bishal_confusing"]
    
    print(f"{'Confusing Face Prediction':<32} | {s_c['predicted_str']} ({s_c['confidence']:.1f}%)  | {e_c['predicted_str']} ({e_c['confidence']*100:.1f}%) | {u_c['predicted_str']} ({u_c['confidence']*100:.1f}%)")
    print(f"{'Confusing Face Correct?':<32} | {'YES' if s_c['is_correct'] else 'NO':<17} | {'YES' if e_c['is_correct'] else 'NO':<17} | {'YES' if u_c['is_correct'] else 'NO':<17}")
    print("-"*90)
    
    print(f"{'Avg Total Latency (ms)':<32} | {s_o['avg_pipeline_time_ms']:.2f} ms          | {e_o['avg_pipeline_time_ms']:.2f} ms          | {u_o['avg_pipeline_time_ms']:.2f} ms")
    print(f"{'Avg Model Forward (ms)':<32} | {s_o['avg_model_time_ms']:.2f} ms          | {e_o['avg_model_time_ms']:.2f} ms            | {u_o['avg_model_time_ms']:.2f} ms")
    print(f"{'Throughput (FPS)':<32} | {1000.0 / s_o['avg_pipeline_time_ms']:.2f} FPS            | {1000.0 / e_o['avg_pipeline_time_ms']:.2f} FPS            | {1000.0 / u_o['avg_pipeline_time_ms']:.2f} FPS")
    print("-"*90)
    
    print(f"{'Model Parameters':<32} | {s_o['num_parameters']:,}        | {e_o['num_parameters']:,}          | {u_o['num_parameters']:,}")
    print(f"{'Weights Size on Disk':<32} | {s_o['disk_size_mb']:.3f} MB          | {e_o['disk_size_mb']:.3f} MB           | {u_o['disk_size_mb']:.3f} MB")
    print("="*90)

if __name__ == "__main__":
    main()
