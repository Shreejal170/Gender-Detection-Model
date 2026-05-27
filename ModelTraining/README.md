# Gender Detection Model Training Pipeline

This directory contains the training pipeline (`GenderDetectionModelTraining.ipynb`) used to train the EfficientNet-B0 backbone for the robust gender classification system. The pipeline is designed to be highly reproducible, leveraging PyTorch and modern deep learning practices to achieve optimal generalization on diverse facial data.

## 1. Datasets & Data Curation
The training pipeline unifies two prominent public datasets from Kaggle to expose the model to high variance in age, ethnicity, and lighting conditions:
- **Face-Age-Gender Dataset** (`aadyasingh55/face-age-gender-dataset`)
- **UTKFace Dataset** (`jangedoo/utkface-new`)

### Preprocessing & Splitting:
- **Label Alignment:** Gender labels are normalized across both datasets effectively establishing a canonical label space (`1` for Male, `0` for Female).
- **Stratified Partitioning:** The combined dataset (~58,000 images) is split into an 80/20 train/validation ratio using `scikit-learn`'s `train_test_split` with `stratify=gender_labels`, ensuring class distribution parity across folds.
- **Fault Tolerance:** Driven by a custom subclass of PyTorch's `Dataset` (`AdvancedGenderDataset`), the data loader includes exception capture routines to gracefully bypass anomalous or corrupted file streams (yielding zero-tensors) without halting the multi-threaded data fetching.

## 2. Augmentation & Pipeline Optimization
To prevent overfitting on the expansive parameter space, a dynamic augmentation policy is applied on-the-fly via `torchvision.transforms`:
- **Training Chain:** 
  - `Resize((224, 224))` (canonical ImageNet dimensions)
  - `RandomHorizontalFlip(p=0.5)`
  - `RandomRotation(degrees=15)`
  - `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`
  - `ToTensor()`
  - `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
- **Validation Chain:** Deterministic pipeline limited to resizing, un-scaled tensor conversion, and ImageNet standardization.

**Hardware Utilization:** Data is dispatched using `DataLoader` abstractions tuned for NVIDIA T4 instances:
- `batch_size=64` to maximize VRAM occupation.
- `num_workers=2` for uninterrupted pre-fetching.
- `pin_memory=True` locking page limits for accelerated native CPU-to-GPU memory transfers.

## 3. Architecture & Transfer Learning
The task leverages **EfficientNet-B0** as the backbone feature extractor, harnessing its state-of-the-art parameter efficiency via compound scaling.
- **Upstream Init:** Pre-trained on ImageNet-1K (`models.EfficientNet_B0_Weights.DEFAULT`), conferring robust low-level filters (Gabor structures, color blobs) out of the box.
- **Classifier Refactoring:** The native 1000-class fully-connected head is truncated. It is replaced with a `nn.Sequential` block:
  - `nn.Dropout(p=0.4, inplace=True)` to forcefully decorrelate representation dependencies.
  - `nn.Linear(num_ftrs, 2)` (dense layer projecting the 1,280-dimensional embedding space onto biological gender logits).

## 4. Optimization Dynamics & Regularization
A carefully constrained non-convex optimization loop handles parameter updates:
- **Loss Function:** `nn.CrossEntropyLoss()` functioning purely on continuous logits (softmax abstraction embedded within).
- **Optimizer:** `AdamW` (Adam with decoupled weight decay), configured with a `lr=1e-4` start (preserving ImageNet priors) and `weight_decay=0.01` extending rigorous L2 regularization.
- **Annealing (Learning Rate Scheduler):** `ReduceLROnPlateau` operates in `min` mode on the validation loss. If stagnation is detected for `patience=2` epochs, the macro-architecture dials down the learning rate scaling factor by `0.5`, enabling fine-grained convergence inside narrow minima.
- **Early Stopping & Checkpointing:** Governed by independent logic monitoring the hold-out validation landscape. With `patience=3`, if the network fails to register a generalization improvement, training aborts dynamically to preempt terminal overfitting. Ground-truth minimums are asynchronously committed directly to `best_gender_model.pth`.

## Artifacts
Running the notebook sequentially orchestrates the environment setup, dataset downloading via `kagglehub`, model distillation, and emits the final compiled `best_gender_model.pth` state dict, which is implicitly leveraged by the central deployment scripts.
