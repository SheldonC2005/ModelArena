# 🎯 ModelArena - Deepfake Detection Competition

**Mission:** Binary video classification for deepfake detection using InceptionResNetV2 + Attention Pooling

![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Local%20GPU-orange)
![Model](https://img.shields.io/badge/Model-InceptionResNetV2%20%2B%20Attention-blue)

## 📊 Project Overview

This repository contains a complete deepfake detection solution using **InceptionResNetV2 + Attention Pooling** architecture. The model was trained on an RTX 3050 Laptop GPU (4GB VRAM) with mixed precision training and gradient accumulation to maximize performance within hardware constraints.

### Key Features

- 🤖 **Single Model Architecture:** InceptionResNetV2 + Attention Pooling
- 🎯 **Attention Mechanism:** Learns to focus on suspicious frames
- 📊 **70:30 Train/Val Split:** 420 training, 180 validation videos
- 🔍 **Hyperparameter Optimization:** Automated tuning with Optuna
- 🎯 **Test-Time Augmentation:** 5 variants per prediction (h-flip, shift-scale, brightness±0.1)
- ⚡ **Mixed Precision Training:** FP16 for 4GB VRAM efficiency
- 🔄 **Gradient Accumulation:** Simulates batch_size=4
- 🛡️ **Hardware Safety:** GPU thermal monitoring, VRAM tracking, OOM fallback
- 📦 **Complete Pipeline:** From training to final predictions

### Architecture

```
Video → Frame Extraction → InceptionResNetV2 → Attention Pooling → Classification
        (24/32/48 frames)    (1536 features)    (learns weights)    (Real/Fake)
                                                                        
      TTA (5 variants) → Ensemble Average → Final Prediction
```

## 📁 Project Structure

```
ModelArena/
├── TRAINING_PIPELINE.ipynb          # Complete training notebook with Optuna
├── INFERENCE_PIPELINE.ipynb         # Inference with TTA & VRAM management
├── README.md                        # This file
├── requirements.txt                 # Python dependencies (Colab/cloud)
├── requirements-local.txt           # Local GPU dependencies (RTX 3050)
├── archive/                         # Dataset
│   ├── train_labels.csv            # 600 video labels
│   ├── test_public.csv             # 200 test videos
│   ├── train/
│   │   ├── fake/  (300 videos)
│   │   └── real/  (300 videos)
│   └── test/      (200 videos)
├── models/                          # Saved model checkpoints (auto-created)
│   ├── inception_resnet_v2_best.pt
│   ├── training_summary.json
│   └── training_curves.png
└── SUBMISSION_DIRECTORY/            # Final submission files
    └── PREDICTIONS.CSV              # Test predictions (filename,label,probability)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12.10
- CUDA 11.2+ Toolkit (for local GPU training)
- RTX 3050 Laptop GPU (4GB VRAM) or better
- ~700MB dataset storage
- ~2GB for model checkpoints

### Local GPU Training Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/SheldonC2005/ModelArena.git
   cd ModelArena
   ```

2. **Install Dependencies**
   ```bash
   # For local GPU (RTX 3050)
   pip install -r requirements-local.txt
   
   # For Google Colab (alternative)
   pip install -r requirements.txt
   ```

3. **Prepare Dataset**
   - Ensure `archive/` folder structure matches above
   - Verify 600 training videos (300 fake + 300 real)
   - Verify 200 test videos

4. **Train Model**
   - Open `TRAINING_PIPELINE.ipynb` in Jupyter/VS Code
   - Run all cells sequentially
   - Training takes ~6-8 hours on RTX 3050
   - Model auto-saves to `models/inception_resnet_v2_best.pt`

5. **Generate Predictions**
   - Open `INFERENCE_PIPELINE.ipynb`
   - Run all cells (~30-45 minutes with TTA)
   - Predictions saved to `SUBMISSION_DIRECTORY/PREDICTIONS.CSV`

## 🎓 Model Specifications

### InceptionResNetV2 + Attention Pooling

| Component | Configuration |
|-----------|--------------|
| **CNN Backbone** | InceptionResNetV2 (pretrained on ImageNet) |
| **Input Size** | 299×299×3 |
| **Feature Dimension** | 1536 |
| **Attention Mechanism** | FC(1536→256)→Tanh→FC(256→1)→Softmax |
| **Attention Dimension** | 256 (tuned by Optuna) |
| **Classifier** | FC(1536→256)→ReLU→Dropout(0.5)→FC(256→2) |
| **Total Parameters** | ~55M |

### Training Configuration

**Dataset Split:**
- Training: 420 videos (70%)
- Validation: 180 videos (30%)
- Test: 200 videos (unlabeled)

**Hardware Optimization (RTX 3050 4GB VRAM):**
- Batch Size: 1 (physical)
- Gradient Accumulation Steps: 4 (effective batch_size=4)
- Mixed Precision: FP16 (AMP)
- Frames per Video: 24 (optimized via frame testing on 10 videos)

**Training Hyperparameters (Optuna-tuned):**
- Learning Rate: ~1e-4 to 1e-3 (tuned)
- Optimizer: AdamW (weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau (patience=2)
- FC Dropout: 0.3 to 0.7 (tuned)
- Max Epochs: 30
- Early Stopping: Patience=3

**Safety Features:**
- GPU Temperature Monitoring: Pause at 78°C, resume at 72°C
- VRAM Usage Tracking: Warning at 90%
- Automatic Cache Clearing: After each video + 90% emergency threshold

### Inference Configuration

**Frame Selection:**
- Auto-tested: {24, 32, 48} frames on 10 sample videos
- Recommendation: Highest average confidence wins
- Default: 24 frames (balanced speed/accuracy)

**Test-Time Augmentation (5 variants):**
1. Original (no augmentation)
2. HorizontalFlip
3. ShiftScaleRotate(shift=0.05, scale=0.05)
4. RandomBrightnessContrast(brightness=+0.1)
5. RandomBrightnessContrast(brightness=-0.1)

**OOM Fallback Strategy:**
- Primary: Use recommended frames_per_video
- Fallback 1: Retry after cache clear
- Fallback 2: Reduce to 16 frames
- Fallback 3: Use default frames (skip video as last resort)

**Output Format:**
```csv
filename,label,probability
video1.mp4,1,0.9234
video2.mp4,0,0.8765
```
- `label`: 0 (Real) or 1 (Fake)
- `probability`: Confidence score of predicted class (softmax output)

## 📈 Training Timeline & Performance

### Training Time (RTX 3050 Laptop 4GB VRAM)

| Phase | Duration | Description |
|-------|----------|-------------|
| Environment Setup | 5 min | GPU check, imports, path verification |
| Optuna Hyperparameter Tuning | 30 min | 10 trials, 5 epochs each |
| Full Training (30 epochs) | 6-8 hrs | With early stopping, thermal pauses |
| Model Saving & Visualization | 2 min | Save checkpoint, plot curves |
| **Total Training** | **~7-9 hrs** | Depends on GPU temperature |

### Inference Time (RTX 3050 4GB VRAM)

| Phase | Duration | Description |
|-------|----------|-------------|
| Model Loading | 30 sec | Load checkpoint, verify architecture |
| Frame Testing (10 videos) | 5 min | Test 24/32/48 frames, recommend best |
| Inference with TTA (200 videos) | 30-45 min | 5 TTA variants per video |
| CSV Generation & Validation | 1 min | Format check, save predictions |
| **Total Inference** | **~35-50 min** | With TTA and safety features |

### Expected Accuracy

- **Validation Set (180 videos):** 80-92%
- **Test Set (200 videos):** 75-90% (depends on test difficulty)
- **With TTA:** +2-5% improvement over single-pass inference

## 🔧 Dependencies

### For Local GPU Training (RTX 3050)

```bash
pip install -r requirements-local.txt
```

**Key packages:**
```
# Deep Learning
torch==2.2.0+cu118
torchvision==0.17.0+cu118
timm==0.9.12

# Computer Vision
opencv-python-headless==4.9.0.80
albumentations==1.3.1

# Hyperparameter Optimization
optuna==3.5.0

# Hardware Monitoring
pynvml==11.5.0
GPUtil==1.4.0
psutil==5.9.7

# Data Processing
pandas==2.1.4
numpy==1.26.3
scikit-learn==1.3.2
```

### For Cloud Training (Google Colab)

```bash
pip install -r requirements.txt
```

See [requirements-local.txt](requirements-local.txt) or [requirements.txt](requirements.txt) for complete lists.

## 💡 Key Implementation Details

### 1. Attention Pooling Mechanism

Instead of averaging all frame features equally, attention learns which frames are most important:

```python
# Attention scores for each frame
attn_scores = attention_network(frame_features)  # [batch, num_frames, 1]
attn_weights = softmax(attn_scores, dim=1)       # [batch, num_frames, 1]

# Weighted sum focuses on "suspicious" frames
video_features = sum(frame_features * attn_weights)
```

**Advantages:**
- Automatically focuses on manipulated regions
- Robust to irrelevant background frames
- Learned end-to-end during training

### 2. On-the-Fly Frame Extraction

Videos are **not pre-processed**. Frames are extracted during training/inference:

```python
# Uniformly sample frames from video
indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
frames = [read_frame(video, idx) for idx in indices]
```

**Benefits:**
- No storage overhead for extracted frames
- Flexible frame sampling strategies
- Easy to experiment with different frame counts

### 3. Mixed Precision Training (FP16)

Uses automatic mixed precision (AMP) for 4GB VRAM efficiency:

```python
with autocast(enabled=True):
    outputs = model(frames)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()  # Scales gradients to prevent underflow
scaler.step(optimizer)         # Unscales before optimizer step
```

**Benefits:**
- ~2× faster training
- ~50% less VRAM usage
- Maintains accuracy with gradient scaling

### 4. Gradient Accumulation

Simulates larger batch sizes on limited VRAM:

```python
for i, (frames, labels) in enumerate(dataloader):
    loss = compute_loss(frames, labels)
    loss = loss / accumulation_steps  # Scale loss
    loss.backward()  # Accumulate gradients
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()    # Update weights
        optimizer.zero_grad()  # Reset gradients
```

**Configuration:**
- Physical batch_size: 1
- Accumulation steps: 4
- Effective batch_size: 4

### 5. Hardware Safety Monitoring

**GPU Temperature Protection:**
```python
if gpu_temp >= 78°C:
    pause_training()
    wait_until(gpu_temp <= 72°C)
    resume_training()
```

**VRAM Management:**
```python
if vram_usage > 90%:
    torch.cuda.empty_cache()
    warning("High VRAM usage detected")
```

### 6. Test-Time Augmentation (TTA)

Average predictions from 5 augmented versions:

```python
tta_transforms = [
    Original(),
    HorizontalFlip(),
    ShiftScaleRotate(),
    BrightnessUp(),
    BrightnessDown()
]

predictions = [model(transform(video)) for transform in tta_transforms]
final_prediction = average(predictions)
```

**Impact:** +2-5% accuracy improvement

## ⚠️ Common Issues & Solutions

### GPU Out of Memory (OOM)

**Inference Pipeline has built-in OOM fallback:**
1. Retry with cache clear
2. Reduce frames to 16
3. Use default frames (skip video)

**For training:**
```python
# Reduce batch size (already 1)
# Reduce frames per video
FRAMES_PER_VIDEO = 16  # Instead of 24
```

### GPU Overheating

**Built-in protection:**
- Training pauses automatically at 78°C
- Resumes when temperature drops to 72°C

**Manual mitigation:**
- Increase laptop cooling (elevated stand, external fan)
- Reduce room temperature
- Lower power limit if necessary

### CUDA Out of Memory During Inference

**The inference pipeline handles this automatically:**
- Emergency cache clearing at 90% VRAM
- Frame reduction fallback
- Per-video cache clearing

### Slow Training

**Check these:**
- ✅ GPU is detected: `torch.cuda.is_available()` → True
- ✅ Mixed precision enabled: `MIXED_PRECISION = True`
- ✅ Thermal throttling not active (check GPU temp)
- ✅ Videos loading correctly from `archive/` folder

### Wrong Prediction Format

**Verify CSV format:**
```python
import pandas as pd
df = pd.read_csv('SUBMISSION_DIRECTORY/PREDICTIONS.CSV')
print(df.head())

# Must have exactly these columns:
# filename,label,probability
# video1.mp4,1,0.9234
```

### Notebook Kernel Crashes

**In VS Code / Jupyter:**
- Restart kernel: Clear outputs, restart, run again
- Check VRAM: Close other GPU applications
- Reduce `num_workers` to 0 in DataLoader

## 📚 File Descriptions

### Main Notebooks

- **TRAINING_PIPELINE.ipynb** (26 cells, 1355 lines)
  - Complete training workflow from scratch
  - GPU verification and environment setup
  - Optuna hyperparameter tuning (10 trials)
  - Full model training with early stopping
  - Hardware safety monitoring (thermal, VRAM)
  - Training visualization and metrics
  - Auto-saves best model to `models/`

- **INFERENCE_PIPELINE.ipynb** (26 cells, 879 lines)
  - Model loading with architecture verification
  - Adaptive frame testing (24/32/48 on 10 videos)
  - Test-Time Augmentation (5 variants)
  - OOM fallback strategy (3-tier)
  - VRAM monitoring and cache management
  - CSV generation with validation
  - Auto-saves to `SUBMISSION_DIRECTORY/PREDICTIONS.CSV`

### Configuration Files

- **requirements-local.txt:** Local GPU training (CUDA 11.8, RTX 3050)
- **requirements.txt:** Cloud training (Google Colab, generic)
- **README.md:** This documentation

### Utilities

- **check_video_props.py:** Inspect video properties (resolution, FPS, frames)
- **verify_setup.py:** Verify project structure before training

## 🎯 Final Output

### PREDICTIONS.CSV Format

```csv
filename,label,probability
777fb79ca0d24d6aa35fae379377a1c2.mp4,1,0.9234
6c90fd9736384fe583272c45abd3b0ef.mp4,0,0.8765
3d0c2c52f3ff425ca4765a335352c709.mp4,1,0.9567
...
```

**Columns:**
- `filename`: Video filename from test set
- `label`: Predicted class (0=Real, 1=Fake)
- `probability`: Confidence score of predicted class (0.0-1.0)

**Validation:**
- ✅ 200 rows (one per test video)
- ✅ No missing values
- ✅ Probabilities in [0.0, 1.0]
- ✅ Labels are 0 or 1
- ✅ All test filenames present

### Model Checkpoint

**Location:** `models/inception_resnet_v2_best.pt`

**Contents:**
```python
checkpoint = {
    'epoch': best_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_acc': best_validation_accuracy,
    'val_f1': validation_f1_score,
    'val_auc': validation_auc,
    'history': training_history,
    'config': model_configuration
}
```

### Training Summary

**Location:** `models/training_summary.json`

Contains complete training metrics, hyperparameters, and performance history.

## 🏆 Best Practices & Tips

### Training

1. **Monitor GPU Temperature:** Built-in protection, but ensure good cooling
2. **Check Training Curves:** Look for overfitting (val_loss increasing while train_loss decreasing)
3. **Optuna Tuning:** Don't skip - significant performance boost
4. **Early Stopping:** Let it run, patience=3 is optimal
5. **Save Checkpoints:** Model auto-saves, but keep backups

### Inference

1. **Use TTA:** Significant accuracy boost (~2-5%)
2. **Frame Testing:** Let it run on 10 videos to optimize performance
3. **Monitor VRAM:** Pipeline handles it, but keep Task Manager open
4. **Verify Predictions:** Check distribution (~50/50 if balanced test set)
5. **CSV Validation:** Built-in checks, but manually verify format

### Hardware Optimization

1. **RTX 3050 Tips:**
   - Close all other applications
   - Use `num_workers=0` (Windows compatibility)
   - Enable mixed precision (FP16)
   - Monitor temperature during long sessions

2. **VRAM Management:**
   - Batch size: 1 (already optimized)
   - Gradient accumulation: 4 steps
   - Cache clearing: Automatic after each video

3. **Thermal Management:**
   - Laptop on elevated stand
   - External cooling fan
   - Ambient temperature < 25°C optimal

### Debugging

1. **Check GPU:** `torch.cuda.is_available()` → True
2. **Verify Dataset:** 300 fake + 300 real + 200 test videos
3. **Test One Video:** Run inference on single video before full batch
4. **Read Logs:** Training logs saved to `logs/` directory
5. **Compare Outputs:** Check `models/training_summary.json`

## 📖 Technical Stack & Resources

### Core Technologies

- **Deep Learning:** PyTorch 2.2.0 (CUDA 11.8)
- **CNN Architecture:** InceptionResNetV2 (timm library)
- **Attention Mechanism:** Custom implementation
- **Hyperparameter Optimization:** Optuna 3.5.0
- **Computer Vision:** OpenCV 4.9.0, Albumentations 1.3.1
- **Hardware Monitoring:** pynvml, GPUtil, psutil

### Learning Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Timm Models](https://github.com/huggingface/pytorch-image-models)
- [Albumentations](https://albumentations.ai/docs/)
- [Optuna Tutorials](https://optuna.readthedocs.io/en/stable/)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [Attention Mechanisms](https://arxiv.org/abs/1409.0473)

### Hardware Requirements

**Minimum (Inference Only):**
- GPU: 4GB VRAM (RTX 3050, GTX 1650)
- RAM: 8GB
- Storage: 2GB
- OS: Windows 10/11, Linux

**Recommended (Training + Inference):**
- GPU: 6GB+ VRAM (RTX 3060, RTX 4060)
- RAM: 16GB
- Storage: 5GB
- OS: Windows 10/11, Linux
- Cooling: External fan recommended

---

## 📊 Project Timeline

**Development:** January 2026 - February 2026

**Phase 1: Research & Planning** (Week 1)
- ✅ Dataset analysis (600 training, 200 test videos)
- ✅ Architecture selection (InceptionResNetV2 + Attention)
- ✅ Hardware optimization strategy (RTX 3050 4GB VRAM)

**Phase 2: Implementation** (Week 2-3)
- ✅ Training pipeline (26 cells, 1355 lines)
- ✅ Hyperparameter tuning with Optuna
- ✅ Hardware safety features (thermal, VRAM)
- ✅ Gradient accumulation for 4GB VRAM

**Phase 3: Training & Optimization** (Week 3-4)
- ✅ Model training (70:30 split)
- ✅ Validation and performance analysis
- ✅ Training curves and metrics visualization

**Phase 4: Inference Pipeline** (Week 4)
- ✅ Inference implementation (26 cells, 879 lines)
- ✅ Test-Time Augmentation (5 variants)
- ✅ OOM fallback strategy
- ✅ Frame testing and optimization

**Phase 5: Testing & Validation** (Week 4)
- ✅ Predictions generation (200 test videos)
- ✅ CSV validation (4 checks)
- ✅ Final verification

---

## 🤝 Contributing

This project is complete and ready for use. For improvements or adaptations:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Test changes thoroughly (training + inference)
4. Commit changes (`git commit -am 'Add improvement'`)
5. Push to branch (`git push origin feature/improvement`)
6. Create Pull Request

**Areas for potential improvement:**
- Additional CNN architectures
- Ensemble methods
- Advanced augmentation strategies
- Distributed training support
- Model compression/quantization

---

## 📄 License

MIT License - Free to use for learning, research, and competitions.

---

## 👤 Author

**SheldonC2005**
- GitHub: [@SheldonC2005](https://github.com/SheldonC2005)
- Repository: [ModelArena](https://github.com/SheldonC2005/ModelArena)

---

## 🙏 Acknowledgments

- **Dataset:** IEEECS VIT ML Competition
- **Pre-trained Models:** [Timm library](https://github.com/huggingface/pytorch-image-models) by Ross Wightman
- **PyTorch Team:** For excellent deep learning framework
- **Optuna Team:** For hyperparameter optimization framework
- **Albumentations Team:** For efficient augmentation library

---

## 📝 Citation

If you use this code for research or publications, please cite:

```bibtex
@misc{modelarena2026,
  author = {SheldonC2005},
  title = {ModelArena: Deepfake Detection with InceptionResNetV2 + Attention},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/SheldonC2005/ModelArena}
}
```

---

## 🔍 Keywords

Deepfake Detection, Video Classification, InceptionResNetV2, Attention Mechanism, Test-Time Augmentation, PyTorch, Mixed Precision Training, RTX 3050, Low VRAM Optimization, Optuna, Binary Classification

---

**Status:** ✅ Complete & Tested | **Last Updated:** February 8, 2026

**Ready for deployment! 🎯🚀**