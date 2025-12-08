# Quick Start Guide - Alternating Dataset Model

## What Was Created

### ✅ New Training Scripts
- **`tokenization_alternating.py`** - Prepares alternating dataset (one sample per pair)
- **`train_model_alternating.py`** - Trains model on alternating data
- **`predict_alternating.py`** - Makes predictions with alternating model

### ✅ Trained Model
- **`results_codebert_js_alternating/final_best_model/`** - Ready-to-use trained model

### ✅ Training Data
- **`train_data_alternating.npz`** - 1,745 samples
- **`val_data_alternating.npz`** - 374 samples  
- **`test_data_alternating.npz`** - 374 samples

### ✅ Visualizations
- **`confusion_matrix_alternating.png`** - Test set confusion matrix

## Performance Summary

| Metric | Result |
|--------|---------|
| **Test Accuracy** | 96.79% |
| **Recall (Insecure)** | 98.41% ⭐ |
| **Precision (Insecure)** | 95.38% |
| **F1-Score** | 96.88% |

**Key Achievement**: The model catches 98.41% of all vulnerabilities while maintaining 95.38% precision!

## How It Works

The alternating selection takes one code sample from each pair:
- **Pair 1** (IDs 1,2): Take insecure ❌
- **Pair 2** (IDs 3,4): Take secure ✅
- **Pair 3** (IDs 5,6): Take insecure ❌
- **Pair 4** (IDs 7,8): Take secure ✅
- And so on...

This gives us **2,493 diverse samples** (half the dataset) with **balanced representation**.

## Quick Test

Run the prediction script to see the model in action:

```bash
python predict_alternating.py
```

Expected output:
```
[SCANNING: VULNERABLE SQLI]
  -> Prediction: INSECURE (Vulnerable)
  -> Confidence: 0.9996

[SCANNING: SECURE SQLI]
  -> Prediction: SECURE
  -> Confidence: 0.9988
```

## Compare With Original Model

| Model | Training Samples | Test Accuracy | Recall (Insecure) |
|-------|-----------------|---------------|-------------------|
| **Original** | ~5,000 | Check confusion_matrix.png | Check results |
| **Alternating** | 2,493 | 96.79% | **98.41%** |

The alternating model achieves excellent results with **50% less training data**!

## Next Steps

1. ✅ Model is trained and ready to use
2. ✅ Prediction script (`predict_alternating.py`) is working
3. ✅ All artifacts saved in `results_codebert_js_alternating/`
4. 📊 View `confusion_matrix_alternating.png` for detailed performance
5. 📖 Read `ALTERNATING_MODEL_SUMMARY.md` for full analysis

## File Structure

```
model-for-detecting-insecure-js-code/
├── data/
│   └── secure_insecure_code_dataset.json
│
├── Original Model Files:
│   ├── tokenization.py
│   ├── train_model.py
│   ├── predict.py
│   ├── results_codebert_js/
│   └── confusion_matrix.png
│
└── Alternating Model Files (NEW):
    ├── tokenization_alternating.py
    ├── train_model_alternating.py
    ├── predict_alternating.py
    ├── results_codebert_js_alternating/
    ├── confusion_matrix_alternating.png
    ├── *_data_alternating.npz (3 files)
    ├── ALTERNATING_MODEL_SUMMARY.md
    └── QUICK_START_ALTERNATING.md (this file)
```

---

**Status**: ✅ Complete and ready for use!

