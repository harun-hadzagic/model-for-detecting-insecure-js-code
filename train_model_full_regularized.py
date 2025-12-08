import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# Suppress Hugging Face UserWarnings related to model loading/saving
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuration ---
MODEL_NAME = 'microsoft/codebert-base'
NUM_LABELS = 2 # 0: Secure, 1: Insecure
OUTPUT_DIR = './results_codebert_js_full_regularized' # Directory where model checkpoints will be saved
LOGGING_DIR = './logs_full_regularized'
CM_PATH = './confusion_matrix_full_regularized.png' # Path to save the confusion matrix image
TRAINING_CURVES_PATH = './training_curves_full_regularized.png'

# Use full dataset files
TRAIN_DATA_FILE = 'train_data.npz'
VAL_DATA_FILE = 'val_data.npz'
TEST_DATA_FILE = 'test_data.npz'

# Force CPU for prediction (set to True if MPS causes issues)
FORCE_CPU_FOR_PREDICTION = False

# --- Regularization Configuration ---
WEIGHT_DECAY = 0.1  # Strong L2 regularization
DROPOUT_RATE = 0.3  # Dropout probability (0.3 = 30% dropout)
LEARNING_RATE = 3e-5  # Slightly lower learning rate
EARLY_STOPPING_PATIENCE = 3  # Stop if no improvement for 3 epochs (increased for larger dataset)
EARLY_STOPPING_THRESHOLD = 0.001  # Minimum change to qualify as improvement

# --- Additional Regularization (Optional) ---
LABEL_SMOOTHING = 0.1  # Label smoothing factor
MAX_GRAD_NORM = 1.0  # Gradient clipping
LR_SCHEDULER_TYPE = "cosine"  # Cosine annealing learning rate schedule
WARMUP_RATIO = 0.1  # 10% of training steps for warmup

# --- 1. Helper Function to Load Tokenized Data ---

def load_npz_data(file_path):
    """Loads tokenized data from .npz file and converts to Hugging Face Dataset format."""
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        print(f"Error: Required file not found: {file_path}")
        print("Please ensure tokenization.py ran successfully and generated the .npz files.")
        exit()

    raw_data = {
        'input_ids': data['input_ids'].tolist(),
        'attention_mask': data['attention_mask'].tolist(),
        'labels': data['labels'].tolist()
    }
    return Dataset.from_dict(raw_data)

# --- 2. Metrics Function (Crucial for Evaluation) ---

def compute_metrics(pred):
    """Calculates accuracy, precision, recall, and F1-score, focusing on the INSECURE class (1)."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', pos_label=1, zero_division=0
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision_insecure': precision,
        'recall_insecure': recall,
    }

# --- 3. Confusion Matrix Plotting Function ---

def plot_confusion_matrix(labels, preds, path):
    """Generates and saves a confusion matrix visualization."""
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(8, 6))
    
    # Use ConfusionMatrixDisplay for clean visualization
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['SECURE (0)', 'INSECURE (1)'])
    
    # Plot using a custom color map
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Vulnerability Detection Confusion Matrix (Full Dataset - Regularized)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(path)
    print(f"\n✅ Confusion Matrix saved to: {path}")

# --- 3.5. Training Curves Plotting Function ---

def plot_training_curves(trainer, save_path=TRAINING_CURVES_PATH):
    """Plots training and validation curves from trainer history to detect overfitting."""
    history = trainer.state.log_history
    
    if not history:
        print("⚠️  No training history available to plot curves.")
        return
    
    # Separate training and evaluation logs
    train_losses = []
    eval_losses = []
    eval_accuracies = []
    eval_f1_scores = []
    eval_precision = []
    eval_recall = []
    steps = []
    eval_steps = []
    
    for entry in history:
        if 'loss' in entry and 'eval_loss' not in entry:
            train_losses.append(entry['loss'])
            steps.append(entry.get('step', len(steps)))
        if 'eval_loss' in entry:
            eval_losses.append(entry['eval_loss'])
            eval_accuracies.append(entry.get('eval_accuracy', 0))
            eval_f1_scores.append(entry.get('eval_f1', 0))
            eval_precision.append(entry.get('eval_precision_insecure', 0))
            eval_recall.append(entry.get('eval_recall_insecure', 0))
            eval_steps.append(entry.get('step', len(eval_steps)))
    
    if not train_losses or not eval_losses:
        print("⚠️  Not enough training history to plot curves.")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Loss curves (most important for overfitting detection)
    axes[0, 0].plot(steps, train_losses, label='Train Loss', marker='o', alpha=0.7, linewidth=2)
    axes[0, 0].plot(eval_steps, eval_losses, label='Validation Loss', marker='s', alpha=0.7, linewidth=2)
    axes[0, 0].set_xlabel('Training Steps')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training vs Validation Loss\n(Full Dataset - Regularized Model)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[0, 1].plot(eval_steps, eval_accuracies, label='Validation Accuracy', marker='s', color='green', linewidth=2)
    axes[0, 1].set_xlabel('Training Steps')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Plot 3: F1 Score
    axes[0, 2].plot(eval_steps, eval_f1_scores, label='Validation F1', marker='s', color='purple', linewidth=2)
    axes[0, 2].set_xlabel('Training Steps')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].set_title('Validation F1 Score Over Time')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([0, 1])
    
    # Plot 4: Precision and Recall
    axes[1, 0].plot(eval_steps, eval_precision, label='Precision (Insecure)', marker='s', color='orange', linewidth=2)
    axes[1, 0].plot(eval_steps, eval_recall, label='Recall (Insecure)', marker='^', color='red', linewidth=2)
    axes[1, 0].set_xlabel('Training Steps')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision vs Recall (Insecure Class)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Plot 5: Loss difference (overfitting indicator)
    if len(train_losses) > 0 and len(eval_losses) > 0:
        # Interpolate to same length for comparison
        min_len = min(len(train_losses), len(eval_losses))
        if min_len > 1:
            # Get the last min_len values from each
            train_subset = train_losses[-min_len:] if len(train_losses) >= min_len else train_losses
            eval_subset = eval_losses[-min_len:] if len(eval_losses) >= min_len else eval_losses
            
            # Align by taking matching lengths
            actual_min = min(len(train_subset), len(eval_subset))
            train_subset = train_subset[:actual_min]
            eval_subset = eval_subset[:actual_min]
            
            loss_diff = [train_subset[i] - eval_subset[i] for i in range(actual_min)]
            axes[1, 1].plot(range(actual_min), loss_diff, label='Train - Val Loss', marker='o', color='red', linewidth=2)
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Epoch/Checkpoint Index')
            axes[1, 1].set_ylabel('Loss Difference')
            axes[1, 1].set_title('Overfitting Indicator\n(Closer to 0 = Better Generalization)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Combined metrics view
    axes[1, 2].plot(eval_steps, eval_accuracies, label='Accuracy', marker='s', alpha=0.7, linewidth=2)
    axes[1, 2].plot(eval_steps, eval_f1_scores, label='F1 Score', marker='^', alpha=0.7, linewidth=2)
    axes[1, 2].set_xlabel('Training Steps')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_title('Combined Validation Metrics')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Training curves saved to: {save_path}")
    plt.close()
    
    # Print overfitting analysis
    if len(eval_losses) >= 2:
        print("\n--- Overfitting Analysis (Full Dataset - Regularized Model) ---")
        if eval_losses[-1] > eval_losses[0]:
            print("⚠️  WARNING: Validation loss increased from {:.4f} to {:.4f}".format(
                eval_losses[0], eval_losses[-1]))
            print("   This suggests potential overfitting!")
        else:
            print("✅ Validation loss decreased from {:.4f} to {:.4f}".format(
                eval_losses[0], eval_losses[-1]))
        
        if len(train_losses) > 0 and len(eval_losses) > 0:
            final_train_loss = train_losses[-1] if train_losses else None
            final_eval_loss = eval_losses[-1]
            if final_train_loss:
                gap = final_train_loss - final_eval_loss
                if abs(gap) < 0.05:  # Very small gap
                    print("✅ Excellent: Train and validation losses are well-aligned (gap: {:.4f})".format(gap))
                elif gap < -0.1:  # Train loss much lower than eval loss
                    print("⚠️  WARNING: Large gap between train ({:.4f}) and validation ({:.4f}) loss".format(
                        final_train_loss, final_eval_loss))
                    print("   Gap: {:.4f} (negative = train loss lower = overfitting risk)".format(gap))
                else:
                    print("✅ Train and validation losses are reasonably aligned (gap: {:.4f})".format(gap))

# --- 4. Main Training Function ---

def train_codebert_classifier():
    # Load tokenized datasets
    print("=" * 70)
    print("  FULL DATASET TRAINING - REGULARIZED MODEL")
    print("  (With Early Stopping & Strong Regularization)")
    print("=" * 70)
    print("\n--- Loading Tokenized Full Dataset ---")
    print(f"Training data: {TRAIN_DATA_FILE}")
    print(f"Validation data: {VAL_DATA_FILE}")
    print(f"Test data: {TEST_DATA_FILE}")
    
    train_dataset = load_npz_data(TRAIN_DATA_FILE)
    val_dataset = load_npz_data(VAL_DATA_FILE)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if model already exists
    best_model_path = f"{OUTPUT_DIR}/final_best_model_full_regularized"
    
    print("\n--- Regularization Configuration ---")
    print(f"  Weight Decay: {WEIGHT_DECAY} (L2 regularization)")
    print(f"  Dropout Rate: {DROPOUT_RATE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Learning Rate Scheduler: {LR_SCHEDULER_TYPE}")
    print(f"  Warmup Ratio: {WARMUP_RATIO}")
    print(f"  Label Smoothing: {LABEL_SMOOTHING}")
    print(f"  Gradient Clipping: {MAX_GRAD_NORM}")
    print(f"  Early Stopping Patience: {EARLY_STOPPING_PATIENCE} epochs")
    print(f"  Early Stopping Threshold: {EARLY_STOPPING_THRESHOLD}")
    
    print("\n--- Configuring Training Arguments ---")
    
    # Calculate warmup steps based on dataset size
    total_steps = len(train_dataset) // 8 * 5  # Assuming batch size 8, 5 epochs max
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOGGING_DIR,
        num_train_epochs=5,  # Max epochs, but early stopping will likely stop earlier
        per_device_train_batch_size=8,             
        per_device_eval_batch_size=16,             
        warmup_steps=warmup_steps,  # Calculated based on dataset size
        weight_decay=WEIGHT_DECAY,  # Strong regularization
        learning_rate=LEARNING_RATE,  # Lower learning rate
        lr_scheduler_type=LR_SCHEDULER_TYPE,  # Cosine annealing
        label_smoothing_factor=LABEL_SMOOTHING,  # Label smoothing
        max_grad_norm=MAX_GRAD_NORM,  # Gradient clipping
        logging_steps=100,  # More frequent logging for larger dataset
        eval_strategy="epoch",                 
        save_strategy="epoch",                       
        load_best_model_at_end=True,                 
        metric_for_best_model="eval_loss",  # Use validation loss for early stopping
        greater_is_better=False,  # Lower loss is better
        save_total_limit=3,  # Keep only last 3 checkpoints
    )
    
    if os.path.exists(best_model_path):
        print(f"\n--- Existing model found at {best_model_path} ---")
        print("Loading existing model for evaluation (skipping training)...")
        model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
    else:
        print("\n--- Initializing CodeBERT Model with Dropout ---")
        # Load model with custom dropout configuration
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=NUM_LABELS,
            hidden_dropout_prob=DROPOUT_RATE,  # Dropout for hidden layers
            attention_probs_dropout_prob=DROPOUT_RATE,  # Dropout for attention
        )
        print(f"✅ Model initialized with dropout rate: {DROPOUT_RATE}")

    # Create early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME),
        callbacks=[early_stopping],  # Add early stopping callback
    )

    if not os.path.exists(best_model_path):
        print("\n--- Evaluating Initial (Untrained) Model ---")
        initial_eval = trainer.evaluate(val_dataset)
        print(f"Initial Validation Accuracy: {initial_eval.get('eval_accuracy', 0):.4f}")
        print(f"Initial Validation F1: {initial_eval.get('eval_f1', 0):.4f}")
        print(f"Initial Validation Loss: {initial_eval.get('eval_loss', 0):.4f}")
        print(f"Initial Validation Precision (Insecure): {initial_eval.get('eval_precision_insecure', 0):.4f}")
        print(f"Initial Validation Recall (Insecure): {initial_eval.get('eval_recall_insecure', 0):.4f}")
        
        print("\n--- Starting Training (Full Dataset - Regularized Model) ---")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"⚠️  Early stopping will stop training if validation loss doesn't improve for {EARLY_STOPPING_PATIENCE} epochs")
        print(f"📊 Training will use cosine learning rate scheduling with {warmup_steps} warmup steps")
        
        trainer.train()
        
        # Plot training curves to detect overfitting
        print("\n--- Generating Training Curves ---")
        plot_training_curves(trainer)
        
        # Save the best model (early stopping ensures we have the best one)
        trainer.save_model(best_model_path)
        print(f"\n✅ Best model saved to: {best_model_path}")
        print(f"   (Model selected based on lowest validation loss)")
    else:
        print("✅ Using existing trained model - skipping training")
    
    # --- Final Evaluation on Test Set ---
    print("\n--- Generating Predictions on Test Set ---")
    test_dataset = load_npz_data(TEST_DATA_FILE)
    
    # Use trainer.predict - the trainer already has the best model loaded
    # For MPS devices, if there's an error, fall back to CPU
    use_cpu_for_prediction = FORCE_CPU_FOR_PREDICTION
    
    if not use_cpu_for_prediction:
        try:
            predictions_output = trainer.predict(test_dataset)
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "mps" in error_msg or "placeholder storage" in error_msg:
                print("⚠️  MPS device error detected. Retrying prediction on CPU...")
                use_cpu_for_prediction = True
            else:
                raise e
    
    if use_cpu_for_prediction:
        print("Using CPU for prediction...")
        # Load model on CPU for prediction
        cpu_model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
        cpu_model = cpu_model.to('cpu')
        
        # Create CPU training args
        import copy
        cpu_args = copy.deepcopy(training_args)
        cpu_args.device = torch.device('cpu')
        cpu_args.dataloader_pin_memory = False  # Disable pin memory for CPU
        
        # Create CPU trainer for prediction only
        cpu_trainer = Trainer(
            model=cpu_model,
            args=cpu_args,
            compute_metrics=compute_metrics,
            tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME),
        )
        predictions_output = cpu_trainer.predict(test_dataset)
    
    labels = predictions_output.label_ids
    # Convert logits to final predicted class IDs
    preds = np.argmax(predictions_output.predictions, axis=1)

    # Compute final metrics and print summary
    results = compute_metrics(predictions_output)
    
    # --- Plot the Confusion Matrix ---
    plot_confusion_matrix(labels, preds, CM_PATH)

    # --- Print Final Results ---
    print("\n" + "=" * 70)
    print("      VULNERABILITY DETECTION RESULTS")
    print("      (Full Dataset - Regularized Model)")
    print("=" * 70)
    print(f"Test Set Size: {len(test_dataset)} samples")
    print(f"Test Accuracy:          {results.get('accuracy', 0):.4f}")
    print(f"Test Precision (Insecure): {results.get('precision_insecure', 0):.4f}")
    print(f"Test Recall (Insecure):   {results.get('recall_insecure', 0):.4f}")
    print(f"Test F1-Score:          {results.get('f1', 0):.4f}")
    print("=" * 70)
    print("\n💡 Regularization Applied:")
    print(f"   - Weight Decay: {WEIGHT_DECAY}")
    print(f"   - Dropout: {DROPOUT_RATE}")
    print(f"   - Label Smoothing: {LABEL_SMOOTHING}")
    print(f"   - Gradient Clipping: {MAX_GRAD_NORM}")
    print(f"   - Learning Rate Scheduler: {LR_SCHEDULER_TYPE}")
    print(f"   - Early Stopping: Enabled (patience={EARLY_STOPPING_PATIENCE})")
    print("\n✅ This model should have better generalization than the non-regularized version.")


if __name__ == '__main__':
    # Determine the device being used
    if torch.cuda.is_available():
        device_message = "Using CUDA GPU for training."
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    elif torch.backends.mps.is_available():
        device_message = "Using Apple Silicon (MPS) for training."
        # Set environment variable to help with MPS issues
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    else:
        device_message = "Using CPU for training. This may take a long time."
    print(device_message)

    train_codebert_classifier()

