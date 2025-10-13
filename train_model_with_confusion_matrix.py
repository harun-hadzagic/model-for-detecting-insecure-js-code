import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
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
OUTPUT_DIR = './results_codebert_js' # Directory where model checkpoints will be saved
LOGGING_DIR = './logs'
CM_PATH = './confusion_matrix.png' # Path to save the confusion matrix image

# --- 1. Helper Function to Load Tokenized Data ---

def load_npz_data(file_path):
    """Loads tokenized data from .npz file and converts to Hugging Face Dataset format."""
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        print(f"Error: Required file not found: {file_path}")
        print("Please ensure your tokenization script ran successfully and generated the .npz files.")
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
    
    # NOTE: The Trainer uses this function for validation. The full matrix visualization
    # will be done separately in the main block using trainer.predict().
    
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
    
    # Plot using a custom color map (Seaborn's rocket_r is good for binary focus)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Vulnerability Detection Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(path)
    print(f"\n✅ Confusion Matrix saved to: {path}")
    

# --- 4. Main Training Function ---

def train_codebert_classifier():
    # Load tokenized datasets
    print("--- Loading Tokenized Data ---")
    train_dataset = load_npz_data('train_data.npz')
    val_dataset = load_npz_data('val_data.npz')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n--- Initializing CodeBERT Model for Classification ---")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_LABELS
    )
    
    print("\n--- Configuring Training Arguments ---")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOGGING_DIR,
        num_train_epochs=5,                          
        per_device_train_batch_size=8,             
        per_device_eval_batch_size=16,             
        warmup_steps=500,                            
        weight_decay=0.01,                           
        learning_rate=5e-5,                          
        logging_steps=50,                            
        eval_strategy="epoch",                 
        save_strategy="epoch",                       
        load_best_model_at_end=True,                 
        metric_for_best_model="recall_insecure",    
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME),
    )

    # Note: Since the model is already trained and saved, we can skip trainer.train() 
    # and load the saved model immediately for prediction.
    # To use the trained model, we must load the best one.
    
    print("\n--- Loading Trained Model for Evaluation ---")
    best_model_path = f"{OUTPUT_DIR}/final_best_model"
    if not os.path.exists(best_model_path):
        print(f"Model checkpoint not found at {best_model_path}. Re-running full training.")
        trainer.train()
        trainer.save_model(best_model_path)
    
    # Load the best model weights
    model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
    trainer.model = model
    
    # --- Final Evaluation on Test Set ---
    print("\n--- Generating Predictions on Test Set ---")
    test_dataset = load_npz_data('test_data.npz')
    
    # Use trainer.predict to get predictions (logits) and true labels
    predictions_output = trainer.predict(test_dataset)
    
    labels = predictions_output.label_ids
    # Convert logits to final predicted class IDs
    preds = np.argmax(predictions_output.predictions, axis=1)

    # Compute final metrics and print summary (same as before)
    results = compute_metrics(predictions_output)
    
    # --- Plot the Confusion Matrix ---
    plot_confusion_matrix(labels, preds, CM_PATH)

    # --- Print Final Results ---
    print("\n=============================================")
    print("      VULNERABILITY DETECTION RESULTS")
    print("=============================================")
    print(f"Test Accuracy:          {results.get('accuracy', 0):.4f}")
    print(f"Test Precision (Insecure): {results.get('precision_insecure', 0):.4f}")
    print(f"Test Recall (Insecure):   {results.get('recall_insecure', 0):.4f}")
    print(f"Test F1-Score:          {results.get('f1', 0):.4f}")
    print("=============================================\n")


if __name__ == '__main__':
    # Determine the device being used
    if torch.cuda.is_available():
        device_message = "Using CUDA GPU for training."
    elif torch.backends.mps.is_available():
        device_message = "Using Apple Silicon (MPS) for training."
    else:
        device_message = "Using CPU for training. This may take a long time."
    print(device_message)

    train_codebert_classifier()