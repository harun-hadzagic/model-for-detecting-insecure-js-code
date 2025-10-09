import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
import os

# Suppress Hugging Face UserWarnings related to model loading/saving
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
MODEL_NAME = 'microsoft/codebert-base'
NUM_LABELS = 2 # 0: Secure, 1: Insecure
OUTPUT_DIR = './results_codebert_js' # Directory where model checkpoints will be saved
LOGGING_DIR = './logs'

# --- 1. Helper Function to Load Tokenized Data ---

def load_npz_data(file_path):
    """Loads tokenized data from .npz file and converts to Hugging Face Dataset format."""
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        print(f"Error: Required file not found: {file_path}")
        print("Please ensure your tokenization script ran successfully and generated the .npz files.")
        exit()

    # Hugging Face Dataset expects Python lists or standard NumPy arrays
    raw_data = {
        'input_ids': data['input_ids'].tolist(),
        'attention_mask': data['attention_mask'].tolist(),
        'labels': data['labels'].tolist()
    }
    
    # Create the Dataset object
    return Dataset.from_dict(raw_data)

# --- 2. Metrics Function (Crucial for Evaluation) ---

def compute_metrics(pred):
    """Calculates accuracy, precision, recall, and F1-score, focusing on the INSECURE class (1)."""
    labels = pred.label_ids
    # Get the predicted class (the one with the highest logit score)
    preds = np.argmax(pred.predictions, axis=1)
    
    # Use 'binary' average and target label 1 (insecure)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', pos_label=1, zero_division=0
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision_insecure': precision,
        'recall_insecure': recall, # Critical metric: how well we catch insecure code (minimize False Negatives)
    }

# --- 3. Main Training Function ---

def train_codebert_classifier():
    # Load tokenized datasets
    print("--- Loading Tokenized Data ---")
    train_dataset = load_npz_data('train_data.npz')
    val_dataset = load_npz_data('val_data.npz')
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n--- Initializing CodeBERT Model for Classification ---")
    # Load CodeBERT model pre-trained weights. The AutoModel handles adding the classification head.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_LABELS
    )
    # Note: The console message about uninitialized weights is expected and means the new classification layer is ready to be trained.
    
    print("\n--- Configuring Training Arguments ---")
    # Define hyperparameters and training settings
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOGGING_DIR,
        num_train_epochs=5,                          # Start with 5 epochs for fine-tuning
        per_device_train_batch_size=8,             # Adjust based on your available RAM/GPU memory
        per_device_eval_batch_size=16,             
        warmup_steps=500,                            
        weight_decay=0.01,                           
        learning_rate=5e-5,                          # Standard learning rate for fine-tuning CodeBERT
        logging_steps=50,                            
        # --- FIX: Changed 'evaluation_strategy' to 'eval_strategy' for version compatibility ---
        eval_strategy="epoch",                 
        # --- FIX: Changed 'save_strategy' to 'save_strategy' (already short) for version compatibility ---
        save_strategy="epoch",                       
        load_best_model_at_end=True,                 
        # CRUCIAL: Selects the model that achieved the highest RECALL for the insecure class
        metric_for_best_model="recall_insecure",    
    )

    # --- 4. Initialize the Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME),
    )

    # --- 5. Start Training ---
    print("\n--- Starting Fine-Tuning Process ---")
    trainer.train()

    print("\n--- Training Complete ---")
    # Save the final best model
    best_model_path = f"{OUTPUT_DIR}/final_best_model"
    trainer.save_model(best_model_path)
    print(f"✅ Best model saved to: {best_model_path}")

    # --- 6. Final Evaluation on Test Set ---
    print("\n--- Final Evaluation on Test Set ---")
    test_dataset = load_npz_data('test_data.npz')
    
    results = trainer.evaluate(test_dataset)
    
    # --- Print Final Results ---
    print("\n=============================================")
    print("      VULNERABILITY DETECTION RESULTS")
    print("=============================================")
    print(f"Test Accuracy:          {results.get('eval_accuracy', 0):.4f}")
    print(f"Test Precision (Insecure): {results.get('eval_precision_insecure', 0):.4f}")
    print(f"Test Recall (Insecure):   {results.get('eval_recall_insecure', 0):.4f}")
    print(f"Test F1-Score:          {results.get('eval_f1', 0):.4f}")
    print("\nNote: High Recall is desirable to minimize missed vulnerabilities.")
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