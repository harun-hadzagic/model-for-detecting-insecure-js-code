import json
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import numpy as np

# --- Configuration ---
JSON_FILE = 'data/secure_insecure_code_dataset.json'
MODEL_NAME = 'microsoft/codebert-base' # Pre-trained model known for code analysis
MAX_LENGTH = 256 # Max tokens per snippet. Adjust based on your code length distribution.

# --- Data Loading and Preprocessing ---

def load_and_prepare_data(file_path):
    """Loads JSON, ensures 'id' is integer, and groups pairs."""
    print(f"Loading data from {file_path}...")
    
    # --- Check 1: Load the data correctly ---
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None, None

    # ----------------------------------------------------
    # --- CRITICAL FIX: Use json_normalize for robust row creation ---
    # ----------------------------------------------------
    try:
        # pd.json_normalize correctly flattens the list of dicts into rows
        df = pd.json_normalize(data)
    except Exception as e:
        print(f"Error normalizing JSON data: {e}")
        return None, None
    
    # ----------------------------------------------------
    # --- CRITICAL FIX 2: Ensure 'id' is a numerical type ---
    # ----------------------------------------------------
    try:
        # Force the 'id' column to be an integer type. 
        df['id'] = df['id'].astype(int)
    except ValueError as ve:
        print("\n--- CRITICAL ERROR: ID Column Check ---")
        print("The 'id' column contains non-numeric values (e.g., strings, lists, or corrupt data).")
        print(f"Type of problematic column: {df['id'].dtype}")
        print(f"First 5 IDs for inspection: {df['id'].head().tolist()}")
        print("Please check your JSON file for incorrect 'id' formatting.")
        raise ve
    
    # 1. Convert text label to numerical label (0 for secure, 1 for insecure)
    df['target'] = df['label'].apply(lambda x: 1 if x == 'insecure' else 0)

    # 2. Group pairs by their starting ID (Original logic, now safe)
    # This assumes 'id' is now a clean integer column
    df['pair_id'] = df['id'].apply(lambda x: x if x % 2 != 0 else x - 1)
    
    # 3. Create a unique identifier for the pairs for splitting purposes
    pair_ids = df['pair_id'].unique()
    
    print(f"Total entries loaded: {len(df)}")
    print(f"Total unique pairs: {len(pair_ids)}")
    
    return df, pair_ids


# --- Tokenization Function ---

def tokenize_code(df, tokenizer):
    """Applies tokenization and returns the encoded sequences."""
    
    # The tokenizer handles cleaning, subword splitting, and mapping to IDs
    # `truncation=True` handles snippets longer than MAX_LENGTH
    # `padding='max_length'` pads all sequences to MAX_LENGTH
    print(f"Tokenizing {len(df)} snippets with max_length={MAX_LENGTH}...")
    
    encodings = tokenizer(
        df['code'].tolist(), 
        truncation=True, 
        padding='max_length', 
        max_length=MAX_LENGTH, 
        return_tensors='np' # Return NumPy arrays for easy handling
    )
    
    # Combine features (input_ids, attention_mask) and labels
    tokenized_data = {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': df['target'].values
    }
    
    return tokenized_data

# --- Main Execution ---

if __name__ == '__main__':
    
    # Load and prepare data
    df, pair_ids = load_and_prepare_data(JSON_FILE)

    # Initialize the code-specific tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Using tokenizer: {MODEL_NAME}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # 1. Split the unique pair IDs first
    train_pairs, test_val_pairs = train_test_split(
        pair_ids, test_size=0.3, random_state=42
    )
    val_pairs, test_pairs = train_test_split(
        test_val_pairs, test_size=0.5, random_state=42
    )

    # 2. Extract the actual data rows for each split
    train_df = df[df['pair_id'].isin(train_pairs)].reset_index(drop=True)
    val_df = df[df['pair_id'].isin(val_pairs)].reset_index(drop=True)
    test_df = df[df['pair_id'].isin(test_pairs)].reset_index(drop=True)

    # Final split verification
    print("\n--- Data Split Summary ---")
    print(f"Training set size: {len(train_df)} snippets ({len(train_df)//2} pairs)")
    print(f"Validation set size: {len(val_df)} snippets ({len(val_df)//2} pairs)")
    print(f"Test set size: {len(test_df)} snippets ({len(test_df)//2} pairs)")
    
    # 3. Tokenize each split
    train_tokens = tokenize_code(train_df, tokenizer)
    val_tokens = tokenize_code(val_df, tokenizer)
    test_tokens = tokenize_code(test_df, tokenizer)
    
    print("\n--- Tokenization Output Examples ---")
    print("Insecure Code Snippet:")
    print(train_df[train_df['target'] == 1]['code'].iloc[0][:100] + '...')
    print("Tokenized Input IDs (First 20 IDs):")
    print(train_tokens['input_ids'][0][:20])
    print("Tokenized Attention Mask (First 20 IDs):")
    print(train_tokens['attention_mask'][0][:20])
    print(f"Label: {train_tokens['labels'][0]} ('insecure')")
    
    # --- Next Step: Save to NumPy files for ML training ---
    
    # Save the tokenized data for use in your TensorFlow or PyTorch model
    np.savez('train_data.npz', **train_tokens)
    np.savez('val_data.npz', **val_tokens)
    np.savez('test_data.npz', **test_tokens)
    
    print("\n✅ Tokenization and splitting complete.")
    print("Data saved to: train_data.npz, val_data.npz, test_data.npz")