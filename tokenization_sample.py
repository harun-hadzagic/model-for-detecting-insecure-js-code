import json
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import numpy as np

# --- Configuration ---
JSON_FILE = 'data/js_dataset.json'
MODEL_NAME = 'microsoft/codebert-base' # Pre-trained model known for code analysis
MAX_LENGTH = 512 # Max tokens per snippet. Adjust based on your code length distribution.
SAMPLE_SIZE = 1  # Sample 10% of the dataset for quicker training/validation
RANDOM_STATE = 42  # For reproducibility

# --- Data Loading and Preprocessing ---

def load_and_prepare_data(file_path):
    """Loads JSON, ensures 'id' is integer, and prepares data."""
    print(f"Loading data from {file_path}...")
    
    # --- Check 1: Load the data correctly ---
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

    # ----------------------------------------------------
    # --- CRITICAL FIX: Use json_normalize for robust row creation ---
    # ----------------------------------------------------
    try:
        # pd.json_normalize correctly flattens the list of dicts into rows
        df = pd.json_normalize(data)
    except Exception as e:
        print(f"Error normalizing JSON data: {e}")
        return None
    
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
    
    # 2. Group pairs by their starting ID (same logic as tokenization.py)
    # This ensures pairs stay together to prevent data leakage
    df['pair_id'] = df['id'].apply(lambda x: x if x % 2 != 0 else x - 1)
    
    # 3. Create a unique identifier for the pairs for splitting purposes
    pair_ids = df['pair_id'].unique()
    
    print(f"Total entries loaded: {len(df)}")
    print(f"Total unique pairs: {len(pair_ids)}")
    print(f"Label distribution:")
    print(f"  Secure: {len(df[df['target'] == 0])} ({len(df[df['target'] == 0])/len(df)*100:.1f}%)")
    print(f"  Insecure: {len(df[df['target'] == 1])} ({len(df[df['target'] == 1])/len(df)*100:.1f}%)")
    
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
    result = load_and_prepare_data(JSON_FILE)
    if result is None:
        exit(1)
    df, pair_ids = result

    # Initialize the code-specific tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"\nUsing tokenizer: {MODEL_NAME}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Step 1: Sample PAIRS (not individual entries) to prevent data leakage
    # Handle the case where SAMPLE_SIZE = 1.0 (100% of dataset)
    if SAMPLE_SIZE >= 1.0:
        print(f"\n--- Using 100% of pairs (full dataset) ---")
        sampled_pairs = pair_ids.copy()
    else:
        print(f"\n--- Sampling {SAMPLE_SIZE*100:.0f}% of pairs ---")
        sampled_pairs, _ = train_test_split(
            pair_ids, 
            test_size=1 - SAMPLE_SIZE, 
            random_state=RANDOM_STATE
        )
    
    sampled_df = df[df['pair_id'].isin(sampled_pairs)].reset_index(drop=True)
    
    print(f"Sampled pairs: {len(sampled_pairs)} ({len(sampled_pairs)/len(pair_ids)*100:.1f}% of total)")
    print(f"Sampled entries: {len(sampled_df)} ({len(sampled_df)/len(df)*100:.1f}% of total)")
    print(f"  Secure: {len(sampled_df[sampled_df['target'] == 0])}")
    print(f"  Insecure: {len(sampled_df[sampled_df['target'] == 1])}")

    # Step 2: Split sampled PAIRS into train (70%), val (15%), test (15%)
    # This ensures pairs stay together and prevents data leakage
    print(f"\n--- Splitting sampled pairs ---")
    # First split: 70% train, 30% temp (which will become val + test)
    train_pairs, temp_pairs = train_test_split(
        sampled_pairs,
        test_size=0.30,  # 30% for val + test
        random_state=RANDOM_STATE
    )
    
    # Second split: Split temp (30%) into val (15%) and test (15%)
    val_pairs, test_pairs = train_test_split(
        temp_pairs,
        test_size=0.5,  # 50% of temp = 15% of total sampled
        random_state=RANDOM_STATE
    )
    
    # Extract the actual data rows for each split (both secure and insecure from each pair)
    train_df = df[df['pair_id'].isin(train_pairs)].reset_index(drop=True)
    val_df = df[df['pair_id'].isin(val_pairs)].reset_index(drop=True)
    test_df = df[df['pair_id'].isin(test_pairs)].reset_index(drop=True)

    # Final split verification
    print("\n--- Data Split Summary ---")
    print(f"Training set: {len(train_df)} snippets ({len(train_pairs)} pairs, {len(train_df)/len(sampled_df)*100:.1f}% of sampled)")
    print(f"  Secure: {len(train_df[train_df['target'] == 0])}, Insecure: {len(train_df[train_df['target'] == 1])}")
    print(f"Validation set: {len(val_df)} snippets ({len(val_pairs)} pairs, {len(val_df)/len(sampled_df)*100:.1f}% of sampled)")
    print(f"  Secure: {len(val_df[val_df['target'] == 0])}, Insecure: {len(val_df[val_df['target'] == 1])}")
    print(f"Test set: {len(test_df)} snippets ({len(test_pairs)} pairs, {len(test_df)/len(sampled_df)*100:.1f}% of sampled)")
    print(f"  Secure: {len(test_df[test_df['target'] == 0])}, Insecure: {len(test_df[test_df['target'] == 1])}")
    
    # Step 3: Tokenize each split
    print(f"\n--- Tokenizing splits ---")
    train_tokens = tokenize_code(train_df, tokenizer)
    val_tokens = tokenize_code(val_df, tokenizer)
    test_tokens = tokenize_code(test_df, tokenizer)
    
    print("\n--- Tokenization Output Examples ---")
    print("Insecure Code Snippet:")
    insecure_sample = train_df[train_df['target'] == 1]
    if len(insecure_sample) > 0:
        print(insecure_sample['code'].iloc[0][:100] + '...')
        print("Tokenized Input IDs (First 20 IDs):")
        print(train_tokens['input_ids'][0][:20])
        print("Tokenized Attention Mask (First 20 IDs):")
        print(train_tokens['attention_mask'][0][:20])
        print(f"Label: {train_tokens['labels'][0]} ('insecure')")
    
    # Step 4: Save to NumPy files for ML training
    print(f"\n--- Saving tokenized data ---")
    
    # Use different filenames based on whether we're using full dataset or sample
    if SAMPLE_SIZE >= 1.0:
        # Full dataset - use standard names for full dataset training script
        train_file = 'train_data.npz'
        val_file = 'val_data.npz'
        test_file = 'test_data.npz'
    else:
        # Sample dataset - use _sample suffix
        train_file = 'train_data_sample.npz'
        val_file = 'val_data_sample.npz'
        test_file = 'test_data_sample.npz'
    
    np.savez(train_file, **train_tokens)
    np.savez(val_file, **val_tokens)
    np.savez(test_file, **test_tokens)
    
    print("\n✅ Tokenization and splitting complete.")
    print("Data saved to:")
    print(f"  - {train_file}")
    print(f"  - {val_file}")
    print(f"  - {test_file}")
    if SAMPLE_SIZE >= 1.0:
        print(f"\nTotal entries processed: {len(sampled_df)} (100% of dataset)")
    else:
        print(f"\nTotal entries processed: {len(sampled_df)} out of {len(df)} ({SAMPLE_SIZE*100:.0f}% sample)")

