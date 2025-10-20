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

def load_and_prepare_data_alternating(file_path):
    """Loads JSON and selects alternating samples from pairs (one per pair)."""
    print(f"Loading data from {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None, None

    try:
        df = pd.json_normalize(data)
    except Exception as e:
        print(f"Error normalizing JSON data: {e}")
        return None, None
    
    # Ensure 'id' is integer
    try:
        df['id'] = df['id'].astype(int)
    except ValueError as ve:
        print("\n--- CRITICAL ERROR: ID Column Check ---")
        print("The 'id' column contains non-numeric values.")
        raise ve
    
    # Convert text label to numerical label (0 for secure, 1 for insecure)
    df['target'] = df['label'].apply(lambda x: 1 if x == 'insecure' else 0)

    # --- NEW: Alternating Selection Logic ---
    # Group by pairs: (1,2), (3,4), (5,6), etc.
    # Pair index = (id - 1) // 2
    df['pair_index'] = (df['id'] - 1) // 2
    
    print(f"Total entries loaded: {len(df)}")
    print(f"Total unique pairs: {df['pair_index'].nunique()}")
    
    # For each pair, alternate between taking insecure (odd pair_index) and secure (even pair_index)
    # Pair 0: take insecure (id 1)
    # Pair 1: take secure (id 4)
    # Pair 2: take insecure (id 5)
    # Pair 3: take secure (id 8)
    
    selected_samples = []
    
    for pair_idx in df['pair_index'].unique():
        pair_df = df[df['pair_index'] == pair_idx]
        
        # Alternate: even pair_index -> take insecure, odd pair_index -> take secure
        if pair_idx % 2 == 0:
            # Take insecure sample (label = 'insecure')
            sample = pair_df[pair_df['label'] == 'insecure']
        else:
            # Take secure sample (label = 'secure')
            sample = pair_df[pair_df['label'] == 'secure']
        
        if not sample.empty:
            selected_samples.append(sample.iloc[0])
    
    # Create new dataframe from selected samples
    alternating_df = pd.DataFrame(selected_samples).reset_index(drop=True)
    
    print(f"\n--- Alternating Selection Summary ---")
    print(f"Selected samples: {len(alternating_df)}")
    print(f"Insecure samples: {len(alternating_df[alternating_df['label'] == 'insecure'])}")
    print(f"Secure samples: {len(alternating_df[alternating_df['label'] == 'secure'])}")
    
    # Extract unique pair indices for splitting
    pair_indices = alternating_df['pair_index'].unique()
    
    return alternating_df, pair_indices


# --- Tokenization Function ---

def tokenize_code(df, tokenizer):
    """Applies tokenization and returns the encoded sequences."""
    
    print(f"Tokenizing {len(df)} snippets with max_length={MAX_LENGTH}...")
    
    encodings = tokenizer(
        df['code'].tolist(), 
        truncation=True, 
        padding='max_length', 
        max_length=MAX_LENGTH, 
        return_tensors='np'
    )
    
    tokenized_data = {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': df['target'].values
    }
    
    return tokenized_data

# --- Main Execution ---

if __name__ == '__main__':
    
    # Load and prepare data with alternating selection
    df, pair_indices = load_and_prepare_data_alternating(JSON_FILE)

    # Initialize the code-specific tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"\nUsing tokenizer: {MODEL_NAME}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Split the unique pair indices first (to prevent data leakage)
    train_pairs, test_val_pairs = train_test_split(
        pair_indices, test_size=0.3, random_state=42
    )
    val_pairs, test_pairs = train_test_split(
        test_val_pairs, test_size=0.5, random_state=42
    )

    # Extract the actual data rows for each split
    train_df = df[df['pair_index'].isin(train_pairs)].reset_index(drop=True)
    val_df = df[df['pair_index'].isin(val_pairs)].reset_index(drop=True)
    test_df = df[df['pair_index'].isin(test_pairs)].reset_index(drop=True)

    # Final split verification
    print("\n--- Data Split Summary ---")
    print(f"Training set size: {len(train_df)} snippets")
    print(f"  - Insecure: {len(train_df[train_df['target'] == 1])}")
    print(f"  - Secure: {len(train_df[train_df['target'] == 0])}")
    print(f"Validation set size: {len(val_df)} snippets")
    print(f"  - Insecure: {len(val_df[val_df['target'] == 1])}")
    print(f"  - Secure: {len(val_df[val_df['target'] == 0])}")
    print(f"Test set size: {len(test_df)} snippets")
    print(f"  - Insecure: {len(test_df[test_df['target'] == 1])}")
    print(f"  - Secure: {len(test_df[test_df['target'] == 0])}")
    
    # Tokenize each split
    train_tokens = tokenize_code(train_df, tokenizer)
    val_tokens = tokenize_code(val_df, tokenizer)
    test_tokens = tokenize_code(test_df, tokenizer)
    
    print("\n--- Tokenization Output Examples ---")
    print("First Training Sample:")
    print(f"Code: {train_df['code'].iloc[0][:100]}...")
    print(f"Label: {train_df['label'].iloc[0]} (target={train_tokens['labels'][0]})")
    print("Tokenized Input IDs (First 20):")
    print(train_tokens['input_ids'][0][:20])
    
    # Save the tokenized data with different filenames to avoid overwriting
    np.savez('train_data_alternating.npz', **train_tokens)
    np.savez('val_data_alternating.npz', **val_tokens)
    np.savez('test_data_alternating.npz', **test_tokens)
    
    print("\n✅ Alternating tokenization and splitting complete.")
    print("Data saved to: train_data_alternating.npz, val_data_alternating.npz, test_data_alternating.npz")

