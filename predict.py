import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
MODEL_PATH = './results_codebert_js/final_best_model'
MODEL_NAME = 'microsoft/codebert-base'
MAX_LENGTH = 256
LABELS = {0: "SECURE", 1: "INSECURE (Vulnerable)"}

# --- Global Model and Tokenizer (Loaded once) ---
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval() # Set model to evaluation mode (crucial for inference)
    
    # Check for available device
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
        
    model.to(DEVICE)
    print(f"✅ Model loaded successfully from {MODEL_PATH} and running on {DEVICE}.")

except Exception as e:
    print(f"❌ Error loading model or tokenizer: {e}")
    print("Please ensure you ran train_model.py successfully and the model path is correct.")
    exit()

# --- Inference Function ---

def classify_code_snippet(code_snippet: str):
    """Tokenizes a single snippet and uses the model to predict its security label."""
    
    # 1. Tokenize the input snippet
    inputs = tokenizer(
        code_snippet,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    
    # 2. Move inputs to the correct device (CPU/GPU/MPS)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # 3. Prediction (Inference)
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 4. Process logits (raw scores) to get probability and predicted class
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    predicted_class_id = np.argmax(probabilities)
    
    # 5. Format Result
    result = {
        "prediction": LABELS[predicted_class_id],
        "confidence": probabilities[predicted_class_id].item(),
        "insecure_confidence": probabilities[1].item()
    }
    return result

# --- Example Usage ---

if __name__ == '__main__':
    
    print("\n--- Testing Model Predictions ---")

    # Example 1: INSECURE (SQL Injection via concatenation)
    insecure_code = """
    app.get('/user', (req, res) => {
      const id = req.query.id;
      db.query("SELECT * FROM users WHERE id = " + id)
        .then(result => res.send(result));
    });
    """
    
    # Example 2: SECURE (Parameterized query)
    secure_code = """
    app.get('/user', (req, res) => {
      const id = req.query.id;
      db.query("SELECT * FROM users WHERE id = ?", [id])
        .then(result => res.send(result));
    });
    """
    
    # Example 3: NEW INSECURE (Command Injection via template literal)
    new_insecure_code = """
    app.get('/run', (req, res) => {
      const user_cmd = req.query.cmd;
      exec(`bash -c "echo 'Running' && ${user_cmd}"`);
    });
    """

    snippets = {
        "VULNERABLE SQLI": insecure_code,
        "SECURE SQLI": secure_code,
        "NEW VULNERABLE CMD INJECTION": new_insecure_code
    }

    for name, code in snippets.items():
        print(f"\n[SCANNING: {name}]")
        prediction = classify_code_snippet(code)
        
        print(f"  -> Prediction: {prediction['prediction']}")
        print(f"  -> Confidence: {prediction['confidence']:.4f}")
        print(f"  -> INSECURE Score: {prediction['insecure_confidence']:.4f}")