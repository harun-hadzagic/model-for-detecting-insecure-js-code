import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
MODEL_PATH = 'results_codebert_js_full_regularized/final_best_model_full_regularized'
MODEL_NAME = 'microsoft/codebert-base'
MAX_LENGTH = 512
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
    print("Please ensure you ran train_model_with_confusion_matrix.py successfully and the model path is correct.")
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
    # After line 68, add:
    print(f"  -> SECURE Score: {probabilities[0]:.4f}")
    print(f"  -> INSECURE Score: {probabilities[1]:.4f}")
    return result

# --- Fresh Test Data (Not Seen During Training) ---

FRESH_TEST_CASES = {
    # ========== SQL INJECTION TESTS ==========
    "SQL Injection - Template Literal": {
        "code": """
        app.post('/api/search', async (req, res) => {
            const searchTerm = req.body.search;
            const query = `SELECT * FROM products WHERE name LIKE '%${searchTerm}%'`;
            const results = await db.query(query);
            res.json(results);
        });
        """,
        "expected": "INSECURE",
        "vulnerability": "SQL Injection via template literal in LIKE clause"
    },
    
    "SQL Injection - Array Join": {
        "code": """
        app.get('/api/users', (req, res) => {
            const userIds = req.query.ids.split(',');
            const query = 'SELECT * FROM users WHERE id IN (' + userIds.join(',') + ')';
            db.query(query, (err, results) => {
                res.json(results);
            });
        });
        """,
        "expected": "INSECURE",
        "vulnerability": "SQL Injection via array join without sanitization"
    },
    
    "SQL Injection - SECURE (Parameterized)": {
        "code": """
        const express = require('express');
        const db = require('./db');
        app.post('/api/search', async (req, res) => {
            const searchTerm = req.body.search;
            const query = 'SELECT * FROM products WHERE name LIKE ?';
            const results = await db.query(query, ['%' + searchTerm + '%']);
            res.json(results);
        });
        """,
        "expected": "SECURE",
        "vulnerability": "Secure parameterized query"
    },
    
    # ========== COMMAND INJECTION TESTS ==========
    "Command Injection - spawn()": {
        "code": """
        const { spawn } = require('child_process');
        app.post('/api/backup', (req, res) => {
            const filename = req.body.filename;
            const tar = spawn('tar', ['-czf', filename, './data']);
            tar.on('close', (code) => res.send('Backup complete'));
        });
        """,
        "expected": "INSECURE",
        "vulnerability": "Command injection via spawn with user-controlled filename"
    },
    
    "Command Injection - execFile()": {
        "code": """
        const { execFile } = require('child_process');
        app.get('/api/convert', (req, res) => {
            const inputFile = req.query.file;
            execFile('convert', [inputFile, 'output.jpg'], (error) => {
                res.send('Conversion done');
            });
        });
        """,
        "expected": "INSECURE",
        "vulnerability": "Command injection via execFile with user input"
    },
    
    "Command Injection - SECURE": {
        "code": """
        const { execFile } = require('child_process');
        const path = require('path');
        app.get('/api/convert', (req, res) => {
            const inputFile = req.query.file;
            const sanitized = path.basename(inputFile);
            if (!sanitized.match(/^[a-zA-Z0-9._-]+$/)) {
                return res.status(400).send('Invalid filename');
            }
            execFile('convert', [sanitized, 'output.jpg'], (error) => {
                res.send('Conversion done');
            });
        });
        """,
        "expected": "SECURE",
        "vulnerability": "Secure command execution with input validation"
    },
    
    # ========== XSS TESTS ==========
    "XSS - document.write()": {
        "code": """
        app.get('/api/render', (req, res) => {
            const userContent = req.query.content;
            res.send('<html><body><script>document.write("' + userContent + '");</script></body></html>');
        });
        """,
        "expected": "INSECURE",
        "vulnerability": "XSS via document.write with unescaped user input"
    },
    
    "XSS - setTimeout()": {
        "code": """
        app.get('/api/execute', (req, res) => {
            const userScript = req.query.script;
            res.send(`<script>setTimeout(function() { ${userScript} }, 1000);</script>`);
        });
        """,
        "expected": "INSECURE",
        "vulnerability": "XSS via setTimeout with user-controlled script"
    },
    
    "XSS - SECURE (Escaped)": {
        "code": """
        const escapeHtml = require('escape-html');
        app.get('/api/render', (req, res) => {
            const userContent = req.query.content;
            const escaped = escapeHtml(userContent);
            res.send('<html><body><div>' + escaped + '</div></body></html>');
        });
        """,
        "expected": "SECURE",
        "vulnerability": "Secure XSS prevention with HTML escaping"
    },
    
    # ========== NOSQL INJECTION TESTS ==========
    "NoSQL Injection - $regex": {
        "code": """
        const MongoClient = require('mongodb').MongoClient;
        app.post('/api/users/search', async (req, res) => {
            const searchPattern = req.body.pattern;
            const query = { username: { $regex: searchPattern } };
            const users = await db.collection('users').find(query).toArray();
            res.json(users);
        });
        """,
        "expected": "INSECURE",
        "vulnerability": "NoSQL injection via $regex with user input"
    },
    
    "NoSQL Injection - $where": {
        "code": """
        app.get('/api/products', async (req, res) => {
            const filter = req.query.filter;
            const query = { $where: `this.category === "${filter}"` };
            const products = await db.collection('products').find(query).toArray();
            res.json(products);
        });
        """,
        "expected": "INSECURE",
        "vulnerability": "NoSQL injection via $where with string interpolation"
    },
    
    "NoSQL Injection - SECURE": {
        "code": """
        app.post('/api/users/search', async (req, res) => {
            const searchPattern = req.body.pattern;
            if (typeof searchPattern !== 'string' || searchPattern.length > 100) {
                return res.status(400).json({ error: 'Invalid pattern' });
            }
            const query = { username: { $regex: searchPattern, $options: 'i' } };
            const users = await db.collection('users').find(query).toArray();
            res.json(users);
        });
        """,
        "expected": "SECURE",
        "vulnerability": "Secure NoSQL query with input validation"
    },
    
    # ========== SSRF TESTS ==========
    "SSRF - Direct URL": {
        "code": """
        const axios = require('axios');
        app.get('/api/fetch', async (req, res) => {
            const url = req.query.url;
            const response = await axios.get(url);
            res.json(response.data);
        });
        """,
        "expected": "INSECURE",
        "vulnerability": "SSRF via direct URL fetch without validation"
    },
    
    "SSRF - SECURE": {
        "code": """
        const axios = require('axios');
        const url = require('url');
        app.get('/api/fetch', async (req, res) => {
            const targetUrl = req.query.url;
            const parsed = url.parse(targetUrl);
            if (parsed.hostname === 'localhost' || parsed.hostname === '127.0.0.1' || parsed.hostname.startsWith('192.168.')) {
                return res.status(403).json({ error: 'Internal URLs not allowed' });
            }
            if (!parsed.protocol.startsWith('http')) {
                return res.status(400).json({ error: 'Only HTTP/HTTPS allowed' });
            }
            const response = await axios.get(targetUrl);
            res.json(response.data);
        });
        """,
        "expected": "SECURE",
        "vulnerability": "Secure SSRF prevention with URL validation"
    },
    
    # ========== PATH TRAVERSAL TESTS ==========
    "Path Traversal - Direct Read": {
        "code": """
        const fs = require('fs');
        app.get('/api/file', (req, res) => {
            const filePath = req.query.path;
            const content = fs.readFileSync(filePath, 'utf8');
            res.send(content);
        });
        """,
        "expected": "INSECURE",
        "vulnerability": "Path traversal via direct file read without sanitization"
    },
    
    "Path Traversal - SECURE": {
        "code": """
        const fs = require('fs');
        const path = require('path');
        app.get('/api/file', (req, res) => {
            const filePath = req.query.path;
            const safePath = path.normalize(filePath).replace(/^(\.\.[\/\\])+/, '');
            const fullPath = path.join(__dirname, 'public', safePath);
            if (!fullPath.startsWith(path.join(__dirname, 'public'))) {
                return res.status(403).send('Access denied');
            }
            const content = fs.readFileSync(fullPath, 'utf8');
            res.send(content);
        });
        """,
        "expected": "SECURE",
        "vulnerability": "Secure path traversal prevention with normalization"
    },
    
    # ========== REDOS TESTS ==========
    "ReDoS - Vulnerable Regex": {
        "code": """
        app.post('/api/validate', (req, res) => {
            const email = req.body.email;
            const emailRegex = /^([a-zA-Z0-9_\.-]+)@([\da-zA-Z\.-]+)\.([a-zA-Z\.]{2,6})$/;
            if (emailRegex.test(email)) {
                res.json({ valid: true });
            } else {
                res.json({ valid: false });
            }
        });
        """,
        "expected": "INSECURE",
        "vulnerability": "ReDoS via complex regex with nested quantifiers"
    },
    
    "ReDoS - SECURE": {
        "code": """
        app.post('/api/validate', (req, res) => {
            const email = req.body.email;
            if (email && email.length > 1000) {
                return res.status(400).json({ error: 'Input too long' });
            }
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (emailRegex.test(email)) {
                res.json({ valid: true });
            } else {
                res.json({ valid: false });
            }
        });
        """,
        "expected": "SECURE",
        "vulnerability": "Secure regex with input length limit"
    },
    
    # ========== EDGE CASES ==========
    "Edge Case - Nested Object Access": {
        "code": """
        app.post('/api/process', (req, res) => {
            const userInput = req.body.data.value;
            const query = 'SELECT * FROM table WHERE column = "' + userInput + '"';
            db.query(query, (err, results) => {
                res.json(results);
            });
        });
        """,
        "expected": "INSECURE",
        "vulnerability": "SQL injection via nested object property"
    },
    
    "Edge Case - Multiple Input Sources": {
        "code": """
        app.put('/api/update/:id', (req, res) => {
            const id = req.params.id;
            const name = req.body.name;
            const query = 'UPDATE users SET name = "' + name + '" WHERE id = ' + id;
            db.query(query, (err) => {
                res.json({ success: true });
            });
        });
        """,
        "expected": "INSECURE",
        "vulnerability": "SQL injection via multiple input sources (params + body)"
    },
    
    "Edge Case - SECURE Complex": {
        "code": """
        const express = require('express');
        const { body, param, validationResult } = require('express-validator');
        app.put('/api/update/:id', 
            param('id').isInt(),
            body('name').isString().isLength({ min: 1, max: 100 }),
            async (req, res) => {
                const errors = validationResult(req);
                if (!errors.isEmpty()) {
                    return res.status(400).json({ errors: errors.array() });
                }
                const id = req.params.id;
                const name = req.body.name;
                await db.query('UPDATE users SET name = ? WHERE id = ?', [name, id]);
                res.json({ success: true });
            }
        );
        """,
        "expected": "SECURE",
        "vulnerability": "Secure code with validation middleware and parameterized queries"
    }
}

# --- Example Usage ---

if __name__ == '__main__':
    
    print("\n" + "=" * 80)
    print("  TESTING MODEL ON FRESH DATA (NOT SEEN DURING TRAINING)")
    print("=" * 80)
    
    results_summary = {
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "by_vulnerability": {}
    }
    
    for test_name, test_case in FRESH_TEST_CASES.items():
        code = test_case["code"]
        expected = test_case["expected"]
        vulnerability = test_case["vulnerability"]
        
        print(f"\n{'='*80}")
        print(f"[TEST: {test_name}]")
        print(f"Vulnerability Type: {vulnerability}")
        print(f"Expected: {expected}")
        print("-" * 80)
        
        try:
            prediction = classify_code_snippet(code)
            predicted_label = prediction['prediction']
            confidence = prediction['confidence']
            insecure_score = prediction['insecure_confidence']
            
            # Determine if prediction matches expected
            is_correct = (
                (expected == "INSECURE" and predicted_label == "INSECURE (Vulnerable)") or
                (expected == "SECURE" and predicted_label == "SECURE")
            )
            
            results_summary["total"] += 1
            if is_correct:
                results_summary["correct"] += 1
                status = "✅ CORRECT"
            else:
                results_summary["incorrect"] += 1
                status = "❌ INCORRECT"
            
            # Track by vulnerability type
            vuln_type = vulnerability.split('-')[0].strip() if '-' in vulnerability else vulnerability.split()[0]
            if vuln_type not in results_summary["by_vulnerability"]:
                results_summary["by_vulnerability"][vuln_type] = {"correct": 0, "total": 0}
            results_summary["by_vulnerability"][vuln_type]["total"] += 1
            if is_correct:
                results_summary["by_vulnerability"][vuln_type]["correct"] += 1
            
            print(f"\n{status}")
            print(f"  Predicted: {predicted_label}")
            print(f"  Confidence: {confidence:.4f}")
            print(f"  INSECURE Score: {insecure_score:.4f}")
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            results_summary["total"] += 1
            results_summary["incorrect"] += 1
    
    # Print Summary
    print("\n" + "=" * 80)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nTotal Tests: {results_summary['total']}")
    print(f"Correct Predictions: {results_summary['correct']} ({results_summary['correct']/results_summary['total']*100:.2f}%)")
    print(f"Incorrect Predictions: {results_summary['incorrect']} ({results_summary['incorrect']/results_summary['total']*100:.2f}%)")
    
    print("\n--- Performance by Vulnerability Type ---")
    for vuln_type, stats in results_summary["by_vulnerability"].items():
        accuracy = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {vuln_type}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
    
    print("\n" + "=" * 80)