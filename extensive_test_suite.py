"""
Extensive Test Suite for JavaScript Vulnerability Detection Model

This test suite contains 50+ test cases covering:
- Underrepresented vulnerability patterns
- Different input sources (req.params, req.headers, req.cookies)
- Different HTTP methods (PUT, DELETE, PATCH)
- Different code structures (router patterns, module.exports)
- Edge cases and boundary conditions
- Real-world complex scenarios
- Obfuscated patterns
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import warnings
from collections import defaultdict

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
    model.eval()
    
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
        
    model.to(DEVICE)
    print(f"✅ Model loaded successfully from {MODEL_PATH} and running on {DEVICE}.\n")

except Exception as e:
    print(f"❌ Error loading model or tokenizer: {e}")
    exit()

# --- Inference Function ---
def classify_code_snippet(code_snippet: str):
    """Tokenizes a single snippet and uses the model to predict its security label."""
    inputs = tokenizer(
        code_snippet,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    predicted_class_id = np.argmax(probabilities)
    
    return {
        "prediction": LABELS[predicted_class_id],
        "confidence": probabilities[predicted_class_id].item(),
        "insecure_confidence": probabilities[1].item(),
        "secure_confidence": probabilities[0].item()
    }

# --- Extensive Test Cases ---

EXTENSIVE_TEST_CASES = {
    # ========== SQL INJECTION - UNDERREPRESENTED PATTERNS ==========
    
    "SQL Injection - UPDATE Statement": {
        "code": """
        app.put('/api/users/:id', (req, res) => {
            const userId = req.params.id;
            const newEmail = req.body.email;
            const query = 'UPDATE users SET email = "' + newEmail + '" WHERE id = ' + userId;
            db.query(query, (err) => {
                res.json({ success: true });
            });
        });
        """,
        "expected": "INSECURE",
        "category": "SQL Injection",
        "subcategory": "UPDATE Statement"
    },
    
    "SQL Injection - INSERT Statement": {
        "code": """
        app.post('/api/users', (req, res) => {
            const username = req.body.username;
            const email = req.body.email;
            const query = 'INSERT INTO users (username, email) VALUES ("' + username + '", "' + email + '")';
            db.query(query, (err) => {
                res.json({ id: db.insertId });
            });
        });
        """,
        "expected": "INSECURE",
        "category": "SQL Injection",
        "subcategory": "INSERT Statement"
    },
    
    "SQL Injection - DELETE Statement": {
        "code": """
        app.delete('/api/posts/:id', (req, res) => {
            const postId = req.params.id;
            const query = 'DELETE FROM posts WHERE id = ' + postId;
            db.query(query, (err) => {
                res.json({ deleted: true });
            });
        });
        """,
        "expected": "INSECURE",
        "category": "SQL Injection",
        "subcategory": "DELETE Statement"
    },
    
    "SQL Injection - req.params Input": {
        "code": """
        app.get('/api/user/:username', (req, res) => {
            const username = req.params.username;
            const query = 'SELECT * FROM users WHERE username = "' + username + '"';
            db.query(query, (err, results) => {
                res.json(results);
            });
        });
        """,
        "expected": "INSECURE",
        "category": "SQL Injection",
        "subcategory": "req.params Input"
    },
    
    "SQL Injection - req.headers Input": {
        "code": """
        app.get('/api/profile', (req, res) => {
            const apiKey = req.headers['x-api-key'];
            const query = 'SELECT * FROM users WHERE api_key = "' + apiKey + '"';
            db.query(query, (err, results) => {
                res.json(results[0]);
            });
        });
        """,
        "expected": "INSECURE",
        "category": "SQL Injection",
        "subcategory": "req.headers Input"
    },
    
    "SQL Injection - req.cookies Input": {
        "code": """
        app.get('/api/session', (req, res) => {
            const sessionId = req.cookies.session_id;
            const query = 'SELECT * FROM sessions WHERE id = "' + sessionId + '"';
            db.query(query, (err, results) => {
                res.json(results[0]);
            });
        });
        """,
        "expected": "INSECURE",
        "category": "SQL Injection",
        "subcategory": "req.cookies Input"
    },
    
    "SQL Injection - Router Pattern": {
        "code": """
        const router = require('express').Router();
        router.get('/search', (req, res) => {
            const term = req.query.q;
            const sql = 'SELECT * FROM products WHERE name LIKE "%' + term + '%"';
            db.query(sql, (err, rows) => res.json(rows));
        });
        module.exports = router;
        """,
        "expected": "INSECURE",
        "category": "SQL Injection",
        "subcategory": "Router Pattern"
    },
    
    "SQL Injection - Module Exports Pattern": {
        "code": """
        module.exports = function(req, res) {
            const userId = req.query.user_id;
            const query = 'SELECT * FROM accounts WHERE user_id = ' + userId;
            db.query(query, (err, data) => {
                res.send(data);
            });
        };
        """,
        "expected": "INSECURE",
        "category": "SQL Injection",
        "subcategory": "Module Exports Pattern"
    },
    
    "SQL Injection - SECURE Parameterized": {
        "code": """
        app.put('/api/users/:id', (req, res) => {
            const userId = req.params.id;
            const newEmail = req.body.email;
            const query = 'UPDATE users SET email = ? WHERE id = ?';
            db.query(query, [newEmail, userId], (err) => {
                res.json({ success: true });
            });
        });
        """,
        "expected": "SECURE",
        "category": "SQL Injection",
        "subcategory": "Secure Parameterized"
    },
    
    # ========== COMMAND INJECTION - UNDERREPRESENTED PATTERNS ==========
    
    "Command Injection - spawn() with req.params": {
        "code": """
        const { spawn } = require('child_process');
        app.post('/api/backup/:filename', (req, res) => {
            const filename = req.params.filename;
            const tar = spawn('tar', ['-czf', filename, './data']);
            tar.on('close', () => res.send('Backup complete'));
        });
        """,
        "expected": "INSECURE",
        "category": "Command Injection",
        "subcategory": "spawn() Pattern"
    },
    
    "Command Injection - execFile() with req.body": {
        "code": """
        const { execFile } = require('child_process');
        app.post('/api/convert', (req, res) => {
            const inputFile = req.body.input;
            execFile('convert', [inputFile, 'output.jpg'], (error) => {
                res.send('Done');
            });
        });
        """,
        "expected": "INSECURE",
        "category": "Command Injection",
        "subcategory": "execFile() Pattern"
    },
    
    "Command Injection - execSync()": {
        "code": """
        const { execSync } = require('child_process');
        app.get('/api/git-status', (req, res) => {
            const branch = req.query.branch;
            const output = execSync('git status ' + branch).toString();
            res.send(output);
        });
        """,
        "expected": "INSECURE",
        "category": "Command Injection",
        "subcategory": "execSync() Pattern"
    },
    
    "Command Injection - Router Pattern": {
        "code": """
        const router = require('express').Router();
        const { exec } = require('child_process');
        router.post('/run', (req, res) => {
            const command = req.body.cmd;
            exec(command, (err, stdout) => res.send(stdout));
        });
        module.exports = router;
        """,
        "expected": "INSECURE",
        "category": "Command Injection",
        "subcategory": "Router Pattern"
    },
    
    "Command Injection - SECURE with Validation": {
        "code": """
        const { spawn } = require('child_process');
        const path = require('path');
        app.post('/api/backup/:filename', (req, res) => {
            const filename = req.params.filename;
            const safeName = path.basename(filename);
            if (!safeName.match(/^[a-zA-Z0-9._-]+$/)) {
                return res.status(400).send('Invalid filename');
            }
            const tar = spawn('tar', ['-czf', safeName, './data']);
            tar.on('close', () => res.send('Backup complete'));
        });
        """,
        "expected": "SECURE",
        "category": "Command Injection",
        "subcategory": "Secure with Validation"
    },
    
    # ========== XSS - UNDERREPRESENTED PATTERNS ==========
    
    "XSS - eval() Function": {
        "code": """
        app.get('/api/execute', (req, res) => {
            const userCode = req.query.code;
            const result = eval(userCode);
            res.json({ result: result });
        });
        """,
        "expected": "INSECURE",
        "category": "XSS",
        "subcategory": "eval() Pattern"
    },
    
    "XSS - setInterval()": {
        "code": """
        app.get('/api/timer', (req, res) => {
            const script = req.query.script;
            res.send('<script>setInterval(function() { ' + script + ' }, 1000);</script>');
        });
        """,
        "expected": "INSECURE",
        "category": "XSS",
        "subcategory": "setInterval() Pattern"
    },
    
    "XSS - Function Constructor": {
        "code": """
        app.post('/api/run', (req, res) => {
            const userFunc = req.body.function;
            const fn = new Function('return ' + userFunc)();
            res.json({ result: fn() });
        });
        """,
        "expected": "INSECURE",
        "category": "XSS",
        "subcategory": "Function Constructor"
    },
    
    "XSS - req.headers Input": {
        "code": """
        app.get('/api/render', (req, res) => {
            const userAgent = req.headers['user-agent'];
            res.send('<div>Your browser: ' + userAgent + '</div>');
        });
        """,
        "expected": "INSECURE",
        "category": "XSS",
        "subcategory": "req.headers Input"
    },
    
    "XSS - SECURE with Escaping": {
        "code": """
        const escapeHtml = require('escape-html');
        app.get('/api/render', (req, res) => {
            const userAgent = req.headers['user-agent'];
            const escaped = escapeHtml(userAgent);
            res.send('<div>Your browser: ' + escaped + '</div>');
        });
        """,
        "expected": "SECURE",
        "category": "XSS",
        "subcategory": "Secure with Escaping"
    },
    
    # ========== NOSQL INJECTION - MORE PATTERNS ==========
    
    "NoSQL Injection - $ne Operator": {
        "code": """
        app.post('/api/users/login', async (req, res) => {
            const username = req.body.username;
            const password = req.body.password;
            const user = await db.collection('users').findOne({
                username: username,
                password: { $ne: password }
            });
            res.json(user);
        });
        """,
        "expected": "INSECURE",
        "category": "NoSQL Injection",
        "subcategory": "$ne Operator"
    },
    
    "NoSQL Injection - $gt Operator": {
        "code": """
        app.get('/api/products', async (req, res) => {
            const minPrice = req.query.min_price;
            const products = await db.collection('products').find({
                price: { $gt: minPrice }
            }).toArray();
            res.json(products);
        });
        """,
        "expected": "INSECURE",
        "category": "NoSQL Injection",
        "subcategory": "$gt Operator"
    },
    
    "NoSQL Injection - req.params": {
        "code": """
        app.get('/api/user/:id', async (req, res) => {
            const userId = req.params.id;
            const user = await db.collection('users').findOne({
                $where: 'this._id == "' + userId + '"'
            });
            res.json(user);
        });
        """,
        "expected": "INSECURE",
        "category": "NoSQL Injection",
        "subcategory": "req.params Input"
    },
    
    "NoSQL Injection - SECURE": {
        "code": """
        app.post('/api/users/login', async (req, res) => {
            const username = String(req.body.username);
            const password = String(req.body.password);
            if (username.length > 100 || password.length > 100) {
                return res.status(400).json({ error: 'Invalid input' });
            }
            const user = await db.collection('users').findOne({
                username: username,
                password: password
            });
            res.json(user);
        });
        """,
        "expected": "SECURE",
        "category": "NoSQL Injection",
        "subcategory": "Secure with Validation"
    },
    
    # ========== SSRF - MORE PATTERNS ==========
    
    "SSRF - PUT Method": {
        "code": """
        const axios = require('axios');
        app.put('/api/fetch', async (req, res) => {
            const url = req.body.url;
            const response = await axios.put(url, req.body.data);
            res.json(response.data);
        });
        """,
        "expected": "INSECURE",
        "category": "SSRF",
        "subcategory": "PUT Method"
    },
    
    "SSRF - DELETE Method": {
        "code": """
        const https = require('https');
        app.delete('/api/delete', (req, res) => {
            const targetUrl = req.query.url;
            https.get(targetUrl, (response) => {
                response.pipe(res);
            });
        });
        """,
        "expected": "INSECURE",
        "category": "SSRF",
        "subcategory": "DELETE Method"
    },
    
    "SSRF - req.headers Input": {
        "code": """
        const axios = require('axios');
        app.get('/api/proxy', async (req, res) => {
            const url = req.headers['x-target-url'];
            const response = await axios.get(url);
            res.json(response.data);
        });
        """,
        "expected": "INSECURE",
        "category": "SSRF",
        "subcategory": "req.headers Input"
    },
    
    "SSRF - SECURE with Whitelist": {
        "code": """
        const axios = require('axios');
        const url = require('url');
        const ALLOWED_DOMAINS = ['api.example.com', 'cdn.example.com'];
        app.get('/api/proxy', async (req, res) => {
            const targetUrl = req.query.url;
            const parsed = url.parse(targetUrl);
            if (!ALLOWED_DOMAINS.includes(parsed.hostname)) {
                return res.status(403).json({ error: 'Domain not allowed' });
            }
            const response = await axios.get(targetUrl);
            res.json(response.data);
        });
        """,
        "expected": "SECURE",
        "category": "SSRF",
        "subcategory": "Secure with Whitelist"
    },
    
    # ========== PATH TRAVERSAL - MORE PATTERNS ==========
    
    "Path Traversal - req.params": {
        "code": """
        const fs = require('fs');
        app.get('/api/file/:path', (req, res) => {
            const filePath = req.params.path;
            const content = fs.readFileSync(filePath, 'utf8');
            res.send(content);
        });
        """,
        "expected": "INSECURE",
        "category": "Path Traversal",
        "subcategory": "req.params Input"
    },
    
    "Path Traversal - req.headers": {
        "code": """
        const fs = require('fs');
        app.get('/api/download', (req, res) => {
            const filename = req.headers['x-filename'];
            const file = fs.readFileSync('./uploads/' + filename);
            res.send(file);
        });
        """,
        "expected": "INSECURE",
        "category": "Path Traversal",
        "subcategory": "req.headers Input"
    },
    
    "Path Traversal - Router Pattern": {
        "code": """
        const router = require('express').Router();
        const fs = require('fs');
        router.get('/file', (req, res) => {
            const path = req.query.path;
            const data = fs.readFileSync(path);
            res.send(data);
        });
        module.exports = router;
        """,
        "expected": "INSECURE",
        "category": "Path Traversal",
        "subcategory": "Router Pattern"
    },
    
    "Path Traversal - SECURE": {
        "code": """
        const fs = require('fs');
        const path = require('path');
        app.get('/api/file/:filename', (req, res) => {
            const filename = req.params.filename;
            const safePath = path.join(__dirname, 'public', path.basename(filename));
            if (!safePath.startsWith(path.join(__dirname, 'public'))) {
                return res.status(403).send('Access denied');
            }
            const content = fs.readFileSync(safePath, 'utf8');
            res.send(content);
        });
        """,
        "expected": "SECURE",
        "category": "Path Traversal",
        "subcategory": "Secure with Normalization"
    },
    
    # ========== REDOS - MORE PATTERNS ==========
    
    "ReDoS - Nested Quantifiers": {
        "code": """
        app.post('/api/validate', (req, res) => {
            const input = req.body.input;
            const regex = /^(a+)+$/;
            if (regex.test(input)) {
                res.json({ valid: true });
            } else {
                res.json({ valid: false });
            }
        });
        """,
        "expected": "INSECURE",
        "category": "ReDoS",
        "subcategory": "Nested Quantifiers"
    },
    
    "ReDoS - Alternation with Quantifiers": {
        "code": """
        app.get('/api/check', (req, res) => {
            const text = req.query.text;
            const pattern = /(a|a+)+b/;
            const match = text.match(pattern);
            res.json({ found: !!match });
        });
        """,
        "expected": "INSECURE",
        "category": "ReDoS",
        "subcategory": "Alternation Pattern"
    },
    
    "ReDoS - SECURE with Timeout": {
        "code": """
        app.post('/api/validate', (req, res) => {
            const input = req.body.input;
            if (input && input.length > 1000) {
                return res.status(400).json({ error: 'Input too long' });
            }
            const regex = /^[a-zA-Z0-9]+$/;
            if (regex.test(input)) {
                res.json({ valid: true });
            } else {
                res.json({ valid: false });
            }
        });
        """,
        "expected": "SECURE",
        "category": "ReDoS",
        "subcategory": "Secure with Length Limit"
    },
    
    # ========== EDGE CASES AND COMPLEX SCENARIOS ==========
    
    "Edge Case - Multiple Input Sources": {
        "code": """
        app.put('/api/update/:id', (req, res) => {
            const id = req.params.id;
            const name = req.body.name;
            const email = req.query.email;
            const query = 'UPDATE users SET name = "' + name + '", email = "' + email + '" WHERE id = ' + id;
            db.query(query, (err) => res.json({ success: true }));
        });
        """,
        "expected": "INSECURE",
        "category": "Edge Cases",
        "subcategory": "Multiple Input Sources"
    },
    
    "Edge Case - Nested Object Access": {
        "code": """
        app.post('/api/process', (req, res) => {
            const userInput = req.body.data.user.value;
            const query = 'SELECT * FROM table WHERE column = "' + userInput + '"';
            db.query(query, (err, results) => res.json(results));
        });
        """,
        "expected": "INSECURE",
        "category": "Edge Cases",
        "subcategory": "Nested Object Access"
    },
    
    "Edge Case - Array Access": {
        "code": """
        app.get('/api/item', (req, res) => {
            const index = req.query.index;
            const items = ['item1', 'item2', 'item3'];
            const query = 'SELECT * FROM items WHERE name = "' + items[index] + '"';
            db.query(query, (err, results) => res.json(results));
        });
        """,
        "expected": "INSECURE",
        "category": "Edge Cases",
        "subcategory": "Array Access"
    },
    
    "Edge Case - Template Literal in SQL": {
        "code": """
        app.get('/api/search', (req, res) => {
            const term = req.query.term;
            const table = req.query.table;
            const query = `SELECT * FROM ${table} WHERE name = '${term}'`;
            db.query(query, (err, results) => res.json(results));
        });
        """,
        "expected": "INSECURE",
        "category": "Edge Cases",
        "subcategory": "Template Literal SQL"
    },
    
    "Edge Case - Chained Vulnerabilities": {
        "code": """
        app.post('/api/process', (req, res) => {
            const filename = req.body.filename;
            const { exec } = require('child_process');
            exec('cat ' + filename, (err, stdout) => {
                res.send(stdout);
            });
        });
        """,
        "expected": "INSECURE",
        "category": "Edge Cases",
        "subcategory": "Command + Path Traversal"
    },
    
    "Edge Case - SECURE Complex": {
        "code": """
        const express = require('express');
        const { body, param, validationResult } = require('express-validator');
        app.put('/api/update/:id', 
            param('id').isInt().toInt(),
            body('name').isString().isLength({ min: 1, max: 100 }).escape(),
            body('email').isEmail().normalizeEmail(),
            async (req, res) => {
                const errors = validationResult(req);
                if (!errors.isEmpty()) {
                    return res.status(400).json({ errors: errors.array() });
                }
                const id = req.params.id;
                const name = req.body.name;
                const email = req.body.email;
                await db.query('UPDATE users SET name = ?, email = ? WHERE id = ?', [name, email, id]);
                res.json({ success: true });
            }
        );
        """,
        "expected": "SECURE",
        "category": "Edge Cases",
        "subcategory": "Complex Secure Example"
    },
    
    # ========== DIFFERENT HTTP METHODS ==========
    
    "HTTP Method - PATCH": {
        "code": """
        app.patch('/api/users/:id', (req, res) => {
            const userId = req.params.id;
            const field = req.body.field;
            const value = req.body.value;
            const query = 'UPDATE users SET ' + field + ' = "' + value + '" WHERE id = ' + userId;
            db.query(query, (err) => res.json({ updated: true }));
        });
        """,
        "expected": "INSECURE",
        "category": "HTTP Methods",
        "subcategory": "PATCH Method"
    },
    
    "HTTP Method - DELETE": {
        "code": """
        app.delete('/api/comments/:id', (req, res) => {
            const commentId = req.params.id;
            const query = 'DELETE FROM comments WHERE id = ' + commentId;
            db.query(query, (err) => res.json({ deleted: true }));
        });
        """,
        "expected": "INSECURE",
        "category": "HTTP Methods",
        "subcategory": "DELETE Method"
    },
    
    # ========== DIFFERENT CODE STRUCTURES ==========
    
    "Code Structure - Class Based": {
        "code": """
        class UserController {
            async getUser(req, res) {
                const userId = req.params.id;
                const query = 'SELECT * FROM users WHERE id = ' + userId;
                db.query(query, (err, results) => res.json(results[0]));
            }
        }
        app.get('/api/user/:id', (req, res) => new UserController().getUser(req, res));
        """,
        "expected": "INSECURE",
        "category": "Code Structure",
        "subcategory": "Class Based"
    },
    
    "Code Structure - Middleware Pattern": {
        "code": """
        function processRequest(req, res, next) {
            const userInput = req.query.input;
            req.processed = 'SELECT * FROM data WHERE value = "' + userInput + '"';
            next();
        }
        app.get('/api/data', processRequest, (req, res) => {
            db.query(req.processed, (err, results) => res.json(results));
        });
        """,
        "expected": "INSECURE",
        "category": "Code Structure",
        "subcategory": "Middleware Pattern"
    },
    
    "Code Structure - Arrow Function": {
        "code": """
        const handler = (req, res) => {
            const id = req.query.id;
            db.query('SELECT * FROM items WHERE id = ' + id, (err, data) => res.json(data));
        };
        app.get('/api/item', handler);
        """,
        "expected": "INSECURE",
        "category": "Code Structure",
        "subcategory": "Arrow Function"
    },
    
    # ========== OBFUSCATED PATTERNS ==========
    
    "Obfuscated - Variable Reassignment": {
        "code": """
        app.get('/api/search', (req, res) => {
            let q = req.query.q;
            let query = 'SELECT * FROM products WHERE name LIKE "%';
            query = query + q;
            query = query + '%"';
            db.query(query, (err, results) => res.json(results));
        });
        """,
        "expected": "INSECURE",
        "category": "Obfuscated",
        "subcategory": "Variable Reassignment"
    },
    
    "Obfuscated - Function Wrapper": {
        "code": """
        function buildQuery(term) {
            return 'SELECT * FROM users WHERE username = "' + term + '"';
        }
        app.get('/api/user', (req, res) => {
            const username = req.query.username;
            const query = buildQuery(username);
            db.query(query, (err, results) => res.json(results));
        });
        """,
        "expected": "INSECURE",
        "category": "Obfuscated",
        "subcategory": "Function Wrapper"
    },
    
    "Obfuscated - SECURE Despite Wrapper": {
        "code": """
        function buildQuery(term) {
            return { username: term };
        }
        app.get('/api/user', async (req, res) => {
            const username = req.query.username;
            const query = buildQuery(username);
            const user = await db.collection('users').findOne(query);
            res.json(user);
        });
        """,
        "expected": "SECURE",
        "category": "Obfuscated",
        "subcategory": "Secure Function Wrapper"
    },
    
    # ========== BOUNDARY CASES ==========
    
    "Boundary - Empty String": {
        "code": """
        app.get('/api/search', (req, res) => {
            const term = req.query.term || '';
            const query = 'SELECT * FROM products WHERE name = "' + term + '"';
            db.query(query, (err, results) => res.json(results));
        });
        """,
        "expected": "INSECURE",
        "category": "Boundary Cases",
        "subcategory": "Empty String"
    },
    
    "Boundary - Null Check": {
        "code": """
        app.get('/api/user', (req, res) => {
            const id = req.query.id;
            if (id) {
                const query = 'SELECT * FROM users WHERE id = ' + id;
                db.query(query, (err, results) => res.json(results));
            }
        });
        """,
        "expected": "INSECURE",
        "category": "Boundary Cases",
        "subcategory": "Null Check"
    },
    
    "Boundary - SECURE with Proper Validation": {
        "code": """
        app.get('/api/user', (req, res) => {
            const id = req.query.id;
            if (!id || !Number.isInteger(parseInt(id))) {
                return res.status(400).json({ error: 'Invalid ID' });
            }
            db.query('SELECT * FROM users WHERE id = ?', [id], (err, results) => {
                res.json(results);
            });
        });
        """,
        "expected": "SECURE",
        "category": "Boundary Cases",
        "subcategory": "Secure with Validation"
    }
}

# --- Main Execution ---

if __name__ == '__main__':
    
    print("=" * 80)
    print("  EXTENSIVE TEST SUITE - JavaScript Vulnerability Detection Model")
    print("=" * 80)
    print(f"\nTotal Test Cases: {len(EXTENSIVE_TEST_CASES)}\n")
    
    results_summary = {
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "by_category": defaultdict(lambda: {"correct": 0, "total": 0}),
        "by_subcategory": defaultdict(lambda: {"correct": 0, "total": 0}),
        "false_positives": 0,
        "false_negatives": 0
    }
    
    # Run all tests
    for test_name, test_case in EXTENSIVE_TEST_CASES.items():
        code = test_case["code"]
        expected = test_case["expected"]
        category = test_case.get("category", "Unknown")
        subcategory = test_case.get("subcategory", "Unknown")
        
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
                status = "✅"
            else:
                results_summary["incorrect"] += 1
                if expected == "SECURE" and predicted_label == "INSECURE (Vulnerable)":
                    results_summary["false_positives"] += 1
                elif expected == "INSECURE" and predicted_label == "SECURE":
                    results_summary["false_negatives"] += 1
                status = "❌"
            
            # Track by category and subcategory
            results_summary["by_category"][category]["total"] += 1
            results_summary["by_subcategory"][f"{category} - {subcategory}"]["total"] += 1
            if is_correct:
                results_summary["by_category"][category]["correct"] += 1
                results_summary["by_subcategory"][f"{category} - {subcategory}"]["correct"] += 1
            
            # Print result (only show incorrect predictions in detail)
            if not is_correct:
                print(f"{status} [{category}] {test_name}")
                print(f"   Expected: {expected}, Got: {predicted_label}")
                print(f"   Confidence: {confidence:.4f}, Insecure Score: {insecure_score:.4f}")
            
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {str(e)}")
            results_summary["total"] += 1
            results_summary["incorrect"] += 1
    
    # Print Summary
    print("\n" + "=" * 80)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nTotal Tests: {results_summary['total']}")
    print(f"Correct Predictions: {results_summary['correct']} ({results_summary['correct']/results_summary['total']*100:.2f}%)")
    print(f"Incorrect Predictions: {results_summary['incorrect']} ({results_summary['incorrect']/results_summary['total']*100:.2f}%)")
    print(f"False Positives (Secure flagged as Insecure): {results_summary['false_positives']}")
    print(f"False Negatives (Insecure missed): {results_summary['false_negatives']}")
    
    print("\n--- Performance by Category ---")
    for category in sorted(results_summary["by_category"].keys()):
        stats = results_summary["by_category"][category]
        accuracy = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {category}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
    
    print("\n--- Performance by Subcategory (Top 10 Worst) ---")
    subcat_stats = sorted(
        results_summary["by_subcategory"].items(),
        key=lambda x: x[1]["correct"] / x[1]["total"] if x[1]["total"] > 0 else 0
    )[:10]
    for subcat, stats in subcat_stats:
        accuracy = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {subcat}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
    
    print("\n" + "=" * 80)

