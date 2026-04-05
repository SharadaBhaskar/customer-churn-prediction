from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import sqlite3
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# ── Create folders if they don't exist ──────────────────────────────────────
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
os.makedirs(app.config['CHARTS_FOLDER'], exist_ok=True)

# ── Database setup ───────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(app.config['DATABASE_PATH'])
    cursor = conn.cursor()

    # Table for single predictions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            tenure INTEGER,
            monthly_charges REAL,
            total_charges REAL,
            contract_type TEXT,
            internet_service TEXT,
            payment_method TEXT,
            churn_prediction TEXT,
            churn_probability REAL,
            risk_badge TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Table for bulk predictions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bulk_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            total_customers INTEGER,
            churn_count INTEGER,
            non_churn_count INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/bulk_upload')
def bulk_upload():
    return render_template('bulk_upload.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/reports')
def reports():
    return render_template('reports.html')

@app.route('/about')
def about():
    return render_template('about.html')

# ── Run the app ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    init_db()
    print("✅ Database initialized successfully!")
    print("🚀 Starting Customer Churn Prediction App...")
    print("🌐 Open your browser and go to: http://127.0.0.1:5000")
    app.run(debug=True)