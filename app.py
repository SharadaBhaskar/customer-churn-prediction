from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import sqlite3
import json
from config import Config
from predict_engine import predict_churn, get_all_predictions

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

# ── Predict Route ─────────────────────────────────────────────────────────────
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        customer_data = {
            'customer_id':      request.form.get('customer_id'),
            'gender':           request.form.get('gender', 'Male'),
            'senior_citizen':   request.form.get('senior_citizen', 0),
            'partner':          request.form.get('partner', 'No'),
            'dependents':       request.form.get('dependents', 'No'),
            'tenure':           request.form.get('tenure', 0),
            'phone_service':    request.form.get('phone_service', 'Yes'),
            'multiple_lines':   request.form.get('multiple_lines',
                                                  'No phone service'),
            'internet_service': request.form.get('internet_service', 'DSL'),
            'online_security':  request.form.get('online_security', 'No'),
            'online_backup':    request.form.get('online_backup', 'No'),
            'device_protection':request.form.get('device_protection', 'No'),
            'tech_support':     request.form.get('tech_support', 'No'),
            'streaming_tv':     request.form.get('streaming_tv', 'No'),
            'streaming_movies': request.form.get('streaming_movies', 'No'),
            'contract_type':    request.form.get('contract_type',
                                                  'Month-to-month'),
            'paperless_billing':request.form.get('paperless_billing', 'Yes'),
            'payment_method':   request.form.get('payment_method',
                                                  'Electronic check'),
            'monthly_charges':  request.form.get('monthly_charges', 0),
            'total_charges':    request.form.get('total_charges', 0),
            'num_products':     request.form.get('num_products', 1)
        }

        result = predict_churn(customer_data)

        if result['success']:
            return render_template('result.html', result=result)
        else:
            flash(f"Prediction error: {result['error']}", 'danger')
            return redirect(url_for('predict'))

    return render_template('predict.html')

# ── Dashboard Route ───────────────────────────────────────────────────────────
@app.route('/dashboard')
def dashboard():
    predictions = get_all_predictions()
    total       = len(predictions)
    high_risk   = sum(1 for p in predictions if p['risk_badge'] == 'High')
    low_risk    = sum(1 for p in predictions if p['risk_badge'] == 'Low')
    churn_rate  = round((high_risk / total * 100), 1) if total > 0 else 0
    return render_template('dashboard.html',
                           predictions=predictions,
                           total=total,
                           high_risk=high_risk,
                           low_risk=low_risk,
                           churn_rate=churn_rate)

# ── Bulk Upload Route ─────────────────────────────────────────────────────────
@app.route('/bulk_upload', methods=['GET', 'POST'])
def bulk_upload():
    return render_template('bulk_upload.html')

# ── Reports Route ─────────────────────────────────────────────────────────────
@app.route('/reports')
def reports():
    predictions = get_all_predictions()
    return render_template('reports.html', predictions=predictions)

# ── About Route ───────────────────────────────────────────────────────────────
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