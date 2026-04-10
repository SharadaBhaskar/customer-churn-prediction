import pandas as pd
import numpy as np
import joblib
import json
import os
import sqlite3
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
DB_PATH    = os.path.join(BASE_DIR, 'database', 'churn.db')

# ── Load Model & Tools ────────────────────────────────────────────────────────
print("🔄 Loading model and tools...")
model          = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
scaler         = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))

with open(os.path.join(MODEL_DIR, 'feature_names.json')) as f:
    feature_names = json.load(f)

with open(os.path.join(MODEL_DIR, 'feature_importances.json')) as f:
    feature_importances = json.load(f)

print("✅ Model loaded successfully!")

# ── Risk Badge ────────────────────────────────────────────────────────────────
def get_risk_badge(probability):
    if probability < 30:
        return 'Low'
    elif probability < 60:
        return 'Medium'
    else:
        return 'High'

# ── Retention Actions ─────────────────────────────────────────────────────────
def get_retention_actions(risk_badge, contract_type, monthly_charges):
    actions = []
    if risk_badge == 'High':
        actions.append("📞 Call customer immediately")
        actions.append("💰 Offer 20% discount on next 3 months")
        actions.append("⬆️ Suggest plan upgrade with extra benefits")
        actions.append("📧 Send personalized retention email")
        if monthly_charges > 70:
            actions.append("🎁 Offer free add-on service for 1 month")
    elif risk_badge == 'Medium':
        actions.append("📧 Send special offer email")
        actions.append("💳 Offer loyalty reward points")
        actions.append("📋 Suggest switching to annual contract")
        if contract_type == 'Month-to-month':
            actions.append("📅 Offer discount for 1-year contract")
    else:
        actions.append("✅ Customer is stable — monitor monthly")
        actions.append("🌟 Send satisfaction survey")
        actions.append("🎯 Offer referral program")
    return actions

# ── Discount Generator ────────────────────────────────────────────────────────
def get_discount(risk_badge, monthly_charges):
    if risk_badge == 'High':
        discount_pct    = 20
        discount_amount = round(monthly_charges * 0.20, 2)
        return {
            'percentage': discount_pct,
            'amount':     discount_amount,
            'message':    f"Offer ${discount_amount} off "
                          f"({discount_pct}% discount) immediately!"
        }
    elif risk_badge == 'Medium':
        discount_pct    = 10
        discount_amount = round(monthly_charges * 0.10, 2)
        return {
            'percentage': discount_pct,
            'amount':     discount_amount,
            'message':    f"Offer ${discount_amount} off "
                          f"({discount_pct}% discount) as loyalty reward."
        }
    else:
        return {
            'percentage': 0,
            'amount':     0,
            'message':    "No discount needed — customer is low risk."
        }

# ── Revenue Loss Estimation ───────────────────────────────────────────────────
def get_revenue_loss(probability, monthly_charges, tenure):
    avg_lifetime     = 24
    remaining_months = max(avg_lifetime - tenure, 6)
    churn_prob       = probability / 100
    potential_loss   = monthly_charges * remaining_months
    expected_loss    = round(churn_prob * potential_loss, 2)

    if expected_loss > 1000:
        priority = 'Critical'
        color    = 'danger'
    elif expected_loss > 500:
        priority = 'High'
        color    = 'warning'
    elif expected_loss > 200:
        priority = 'Medium'
        color    = 'info'
    else:
        priority = 'Low'
        color    = 'success'

    return {
        'monthly_charges':  monthly_charges,
        'remaining_months': remaining_months,
        'potential_loss':   round(potential_loss, 2),
        'expected_loss':    expected_loss,
        'priority':         priority,
        'color':            color
    }

# ── Loyalty Score ─────────────────────────────────────────────────────────────
def get_loyalty_score(tenure, monthly_charges):
    tenure_score  = min(tenure / 72 * 50, 50)
    charges_score = min(monthly_charges / 120 * 50, 50)
    score         = round(tenure_score + charges_score, 1)
    if score >= 75:
        level = "Platinum"
    elif score >= 50:
        level = "Gold"
    elif score >= 25:
        level = "Silver"
    else:
        level = "Bronze"
    return {'score': score, 'level': level}

# ── Customer Segmentation ─────────────────────────────────────────────────────
def get_customer_segment(tenure, monthly_charges, probability):
    if tenure > 48 and monthly_charges > 60:
        segment = "Loyal"
        color   = "success"
        desc    = "Long-term high-value customer"
    elif probability > 60:
        segment = "At Risk"
        color   = "danger"
        desc    = "High churn probability — act now"
    elif tenure < 6:
        segment = "New"
        color   = "info"
        desc    = "Recently joined — needs onboarding"
    elif monthly_charges < 30:
        segment = "Budget"
        color   = "secondary"
        desc    = "Low spender — offer upgrade"
    else:
        segment = "Regular"
        color   = "primary"
        desc    = "Stable customer — maintain engagement"
    return {'segment': segment, 'color': color, 'desc': desc}

# ── Retention Campaign Generator ──────────────────────────────────────────────
def get_retention_campaign(risk_badge, segment, monthly_charges, tenure):
    campaigns = []
    if risk_badge == 'High':
        campaigns.append({
            'type':    '📱 SMS Campaign',
            'message': f"Special offer just for you! "
                       f"Get 20% off your next bill. "
                       f"Reply YES to activate.",
            'timing':  'Send immediately'
        })
        campaigns.append({
            'type':    '📧 Email Campaign',
            'message': f"We value your {tenure} months "
                       f"with us! Here's an exclusive "
                       f"retention offer worth "
                       f"${round(monthly_charges*0.2,2)}/month.",
            'timing':  'Send within 24 hours'
        })
        campaigns.append({
            'type':    '📞 Call Campaign',
            'message': "Personal call from retention "
                       "team to discuss upgrade options.",
            'timing':  'Call within 48 hours'
        })
    elif risk_badge == 'Medium':
        campaigns.append({
            'type':    '📧 Email Campaign',
            'message': "You're a valued customer! "
                       "Enjoy 10% loyalty discount "
                       "this month.",
            'timing':  'Send within 3 days'
        })
        campaigns.append({
            'type':    '📱 SMS Campaign',
            'message': "Upgrade to annual plan and "
                       "save 15% every month!",
            'timing':  'Send within 1 week'
        })
    else:
        campaigns.append({
            'type':    '📧 Newsletter',
            'message': "Monthly newsletter with "
                       "tips and exclusive member "
                       "benefits.",
            'timing':  'Send monthly'
        })
    return campaigns

# ── Main Prediction Function ──────────────────────────────────────────────────
def predict_churn(customer_data):
    try:
        # Build input dataframe
        input_df = pd.DataFrame([{
            'gender':           customer_data.get('gender', 'Male'),
            'SeniorCitizen':    int(customer_data.get('senior_citizen', 0)),
            'Partner':          customer_data.get('partner', 'No'),
            'Dependents':       customer_data.get('dependents', 'No'),
            'tenure':           int(customer_data.get('tenure', 0)),
            'PhoneService':     customer_data.get('phone_service', 'Yes'),
            'MultipleLines':    customer_data.get('multiple_lines',
                                                   'No phone service'),
            'InternetService':  customer_data.get('internet_service', 'DSL'),
            'OnlineSecurity':   customer_data.get('online_security', 'No'),
            'OnlineBackup':     customer_data.get('online_backup', 'No'),
            'DeviceProtection': customer_data.get('device_protection', 'No'),
            'TechSupport':      customer_data.get('tech_support', 'No'),
            'StreamingTV':      customer_data.get('streaming_tv', 'No'),
            'StreamingMovies':  customer_data.get('streaming_movies', 'No'),
            'Contract':         customer_data.get('contract_type',
                                                   'Month-to-month'),
            'PaperlessBilling': customer_data.get('paperless_billing', 'Yes'),
            'PaymentMethod':    customer_data.get('payment_method',
                                                   'Electronic check'),
            'MonthlyCharges':   float(customer_data.get('monthly_charges', 0)),
            'TotalCharges':     float(customer_data.get('total_charges', 0))
        }])

        # Encode categorical columns
        for col, le in label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = le.transform(input_df[col])
                except ValueError:
                    input_df[col] = 0

        # Reorder columns to match training
        input_df = input_df[feature_names]

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        probability  = model.predict_proba(input_scaled)[0][1] * 100
        prediction   = 'Churn' if probability >= 50 else 'No Churn'
        risk_badge   = get_risk_badge(probability)

        # Extra insights
        monthly_charges = float(customer_data.get('monthly_charges', 0))
        contract_type   = customer_data.get('contract_type', 'Month-to-month')
        tenure          = int(customer_data.get('tenure', 0))

        actions      = get_retention_actions(risk_badge,
                                              contract_type,
                                              monthly_charges)
        discount     = get_discount(risk_badge, monthly_charges)
        loyalty      = get_loyalty_score(tenure, monthly_charges)
        revenue_loss = get_revenue_loss(probability,
                                         monthly_charges,
                                         tenure)
        segment      = get_customer_segment(tenure,
                                             monthly_charges,
                                             probability)
        campaigns    = get_retention_campaign(risk_badge,
                                               segment['segment'],
                                               monthly_charges,
                                               tenure)
        top_features = list(feature_importances.keys())[:5]

        # Save to database
        save_prediction(customer_data, prediction,
                        round(probability, 2), risk_badge)

        return {
            'success':      True,
            'customer_id':  customer_data.get('customer_id', 'N/A'),
            'prediction':   prediction,
            'probability':  round(probability, 2),
            'risk_badge':   risk_badge,
            'actions':      actions,
            'discount':     discount,
            'loyalty':      loyalty,
            'revenue_loss': revenue_loss,
            'segment':      segment,
            'campaigns':    campaigns,
            'top_features': top_features
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}

# ── Save to Database ──────────────────────────────────────────────────────────
def save_prediction(customer_data, prediction, probability, risk_badge):
    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (
                customer_id, tenure, monthly_charges, total_charges,
                contract_type, internet_service, payment_method,
                churn_prediction, churn_probability, risk_badge
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            customer_data.get('customer_id', 'N/A'),
            customer_data.get('tenure', 0),
            customer_data.get('monthly_charges', 0),
            customer_data.get('total_charges', 0),
            customer_data.get('contract_type', ''),
            customer_data.get('internet_service', ''),
            customer_data.get('payment_method', ''),
            prediction,
            probability,
            risk_badge
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")

# ── Get All Predictions ───────────────────────────────────────────────────────
def get_all_predictions():
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY timestamp DESC",
            conn
        )
        conn.close()
        return df.to_dict('records')
    except Exception as e:
        print(f"DB Error: {e}")
        return []