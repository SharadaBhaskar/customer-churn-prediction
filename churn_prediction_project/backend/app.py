from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import os
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "../templates"),
    static_folder=os.path.join(BASE_DIR, "../static")
)

app.secret_key = "churnpredict_secret_key"

# ---------------- AUTO INJECT current_user TO ALL TEMPLATES ----------------
@app.context_processor
def inject_user():
    return dict(current_user=None, customer=None)

# ---------------- LOAD MODEL ONCE ----------------
model = joblib.load(os.path.join(BASE_DIR, "../models/churn_model.pkl"))

# ---------------- DASHBOARD ----------------
@app.route("/")
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form.get("email")
        password = request.form.get("password")
        if email and password:
            session["user"] = email
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password", "error")
    return render_template("auth/login.html")

# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        full_name = request.form.get("full_name")
        email     = request.form.get("email")
        password  = request.form.get("password")
        confirm   = request.form.get("confirm_password")
        if password != confirm:
            flash("Passwords do not match", "error")
        elif full_name and email and password:
            flash("Account created successfully! Please login.", "success")
            return redirect(url_for("login"))
        else:
            flash("Please fill all required fields", "error")
    return render_template("auth/register.html")

# ---------------- CUSTOMERS ----------------
@app.route("/customers")
def customers():
    return render_template("customer/customer.html")

@app.route("/customers/add", methods=["GET", "POST"])
def add_customer():
    if request.method == "POST":
        flash("Customer added successfully!", "success")
        return redirect(url_for("customers"))
    return render_template("customer/add_customer.html")

@app.route("/customers/update/<int:customer_id>", methods=["GET", "POST"])
def update_customer(customer_id):
    if request.method == "POST":
        flash("Customer updated successfully!", "success")
        return redirect(url_for("customers"))
    return render_template("customer/update_customer.html")

# ---------------- CHURN PREDICTION ----------------
@app.route("/predict", methods=["GET"])
def prediction():
    return render_template("churn/prediction.html")

@app.route("/predict/result")
def prediction_result():
    return render_template("churn/prediction_result.html", churn_probability=0)

# ---------------- SEGMENTATION ----------------
@app.route("/segmentation")
def segmentation():
    return render_template("segmentation/segmentation.html")

@app.route("/retention")
def retention():
    return render_template("segmentation/retention.html")

@app.route("/sms-alert", methods=["GET", "POST"])
def sms_alert():
    if request.method == "POST":
        flash("SMS sent successfully!", "success")
    return render_template("segmentation/sms_alert.html")

# ---------------- CAMPAIGNS ----------------
@app.route("/campaigns", methods=["GET", "POST"])
def campaigns():
    if request.method == "POST":
        flash("Campaign created successfully!", "success")
        return redirect(url_for("campaign_results"))
    return render_template("campaign/campaign.html")

@app.route("/campaigns/results")
def campaign_results():
    return render_template("campaign/campaign_results.html")

# ---------------- ANALYTICS & REPORTS ----------------
@app.route("/analytics")
def analytics():
    return render_template("analytics/analytics.html")

@app.route("/reports", methods=["GET", "POST"])
def reports():
    if request.method == "POST":
        flash("Report generated!", "success")
    return render_template("analytics/reports.html")

# ---------------- FEEDBACK ----------------
@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if request.method == "POST":
        flash("Thank you for your feedback!", "success")
    return render_template("feedback.html")

# ---------------- PROFILE & SETTINGS ----------------
@app.route("/profile", methods=["GET", "POST"])
def profile():
    if request.method == "POST":
        flash("Profile updated successfully!", "success")
    return render_template("user/profile.html")

@app.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        flash("Settings saved!", "success")
    return render_template("user/settings.html")

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------- ML PREDICTION API ----------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = np.array([[
        float(data["tenure"]),
        float(data["monthly_charges"]),
        float(data["total_charges"])
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1] * 100

    if probability > 70:
        label = "High Risk - Will Churn"
    elif probability > 40:
        label = "Medium Risk"
    else:
        label = "Low Risk"

    return jsonify({
        "prediction": label,
        "probability": round(probability, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)