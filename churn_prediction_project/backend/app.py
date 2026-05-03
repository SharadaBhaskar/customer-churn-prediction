from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, Response
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os, sqlite3
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "../database.db")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "../templates"),
    static_folder=os.path.join(BASE_DIR, "../static")
)
app.secret_key = "churnpredict_secret_2025"

# ════════════════════════════════════════
#  DATABASE
# ════════════════════════════════════════
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name     TEXT NOT NULL,
        email         TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role          TEXT DEFAULT 'Customer',
        phone         TEXT DEFAULT '',
        created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS customers (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name       TEXT NOT NULL,
        email           TEXT,
        phone           TEXT,
        age             INTEGER,
        plan_type       TEXT,
        tenure          INTEGER DEFAULT 0,
        monthly_charges REAL DEFAULT 0,
        total_charges   REAL DEFAULT 0,
        contract_type   TEXT,
        payment_method  TEXT,
        support_calls   INTEGER DEFAULT 0,
        status          TEXT DEFAULT 'Active',
        created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id     INTEGER,
        customer_name   TEXT,
        tenure          REAL,
        monthly_charges REAL,
        total_charges   REAL,
        contract_type   TEXT,
        support_calls   INTEGER,
        probability     REAL,
        risk_level      TEXT,
        created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS campaigns (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        campaign_name  TEXT NOT NULL,
        target_segment TEXT,
        campaign_type  TEXT,
        message        TEXT,
        start_date     TEXT,
        end_date       TEXT,
        status         TEXT DEFAULT 'Active',
        sent           INTEGER DEFAULT 0,
        responses      INTEGER DEFAULT 0,
        created_at     DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id       INTEGER,
        name          TEXT,
        rating        INTEGER,
        category      TEXT,
        feedback_text TEXT,
        created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")

    exists = c.execute("SELECT id FROM users WHERE email='admin@churnpredict.com'").fetchone()
    if not exists:
        c.execute("INSERT INTO users (full_name,email,password_hash,role) VALUES (?,?,?,?)",
            ("Admin User","admin@churnpredict.com",generate_password_hash("admin123"),"Admin"))
        print("✅ Admin created: admin@churnpredict.com / admin123")

    conn.commit()
    conn.close()

# ════════════════════════════════════════
#  DECORATORS
# ════════════════════════════════════════
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login first.", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def staff_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login first.", "error")
            return redirect(url_for("login"))
        if session.get("user_role") not in ["Admin","Manager","Analyst"]:
            flash("Access denied. Staff only.", "error")
            return redirect(url_for("customer_feedback"))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("user_role") != "Admin":
            flash("Admin access required.", "error")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return decorated

# ════════════════════════════════════════
#  CONTEXT PROCESSOR
# ════════════════════════════════════════
@app.context_processor
def inject_globals():
    current_user = None
    if "user_id" in session:
        conn = get_db()
        current_user = conn.execute(
            "SELECT * FROM users WHERE id=?", (session["user_id"],)
        ).fetchone()
        conn.close()
    return dict(current_user=current_user, customer=None)

# ════════════════════════════════════════
#  LOAD MODEL
# ════════════════════════════════════════
model = joblib.load(os.path.join(BASE_DIR, "../models/churn_model.pkl"))

# ════════════════════════════════════════
#  AUTH
# ════════════════════════════════════════
@app.route("/login", methods=["GET","POST"])
def login():
    if "user_id" in session:
        if session.get("user_role") == "Customer":
            return redirect(url_for("customer_feedback"))
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email    = request.form.get("email","").strip().lower()
        password = request.form.get("password","")
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE LOWER(email)=?", (email,)).fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"]    = user["id"]
            session["user_name"]  = user["full_name"]
            session["user_role"]  = user["role"]
            session["user_email"] = user["email"]
            if user["role"] == "Customer":
                return redirect(url_for("customer_feedback"))
            else:
                return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password. Please try again.", "error")

    return render_template("auth/login.html")


@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        full_name = request.form.get("full_name","").strip()
        email     = request.form.get("email","").strip().lower()
        password  = request.form.get("password","")
        confirm   = request.form.get("confirm_password","")

        if not full_name or not email or not password:
            flash("Please fill all required fields.", "error")
        elif password != confirm:
            flash("Passwords do not match.", "error")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
        else:
            conn = get_db()
            exists = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
            if exists:
                flash("Email already registered. Please login.", "error")
                conn.close()
            else:
                conn.execute(
                    "INSERT INTO users (full_name,email,password_hash,role) VALUES (?,?,?,?)",
                    (full_name, email, generate_password_hash(password), "Customer")
                )
                conn.commit()
                conn.close()
                flash("Account created! You can now login.", "success")
                return redirect(url_for("login"))

    return render_template("auth/register.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))

# ════════════════════════════════════════
#  CUSTOMER FEEDBACK PAGE
# ════════════════════════════════════════
@app.route("/customer/feedback", methods=["GET","POST"])
@login_required
def customer_feedback():
    if session.get("user_role") in ["Admin","Manager","Analyst"]:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        feedback_text = request.form.get("feedback_text","").strip()
        rating        = request.form.get("rating", 5)
        category      = request.form.get("category","General")

        if not feedback_text:
            flash("Please write your feedback before submitting.", "error")
        else:
            conn = get_db()
            conn.execute(
                "INSERT INTO feedback (user_id,name,rating,category,feedback_text) VALUES (?,?,?,?,?)",
                (session["user_id"], session["user_name"], rating, category, feedback_text)
            )
            conn.commit()
            conn.close()
            flash("Thank you for your feedback! 🎉", "success")
            return redirect(url_for("customer_feedback"))

    conn = get_db()
    feedbacks = conn.execute("SELECT * FROM feedback ORDER BY created_at DESC").fetchall()
    conn.close()
    return render_template("feedback/customer_feedback.html", feedbacks=feedbacks)


@app.route("/api/feedback")
@login_required
def api_feedback():
    conn = get_db()
    rows = conn.execute("SELECT * FROM feedback ORDER BY created_at DESC LIMIT 50").fetchall()
    conn.close()
    return jsonify([{
        "id":            r["id"],
        "name":          r["name"] or "Anonymous",
        "rating":        r["rating"],
        "category":      r["category"],
        "feedback_text": r["feedback_text"],
        "created_at":    r["created_at"]
    } for r in rows])

# ════════════════════════════════════════
#  STAFF — DASHBOARD
# ════════════════════════════════════════
@app.route("/")
@app.route("/dashboard")
@staff_required
def dashboard():
    conn = get_db()
    total    = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    active   = conn.execute("SELECT COUNT(*) FROM customers WHERE status='Active'").fetchone()[0]
    churned  = conn.execute("SELECT COUNT(*) FROM customers WHERE status='Churned'").fetchone()[0]
    recent   = conn.execute("SELECT * FROM customers ORDER BY created_at DESC LIMIT 5").fetchall()
    conn.close()
    retention = round((active/total*100),1) if total > 0 else 0
    return render_template("dashboard.html",
        total_customers=total, active_customers=active,
        churn_customers=churned, retention_rate=retention,
        recent_customers=recent
    )

# ════════════════════════════════════════
#  STAFF — CUSTOMERS
# ════════════════════════════════════════
@app.route("/customers")
@staff_required
def customers():
    conn = get_db()
    all_c   = conn.execute("SELECT * FROM customers ORDER BY created_at DESC").fetchall()
    total   = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    active  = conn.execute("SELECT COUNT(*) FROM customers WHERE status='Active'").fetchone()[0]
    churned = conn.execute("SELECT COUNT(*) FROM customers WHERE status='Churned'").fetchone()[0]
    conn.close()
    return render_template("customer/customer.html",
        customers=all_c, total=total, active=active, churned=churned, at_risk=0
    )

@app.route("/customers/add", methods=["GET","POST"])
@staff_required
def add_customer():
    if request.method == "POST":
        conn    = get_db()
        tenure  = float(request.form.get("tenure") or 0)
        monthly = float(request.form.get("monthly_charges") or 0)
        conn.execute("""INSERT INTO customers
            (full_name,email,phone,age,plan_type,tenure,
             monthly_charges,total_charges,contract_type,payment_method,support_calls)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""", (
            request.form.get("full_name"), request.form.get("email"),
            request.form.get("phone"), request.form.get("age") or None,
            request.form.get("plan_type"), tenure, monthly, round(tenure*monthly,2),
            request.form.get("contract_type"), request.form.get("payment_method"),
            request.form.get("support_calls") or 0
        ))
        conn.commit()
        conn.close()
        flash("Customer added successfully!", "success")
        return redirect(url_for("customers"))
    return render_template("customer/add_customer.html")

@app.route("/customers/update/<int:cid>", methods=["GET","POST"])
@staff_required
def update_customer(cid):
    conn = get_db()
    if request.method == "POST":
        conn.execute("""UPDATE customers SET
            full_name=?,email=?,phone=?,age=?,plan_type=?,
            tenure=?,monthly_charges=?,contract_type=?,payment_method=?,support_calls=?
            WHERE id=?""", (
            request.form.get("full_name"), request.form.get("email"),
            request.form.get("phone"), request.form.get("age") or None,
            request.form.get("plan_type"), request.form.get("tenure") or 0,
            request.form.get("monthly_charges") or 0,
            request.form.get("contract_type"), request.form.get("payment_method"),
            request.form.get("support_calls") or 0, cid
        ))
        conn.commit()
        conn.close()
        flash("Customer updated!", "success")
        return redirect(url_for("customers"))
    cust = conn.execute("SELECT * FROM customers WHERE id=?", (cid,)).fetchone()
    conn.close()
    return render_template("customer/update_customer.html", customer=cust)

@app.route("/customers/delete/<int:cid>")
@staff_required
def delete_customer(cid):
    conn = get_db()
    conn.execute("DELETE FROM customers WHERE id=?", (cid,))
    conn.commit()
    conn.close()
    flash("Customer deleted.", "success")
    return redirect(url_for("customers"))

# ════════════════════════════════════════
#  STAFF — PREDICTION
# ════════════════════════════════════════
@app.route("/predict", methods=["GET"])
@staff_required
def prediction():
    conn = get_db()
    all_c = conn.execute("SELECT id,full_name FROM customers ORDER BY full_name").fetchall()
    conn.close()
    return render_template("churn/prediction.html", all_customers=all_c)

@app.route("/predict", methods=["POST"])
@staff_required
def predict():
    tenure  = float(request.form.get("tenure") or 0)
    monthly = float(request.form.get("monthly_charges") or 0)
    total   = float(request.form.get("total_charges") or tenure*monthly)
    cid     = request.form.get("customer_id")
    cname   = request.form.get("customer_name","Unknown")

    prob = round(float(model.predict_proba(np.array([[tenure,monthly,total]]))[0][1])*100,2)
    risk = "High Risk" if prob>70 else ("Medium Risk" if prob>40 else "Low Risk")

    conn = get_db()
    conn.execute("""INSERT INTO predictions
        (customer_id,customer_name,tenure,monthly_charges,total_charges,probability,risk_level)
        VALUES (?,?,?,?,?,?,?)""",
        (cid,cname,tenure,monthly,total,prob,risk))
    conn.commit()
    conn.close()

    return render_template("churn/prediction_result.html",
        churn_probability=prob, risk_level=risk,
        customer_name=cname, tenure=tenure,
        monthly_charges=monthly, total_charges=total
    )
# ════════════════════════════════════════
#  BULK PREDICTION UPLOAD
# ════════════════════════════════════════
@app.route("/predict/bulk", methods=["GET","POST"])
@staff_required
def bulk_predict():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file uploaded.", "error")
            return redirect(url_for("bulk_predict"))

        file = request.files["file"]
        if file.filename == "":
            flash("No file selected.", "error")
            return redirect(url_for("bulk_predict"))

        if not file.filename.endswith(".csv"):
            flash("Only CSV files are supported.", "error")
            return redirect(url_for("bulk_predict"))

        import csv, io
        stream    = io.StringIO(file.stream.read().decode("utf-8"), newline=None)
        reader    = csv.DictReader(stream)
        results   = []
        errors    = []
        conn      = get_db()

        for i, row in enumerate(reader, start=1):
            try:
                # Try to get required columns — support multiple naming formats
                name    = row.get("full_name") or row.get("name") or row.get("customerID") or f"Customer {i}"
                tenure  = float(row.get("tenure") or 0)
                monthly = float(row.get("monthly_charges") or row.get("MonthlyCharges") or 0)
                total   = float(row.get("total_charges") or row.get("TotalCharges") or tenure * monthly or 0)

                prob = round(float(model.predict_proba(np.array([[tenure, monthly, total]]))[0][1]) * 100, 2)
                risk = "High Risk" if prob > 70 else ("Medium Risk" if prob > 40 else "Low Risk")

                # Save to predictions table
                conn.execute("""INSERT INTO predictions
                    (customer_name, tenure, monthly_charges, total_charges, probability, risk_level)
                    VALUES (?,?,?,?,?,?)""",
                    (name, tenure, monthly, total, prob, risk))

                results.append({
                    "name":    name,
                    "tenure":  tenure,
                    "monthly": monthly,
                    "total":   total,
                    "prob":    prob,
                    "risk":    risk
                })
            except Exception as e:
                errors.append(f"Row {i}: {str(e)}")

        conn.commit()
        conn.close()

        return render_template("churn/bulk_result.html",
            results=results,
            errors=errors,
            total=len(results),
            high_risk=sum(1 for r in results if r["risk"] == "High Risk"),
            medium_risk=sum(1 for r in results if r["risk"] == "Medium Risk"),
            low_risk=sum(1 for r in results if r["risk"] == "Low Risk")
        )

    return render_template("churn/bulk_predict.html")
@app.route("/predict/bulk/sample")
@staff_required
def bulk_sample():
    output = "full_name,tenure,monthly_charges,total_charges\nRahul Sharma,12,1299,15588\nPriya Verma,6,899,5394\nAmit Patel,24,1599,38376\nNeha Singh,3,699,2097\nRohit Gupta,36,1999,71964\n"
    return Response(output, mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=sample_bulk_upload.csv"})
# ════════════════════════════════════════
#  STAFF — OTHER PAGES
# ════════════════════════════════════════
@app.route("/segmentation")
@staff_required
def segmentation():
    return render_template("segmentation/segmentation.html")

@app.route("/retention")
@staff_required
def retention():
    return render_template("segmentation/retention.html")

@app.route("/sms-alert", methods=["GET","POST"])
@staff_required
def sms_alert():
    if request.method == "POST":
        flash("SMS sent successfully!", "success")
    return render_template("segmentation/sms_alert.html")

@app.route("/campaigns", methods=["GET","POST"])
@staff_required
def campaigns():
    if request.method == "POST":
        conn = get_db()
        conn.execute("""INSERT INTO campaigns
            (campaign_name,target_segment,campaign_type,message,start_date,end_date)
            VALUES (?,?,?,?,?,?)""", (
            request.form.get("campaign_name"), request.form.get("target_segment"),
            request.form.get("campaign_type"), request.form.get("message"),
            request.form.get("start_date"), request.form.get("end_date")
        ))
        conn.commit()
        conn.close()
        flash("Campaign created!", "success")
        return redirect(url_for("campaign_results"))
    conn = get_db()
    recent = conn.execute("SELECT * FROM campaigns ORDER BY created_at DESC LIMIT 5").fetchall()
    conn.close()
    return render_template("campaign/campaign.html", campaigns=recent)

@app.route("/campaigns/results")
@staff_required
def campaign_results():
    conn = get_db()
    all_c = conn.execute("SELECT * FROM campaigns ORDER BY created_at DESC").fetchall()
    conn.close()
    return render_template("campaign/campaign_results.html", campaign_list=all_c)

@app.route("/analytics")
@staff_required
def analytics():
    conn = get_db()
    total       = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    churned     = conn.execute("SELECT COUNT(*) FROM customers WHERE status='Churned'").fetchone()[0]
    churn_rate  = round(churned/total*100,1) if total > 0 else 0
    avg_tenure  = conn.execute("SELECT AVG(tenure) FROM customers").fetchone()[0] or 0
    avg_charges = conn.execute("SELECT AVG(monthly_charges) FROM customers").fetchone()[0] or 0
    conn.close()
    return render_template("analytics/analytics.html",
        churn_rate=churn_rate,
        avg_tenure=round(avg_tenure,1),
        monthly_charges=round(avg_charges,0)
    )

@app.route("/reports", methods=["GET","POST"])
@staff_required
def reports():
    if request.method == "POST":
        flash("Report generated!", "success")
    return render_template("analytics/reports.html")

# ════════════════════════════════════════
#  REPORT DOWNLOADS
# ════════════════════════════════════════
@app.route("/reports/churn")
@staff_required
def report_churn():
    conn = get_db()
    rows = conn.execute("SELECT * FROM predictions ORDER BY created_at DESC").fetchall()
    conn.close()
    output = "ID,Customer,Tenure,Monthly Charges,Probability,Risk Level,Date\n"
    for r in rows:
        output += f"{r['id']},{r['customer_name']},{r['tenure']},{r['monthly_charges']},{r['probability']},{r['risk_level']},{r['created_at']}\n"
    return Response(output, mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=churn_report.csv"})

@app.route("/reports/customers")
@staff_required
def report_customers():
    conn = get_db()
    rows = conn.execute("SELECT * FROM customers ORDER BY created_at DESC").fetchall()
    conn.close()
    output = "ID,Name,Email,Phone,Plan,Tenure,Monthly Charges,Status,Date\n"
    for r in rows:
        output += f"{r['id']},{r['full_name']},{r['email']},{r['phone']},{r['plan_type']},{r['tenure']},{r['monthly_charges']},{r['status']},{r['created_at']}\n"
    return Response(output, mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=customers_report.csv"})

@app.route("/reports/revenue")
@staff_required
def report_revenue():
    conn = get_db()
    rows = conn.execute("SELECT full_name,plan_type,monthly_charges,total_charges,tenure FROM customers ORDER BY total_charges DESC").fetchall()
    conn.close()
    output = "Name,Plan,Monthly Charges,Total Charges,Tenure\n"
    for r in rows:
        output += f"{r['full_name']},{r['plan_type']},{r['monthly_charges']},{r['total_charges']},{r['tenure']}\n"
    return Response(output, mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=revenue_report.csv"})

@app.route("/reports/campaigns")
@staff_required
def report_campaigns():
    conn = get_db()
    rows = conn.execute("SELECT * FROM campaigns ORDER BY created_at DESC").fetchall()
    conn.close()
    output = "ID,Campaign Name,Segment,Type,Status,Sent,Responses,Start Date,End Date\n"
    for r in rows:
        output += f"{r['id']},{r['campaign_name']},{r['target_segment']},{r['campaign_type']},{r['status']},{r['sent']},{r['responses']},{r['start_date']},{r['end_date']}\n"
    return Response(output, mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=campaigns_report.csv"})

@app.route("/reports/analytics")
@staff_required
def report_analytics():
    conn = get_db()
    rows = conn.execute("SELECT * FROM predictions ORDER BY created_at DESC").fetchall()
    conn.close()
    output = "ID,Customer,Probability,Risk Level,Date\n"
    for r in rows:
        output += f"{r['id']},{r['customer_name']},{r['probability']},{r['risk_level']},{r['created_at']}\n"
    return Response(output, mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=analytics_report.csv"})

@app.route("/reports/retention")
@staff_required
def report_retention():
    conn = get_db()
    rows = conn.execute("SELECT * FROM customers WHERE status='Active' ORDER BY tenure DESC").fetchall()
    conn.close()
    output = "ID,Name,Email,Plan,Tenure,Monthly Charges\n"
    for r in rows:
        output += f"{r['id']},{r['full_name']},{r['email']},{r['plan_type']},{r['tenure']},{r['monthly_charges']}\n"
    return Response(output, mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=retention_report.csv"})

@app.route("/reports/custom", methods=["POST"])
@staff_required
def report_custom():
    report_type = request.form.get("report_type","all")
    conn = get_db()
    if report_type == "churn":
        rows = conn.execute("SELECT * FROM predictions ORDER BY created_at DESC").fetchall()
        output = "ID,Customer,Probability,Risk Level,Date\n"
        for r in rows:
            output += f"{r['id']},{r['customer_name']},{r['probability']},{r['risk_level']},{r['created_at']}\n"
    elif report_type == "customers":
        rows = conn.execute("SELECT * FROM customers ORDER BY created_at DESC").fetchall()
        output = "ID,Name,Email,Plan,Status\n"
        for r in rows:
            output += f"{r['id']},{r['full_name']},{r['email']},{r['plan_type']},{r['status']}\n"
    elif report_type == "revenue":
        rows = conn.execute("SELECT * FROM customers ORDER BY total_charges DESC").fetchall()
        output = "Name,Monthly Charges,Total Charges\n"
        for r in rows:
            output += f"{r['full_name']},{r['monthly_charges']},{r['total_charges']}\n"
    else:
        rows = conn.execute("SELECT * FROM customers ORDER BY created_at DESC").fetchall()
        output = "ID,Name,Email,Plan,Status,Tenure,Monthly Charges\n"
        for r in rows:
            output += f"{r['id']},{r['full_name']},{r['email']},{r['plan_type']},{r['status']},{r['tenure']},{r['monthly_charges']}\n"
    conn.close()
    return Response(output, mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={report_type}_report.csv"})

# ════════════════════════════════════════
#  STAFF — FEEDBACK VIEW
# ════════════════════════════════════════
@app.route("/feedback")
@staff_required
def feedback():
    conn = get_db()
    all_fb   = conn.execute("SELECT * FROM feedback ORDER BY created_at DESC").fetchall()
    avg_rat  = conn.execute("SELECT AVG(rating) FROM feedback").fetchone()[0] or 0
    total_fb = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    conn.close()
    return render_template("feedback/admin_feedback.html",
        feedbacks=all_fb,
        avg_rating=round(avg_rat,1),
        total_feedback=total_fb
    )

# ════════════════════════════════════════
#  PROFILE & SETTINGS
# ════════════════════════════════════════
@app.route("/profile", methods=["GET","POST"])
@login_required
def profile():
    if request.method == "POST":
        fname = request.form.get("full_name")
        email = request.form.get("email")
        phone = request.form.get("phone")
        role  = request.form.get("role")
        conn  = get_db()
        conn.execute(
            "UPDATE users SET full_name=?,email=?,phone=?,role=? WHERE id=?",
            (fname, email, phone, role, session["user_id"])
        )
        conn.commit()
        conn.close()
        session["user_name"] = fname
        flash("Profile updated!", "success")
        return redirect(url_for("profile"))
    return render_template("user/profile.html")

@app.route("/settings", methods=["GET","POST"])
@login_required
def settings():
    if request.method == "POST":
        flash("Settings saved!", "success")
    return render_template("user/settings.html")

# ════════════════════════════════════════
#  ADMIN — MANAGE STAFF ACCOUNTS
# ════════════════════════════════════════
@app.route("/admin/users")
@staff_required
@admin_required
def manage_users():
    conn = get_db()
    users = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    conn.close()
    return render_template("admin/users.html", users=users)

@app.route("/admin/create-staff", methods=["POST"])
@staff_required
@admin_required
def create_staff():
    full_name = request.form.get("full_name","").strip()
    email     = request.form.get("email","").strip().lower()
    password  = request.form.get("password","")
    role      = request.form.get("role","Analyst")

    if role not in ["Admin","Manager","Analyst"]:
        flash("Invalid role.", "error")
        return redirect(url_for("manage_users"))

    conn = get_db()
    exists = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
    if exists:
        flash("Email already exists.", "error")
    else:
        conn.execute(
            "INSERT INTO users (full_name,email,password_hash,role) VALUES (?,?,?,?)",
            (full_name, email, generate_password_hash(password), role)
        )
        conn.commit()
        flash(f"{role} account created for {full_name}!", "success")
    conn.close()
    return redirect(url_for("manage_users"))

@app.route("/admin/delete-user/<int:uid>")
@staff_required
@admin_required
def delete_user(uid):
    if uid == session["user_id"]:
        flash("You cannot delete your own account.", "error")
        return redirect(url_for("manage_users"))
    conn = get_db()
    conn.execute("DELETE FROM users WHERE id=?", (uid,))
    conn.commit()
    conn.close()
    flash("User deleted.", "success")
    return redirect(url_for("manage_users"))

# ════════════════════════════════════════
#  INIT DB
# ════════════════════════════════════════
with app.app_context():
    init_db()

if __name__ == "__main__":
    app.run(debug=True)