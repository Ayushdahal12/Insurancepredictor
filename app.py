import gradio as gr
import joblib
import numpy as np
import os
import json
import re


USERS_FILE = "users.json"

def load_users():
    default_users = {
        "ayushdahal@gmail.com": "password123",
        "teacher@gmail.com": "teacher123",
        "student@gmail.com": "student123",
    }
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            saved = json.load(f)
            default_users.update(saved)
            return default_users
    return default_users

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

# ── LOAD MODel 
MODEL_LOADED = False
model = None
if os.path.exists("model.pkl"):
    try:
        model = joblib.load("model.pkl")
        MODEL_LOADED = True
        print("✅ model.pkl loaded!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print("⚠️  model.pkl not found. Run: python3 train_model.py")

USD_TO_NPR = 133.0

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');
* { font-family: 'Nunito', sans-serif !important; }
body, .gradio-container {
    background: linear-gradient(135deg, #e0f2fe 0%, #f0fdf4 50%, #fef9c3 100%) !important;
    min-height: 100vh !important;
}
footer { display: none !important; }
.card { 
    background:white; 
    border-radius:20px; 
    padding:32px 36px; 
    box-shadow:0 8px 40px rgba(0,0,0,0.10); 
    max-width:500px; 
    margin:auto !important;
    margin-top: 40px !important;
}
.main-card { 
    background:white; 
    border-radius:20px; 
    padding:32px 36px; 
    box-shadow:0 8px 40px rgba(0,0,0,0.10); 
    max-width:640px; 
    margin:auto !important;
}
#big-title h1 { 
    text-align:center; 
    font-size:3.5rem !important; 
    font-weight:900 !important; 
    background:linear-gradient(135deg,#1d4ed8,#0ea5e9,#6366f1) !important; 
    -webkit-background-clip:text !important; 
    -webkit-text-fill-color:transparent !important; 
    background-clip:text !important; 
    margin-top: 60px !important;
    margin-bottom:4px !important; 
    letter-spacing:-1px !important; 
    line-height:1.1 !important; 
}
#subtitle-txt h3 { 
    text-align:center; 
    color:#64748b !important; 
    font-size:1.1rem !important; 
    margin-bottom:20px !important; 
    font-weight:600 !important; 
}
"""

# functions  

def do_login(email, password):
    email = email.strip().lower()
    if not email or not password:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "⚠️ Please enter email and password.",
            ""
        )
    if not is_valid_email(email):
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "❌ Enter a valid email like name@gmail.com",
            ""
        )
    users = load_users()
    if email in users and users[email] == password:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            "",
            email
        )
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        "❌ Wrong email or password.",
        ""
    )

def do_register(email, password):
    email = email.strip().lower()
    if not email or not password:
        return "⚠️ Please enter email and password."
    if not is_valid_email(email):
        return "❌ Enter a valid email like name@gmail.com"
    if len(password) < 6:
        return "❌ Password must be at least 6 characters."
    users = load_users()
    if email in users:
        return "❌ Email already registered. Please login."
    users[email] = password
    save_users(users)
    return "✅ Registered! Now click Login tab and sign in."

def get_welcome(username):
    if username:
        return f"👋 Welcome, **{username}**!"
    return ""

def do_logout():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        "", "", "", "", ""
    )

def do_predict(age, gender, bmi, children, smoker, region, logged_username):
    if not logged_username:
        return "⚠️ Please login first."
    if not MODEL_LOADED or model is None:
        return "⚠️ model.pkl not found! Run: python3 train_model.py"
    try:
        age      = float(age)
        bmi      = float(bmi)
        children = int(float(children))
    except:
        return "⚠️ Please enter valid numbers for Age, BMI and Children."
    gender_enc = 0 if gender == "Male" else 1
    smoker_enc = 1 if smoker == "Yes"  else 0
    region_enc = {"Southeast": 0, "Southwest": 1, "Northeast": 2, "Northwest": 3}[region]
    features   = np.array([[age, gender_enc, bmi, children, smoker_enc, region_enc]])
    cost_usd   = model.predict(features)[0]
    cost_npr   = cost_usd * USD_TO_NPR
    return (
        f"Estimated Insurance Cost\n\n"
        f"🇺🇸  **${cost_usd:,.2f} USD**\n\n"
        f"🇳🇵  **रू {cost_npr:,.2f} NPR**\n\n"
        f"Age: {age:.0f} | {gender} | BMI: {bmi} | Children: {children} | Smoker: {smoker} | {region}"
    )

def calc_bmi(weight_kg, height_ft):
    try:
        w = float(weight_kg)
        h = float(height_ft)
        if w <= 0 or h <= 0:
            return "⚠️ Enter valid values."
        height_m = h * 0.3048
        bmi = w / (height_m ** 2)
        if   bmi < 18.5: cat = "🔵 Underweight"
        elif bmi < 25.0: cat = "🟢 Normal"
        elif bmi < 30.0: cat = "🟡 Overweight"
        else:            cat = "🔴 Obese"
        return f"Your BMI: **{bmi:.1f}** — {cat}\n\n_(Copy this into the BMI field on Predictor tab!)_"
    except:
        return "⚠️ Invalid input. Enter numbers only."

def convert_weight(value, direction):
    try:
        v = float(value)
        if v <= 0:
            return "⚠️ Enter a valid weight."
        if direction == "kg → lbs":
            return f"**{v:.2f} kg  =  {v * 2.20462:.2f} lbs**"
        else:
            return f"**{v:.2f} lbs  =  {v / 2.20462:.2f} kg**"
    except:
        return "⚠️ Invalid input. Enter a number."

def go_to_bmi():       return gr.update(selected="bmi_tab")
def go_to_weight():    return gr.update(selected="weight_tab")
def go_to_predictor(): return gr.update(selected="predictor_tab")

# ── UI ─────────────────────────────────────────────────────

with gr.Blocks(css=CSS, title="Insurance Cost Predictor") as demo:

    logged_username = gr.State("")

    gr.Markdown("# Medical Insurance Cost Predictor", elem_id="big-title")
    gr.Markdown("### Predict your insurance cost based on your details", elem_id="subtitle-txt")

    # ── LOGIN PAGE ─────────────────────────────────────────
    with gr.Column(visible=True, elem_classes="card") as login_page:
        gr.Markdown("### Login ")
        login_email = gr.Textbox(label="Email", placeholder="yourname@gmail.com")
        login_pwd   = gr.Textbox(label="Password", placeholder="••••••••", type="password")
        login_btn   = gr.Button("Login", variant="primary")
        login_msg   = gr.Markdown("", elem_id="msg-md")
        with gr.Column(elem_id="switch-btn"):
            go_register_btn = gr.Button("Don't have an account? Register here →")

    # ── REGISTER PAGE ──────────────────────────────────────
    with gr.Column(visible=False, elem_classes="card") as register_page:
        gr.Markdown("### 📝 Create Account")
        reg_email = gr.Textbox(label="Email", placeholder="yourname@gmail.com")
        reg_pwd   = gr.Textbox(label="Choose Password (min 6 characters)", placeholder="••••••••", type="password")
        with gr.Column(elem_id="reg-btn"):
            reg_btn = gr.Button("Register")
        reg_msg = gr.Markdown("", elem_id="msg-md")
        with gr.Column(elem_id="switch-btn"):
            go_login_btn = gr.Button("Already have an account? Login here →")


    with gr.Column(visible=False) as main_panel:

        with gr.Row():
            welcome_md = gr.Markdown("", elem_id="welcome-md")
            logout_btn = gr.Button("Logout", variant="secondary", scale=0, min_width=130)

        with gr.Tabs() as tabs:

            # TAB 1 — Insurance Predictor
            with gr.Tab("🏥 Insurance Predictor", id="predictor_tab"):
                with gr.Column(elem_classes="main-card"):
                    with gr.Row():
                        with gr.Column(elem_id="bmi-open-btn"):
                            bmi_open_btn = gr.Button("📊 BMI Calculator")
                        with gr.Column(elem_id="weight-open-btn"):
                            weight_open_btn = gr.Button("kg ↔ lbs Converter")

                    gr.Markdown("### 📋 Enter Your Details")
                    with gr.Row():
                        age = gr.Textbox(label="Age (years)", value="30", placeholder="e.g. 25")
                        bmi = gr.Textbox(label="BMI", value="25.0", placeholder="e.g. 22.5")
                    with gr.Row():
                        gender = gr.Radio(["Male", "Female"], value="Male", label="👤 Gender")
                        smoker = gr.Radio(["No", "Yes"],      value="No",   label="🚬 Smoker?")
                    with gr.Row():
                        children = gr.Textbox(label="Children", value="0", placeholder="e.g. 2")
                        region   = gr.Dropdown(
                            choices=["Southeast", "Southwest", "Northeast", "Northwest"],
                            value="Southeast", label="📍 Region"
                        )
                    predict_btn = gr.Button("Predict Insurance Cost", variant="primary")
                    result_out  = gr.Markdown(
                        "Result will appear here after clicking Predict...",
                        elem_id="result-box"
                    )

            # TAB 2 — BMI Calculator
            with gr.Tab("📊 BMI Calculator", id="bmi_tab"):
                with gr.Column(elem_classes="main-card"):
                    gr.Markdown("## 📊 BMI Calculator")
                    gr.Markdown("Enter your weight and height to calculate your Body Mass Index.")
                    with gr.Row():
                        bmi_weight = gr.Textbox(label="Weight (kg)", value="70", placeholder="e.g. 65")
                        bmi_height = gr.Textbox(label="📏 Height (ft)", value="5.7", placeholder="e.g. 5.7 or 5.10")
                    bmi_calc_btn = gr.Button("Calculate My BMI", variant="primary")
                    bmi_result   = gr.Markdown("Your BMI will appear here...", elem_id="bmi-result")
                    gr.Markdown("Below 18.5 = Underweight &nbsp;|&nbsp; 18.5–24.9 = Normal &nbsp;|&nbsp; 🟡 25–29.9 = Overweight &nbsp;|&nbsp; 🔴 30+ = Obese")
                    back_from_bmi = gr.Button("← Back to Insurance Predictor", variant="secondary")

            # TAB 3 — Weight Converter
            with gr.Tab("⚖️ kg ↔ lbs", id="weight_tab"):
                with gr.Column(elem_classes="main-card"):
                    gr.Markdown("## ⚖️ Weight Converter")
                    gr.Markdown("Convert weight between kilograms and pounds.")
                    w_value  = gr.Textbox(label="Enter Weight", value="70", placeholder="e.g. 70")
                    w_dir    = gr.Radio(["kg → lbs", "lbs → kg"], value="kg → lbs", label="Convert Direction")
                    w_btn    = gr.Button("Convert Weight", variant="primary")
                    w_result = gr.Markdown("Result will appear here...", elem_id="weight-result")
                    gr.Markdown("**Reference:** 1 kg = 2.20462 lbs &nbsp;|&nbsp; 1 lbs = 0.45359 kg")
                    back_from_weight = gr.Button("← Back to Insurance Predictor", variant="secondary")

    # ── EVENTS ─────────────────────────────────────────────

    go_register_btn.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[login_page, register_page]
    )
    go_login_btn.click(
        fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[login_page, register_page]
    )

    login_btn.click(
        fn=do_login,
        inputs=[login_email, login_pwd],
        outputs=[login_page, register_page, main_panel, login_msg, logged_username],
    ).then(
        fn=get_welcome,
        inputs=[logged_username],
        outputs=[welcome_md]
    )

    reg_btn.click(
        fn=do_register,
        inputs=[reg_email, reg_pwd],
        outputs=[reg_msg]
    )

    logout_btn.click(
        fn=do_logout,
        inputs=[],
        outputs=[login_page, register_page, main_panel, login_msg, logged_username, login_email, login_pwd, result_out],
    )

    predict_btn.click(
        fn=do_predict,
        inputs=[age, gender, bmi, children, smoker, region, logged_username],
        outputs=[result_out],
    )

    bmi_open_btn.click(fn=go_to_bmi,           outputs=[tabs])
    weight_open_btn.click(fn=go_to_weight,     outputs=[tabs])
    back_from_bmi.click(fn=go_to_predictor,    outputs=[tabs])
    back_from_weight.click(fn=go_to_predictor, outputs=[tabs])

    bmi_calc_btn.click(fn=calc_bmi,    inputs=[bmi_weight, bmi_height], outputs=[bmi_result])
    w_btn.click(fn=convert_weight,     inputs=[w_value, w_dir],         outputs=[w_result])

if __name__ == "__main__":
    demo.launch()