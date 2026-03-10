# 🏥 Medical Insurance Cost Prediction
**BIT 7th Semester AI/ML Project**

---

## 📁 Project Folder Structure

```
project/
├── app.py            ← Main Gradio app (run this)
├── train_model.py    ← Train the ML model (run once)
├── model.pkl         ← Auto-created after training
├── insurance.csv     ← Download from Kaggle (see below)
└── requirements.txt  ← Python packages
```

---

## ⚙️ Setup Steps (Do this on your laptop)

### Step 1 — Install Python packages
Open terminal inside the `project/` folder and run:
```bash
pip install -r requirements.txt
```

### Step 2 — Download the dataset
1. Go to: https://www.kaggle.com/datasets/mirichoi0218/insurance
2. Download `insurance.csv`
3. Place it inside the `project/` folder

### Step 3 — Train the model
```bash
python train_model.py
```
This creates `model.pkl` in your folder.

### Step 4 — Run the app locally
```bash
python app.py
```
Open browser at: http://localhost:7860

---

## 🔐 Login Credentials (Demo)

| Username   | Password     |
|------------|--------------|
| ayush      | password123  |
| testuser   | 1234         |
| student    | bit2024      |

---

## 🚀 Deploy on Hugging Face

1. Create account at https://huggingface.co
2. Click **New Space** → Choose **Gradio** SDK
3. Upload these files:
   - `app.py`
   - `model.pkl`
   - `requirements.txt`
4. Wait ~2 minutes for build
5. Share the public link: `https://your-space-name.hf.space`

---

## 🧠 How the ML Model Works

- **Dataset**: Kaggle Medical Insurance (1338 rows)
- **Features**: Age, Sex, BMI, Children, Smoker, Region
- **Target**: `charges` (insurance cost in $)
- **Model**: Linear Regression
- **Encoding**: sex (male=0, female=1), smoker (no=0, yes=1), region (SE=0, SW=1, NE=2, NW=3)
