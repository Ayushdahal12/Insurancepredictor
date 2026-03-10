# 🏥 Medical Insurance Cost Prediction
--

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

## 🧠 How the ML Model Works

- **Dataset**: Kaggle Medical Insurance (1338 rows)
- **Features**: Age, Sex, BMI, Children, Smoker, Region
- **Target**: `charges` (insurance cost in $)
- **Model**: Linear Regression
- **Encoding**: sex (male=0, female=1), smoker (no=0, yes=1), region (SE=0, SW=1, NE=2, NW=3)
