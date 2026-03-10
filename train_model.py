
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ------- 1. Load Dataset -------
print("Loading dataset...")
df = pd.read_csv("insurance.csv")
print(df.head())

# ------- 2. Encode Categorical Columns -------
df["gender"]    = df["gender"].map({"male": 0, "female": 1})
df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
df["region"] = df["region"].map({
    "southeast": 0,
    "southwest": 1,
    "northeast": 2,
    "northwest": 3
})

# ------- 3. Features & Target -------
X = df[["age", "gender", "bmi", "children", "smoker", "region"]]
y = df["charges"]

# ------- 4. Split -------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------- 5. Train Model -------
print("Training model...")
model = LinearRegression()
model.fit(X_train, y_train)

# ------- 6. Evaluate -------
preds = model.predict(X_test)
print(f"MAE  : ${mean_absolute_error(y_test, preds):,.2f}")
print(f"R²   : {r2_score(y_test, preds):.4f}")

# ------- 7. Save Model -------
joblib.dump(model, "model.pkl")
print("✅ model.pkl saved successfully!")
