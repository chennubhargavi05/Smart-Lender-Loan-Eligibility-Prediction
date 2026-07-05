import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Load Dataset
df = pd.read_csv("dataset/loan_prediction_cleaned.csv")

# Encode categorical columns
label_encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = label_encoder.fit_transform(df[col])

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

models = {
    "decision_tree.pkl": DecisionTreeClassifier(random_state=42),
    "random_forest.pkl": RandomForestClassifier(random_state=42),
    "knn_model.pkl": KNeighborsClassifier(),
    "xgboost_model.pkl": XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )
}

for filename, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, "models/" + filename)

print("All models trained and saved successfully!")