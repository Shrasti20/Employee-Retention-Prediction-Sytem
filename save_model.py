import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
import joblib

# dummy data
data = pd.DataFrame({
    "Department": ["HR", "IT", "Sales", "HR"],
    "Salary": [30000, 50000, 40000, 35000],
    "Attrition": [0, 1, 0, 1]
})

# encoder
encoder = OrdinalEncoder()
data["Department"] = encoder.fit_transform(data[["Department"]])

# features & target
X = data[["Department", "Salary"]]
y = data["Attrition"]

# model
model = XGBClassifier()
model.fit(X, y)

# save files
joblib.dump(model, "final_xgboost_model.pkl")
joblib.dump(encoder, "ordinal_encoder.pkl")

print("Model + Encoder saved ✅")