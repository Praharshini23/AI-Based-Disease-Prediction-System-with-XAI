import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv()

user_symptoms = pd.DataFrame(
    [[0] * len(X.columns)],
    columns=X.columns
)

user_symptoms["itching"] = 1
user_symptoms["skin_rash"] = 1
user_symptoms["nodal_skin_eruptions"] = 1

prediction = model.predict(user_symptoms)[0]

print("Predicted Disease:", prediction)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(user_symptoms)

print("\n===== SHAP INFO =====")
print("Type:", type(shap_values))

try:
    print("Length:", len(shap_values))
except Exception as e:
    print("Length Error:", e)

try:
    print("Shape:", shap_values.shape)
except Exception as e:
    print("Shape Error:", e)

print("\nSHAP Generated Successfully")
