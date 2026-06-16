
user_symptoms = pd.DataFrame(
    [[0] * len(X.columns)],
    columns=X.columns
)

user_symptoms["itching"] = 1
user_symptoms["skin_rash"] = 1
user_symptoms["nodal_skin_eruptions"] = 1

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(user_symptoms)

print("SHAP generated successfully")
