
user_symptoms = pd.DataFrame(
    [[0] * len(X_train.columns)],
    columns=X_train.columns
)

user_symptoms["itching"] = 1
user_symptoms["skin_rash"] = 1
user_symptoms["nodal_skin_eruptions"] = 1

prediction = model.predict(user_symptoms)

print("\nPredicted Disease:", prediction[0])


user_symptoms = pd.DataFrame(
    [[0] * len(X_train.columns)],
    columns=X_train.columns
)
