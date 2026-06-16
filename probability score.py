
user_symptoms["itching"] = 1
user_symptoms["skin_rash"] = 1
user_symptoms["nodal_skin_eruptions"] = 1

prediction = model.predict(user_symptoms)

probabilities = model.predict_proba(user_symptoms)[0]

disease_probs = list(zip(model.classes_, probabilities))

disease_probs.sort(key=lambda x: x[1], reverse=True)

print("\nPredicted Disease:", prediction[0])

print("\nTop 3 Predictions:")

for disease, prob in disease_probs[:3]:
    print(f"{disease}: {prob*100:.2f}%")
