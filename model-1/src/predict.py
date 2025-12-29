import joblib
from scipy.sparse import hstack

# Load trained model and vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_category(description, amount):
    text_vec = vectorizer.transform([description])
    combined = hstack((text_vec, [[amount]]))
    prediction = model.predict(combined)
    return prediction[0]

if __name__ == "__main__":
    text = input("Enter expense description: ")
    amount = float(input("Enter amount: "))
    category = predict_category(text, amount)
    print("Predicted Category:", category)
