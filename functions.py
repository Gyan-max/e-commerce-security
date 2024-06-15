import numpy as np
import pandas as pd
import pickle
from flask import render_template


def load_model():
    with open("model.pickle", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pickle", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict_spam(email_content, model, vectorizer):
    data = pd.DataFrame({"Message": [email_content]})
    email_features = vectorizer.transform(data["Message"])
    prediction = model.predict(email_features)[0]
    return "Not Spam" if prediction == 1 else "Spam"


def load_credit_model():
    with open("credit_model.pickle", "rb") as f:
        credit_model = pickle.load(f)
    return credit_model


def predict_credit(card_content, credit_model):
    card_features = card_content.split(",")
    if len(card_features) != 30:
        return render_template("creditcard.html", credit_transaction_prediction=None,
                               error="Invalid number of features. Please provide 30 comma-separated values.")
    card_features = np.array(card_features, dtype=np.float64).reshape(1, -1)

    credit_prediction = credit_model.predict(card_features)[0]
    return "Fraudulent transaction" if credit_prediction == 1 else "Legitimate transaction"


def load_password_model():
    with open("final_model.pickle", "rb") as f:
        password_model = pickle.load(f)
    with open("tfidf_password_strength.pickle", "rb") as f:
        password_vectorizer = pickle.load(f)
    return password_model, password_vectorizer


def word_divide_char(password_content):
    return list(password_content)


def check_password_strength(password_content, password_model, password_vectorizer):
    password_features = password_vectorizer.transform([password_content])  # Pass string directly
    password_prediction = password_model.predict(password_features)[0]
    return password_prediction
