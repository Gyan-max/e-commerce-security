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




# Load the pre-trained vectorizer from a pickle file
def load_vectorizer():
    with open("tfidf_password_strength.pickle", 'rb') as file:
        saved_vectorizer = pickle.load(file)
    return saved_vectorizer

# Load the pre-trained model from a pickle file
def load_passmodel():
    with open("final_model.pickle", 'rb') as file:
        final_model = pickle.load(file)
    return final_model

def word_divide_char(inputs):
    character = []
    for i in inputs:
        character.append(i)
    return character

def check_password_strength(password, saved_vectorizer, final_model):
    password = np.array([password]) 
    vectorized_password = saved_vectorizer.transform(password)
    password_prediction = final_model.predict(vectorized_password)
    return password_prediction

