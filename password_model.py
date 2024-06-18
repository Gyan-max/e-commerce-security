import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Load the data
data = pd.read_csv("pass.csv", on_bad_lines='skip')

# Display the first few rows of the dataframe to understand its structure
# print(data.head())
# df = pd.DataFrame(data)

columns_to_remove = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6']

# Remove the specified columns
data = data.drop(columns=columns_to_remove, errors='ignore')


data.shape

data["strength"].unique()

"""0 means->the password strenght is weak

1 means->the password strenght is medium

2 means->the password strenght is strong
"""


data = data.dropna().sample(frac=1).reset_index(drop=True)  # Remove null values and shuffle the data

data.isnull().sum()

data[data["password"].isnull()]

#check duplicate
data.duplicated().sum()

data.isnull().any()


# count of strengths or class value to be predicted
data['strength'].value_counts()

password_tuple = np.array(data)

password_tuple

import random

random.shuffle(password_tuple)

x = [labels[0] for labels in password_tuple]
y = [labels[1] for labels in password_tuple]


def word_divide_char(inputs):
    character = []
    for i in inputs:
        character.append(i)
    return character


# Convert a given word (string) to a list of individual characters
# def word_to_char(word):
#     return list(word)

vectorizer = TfidfVectorizer(tokenizer=word_divide_char)
X = vectorizer.fit_transform(x)

X.shape

vectorizer.get_feature_names_out()

first_document_vector = X[0]
first_document_vector

first_document_vector.T.todense()

df = pd.DataFrame(first_document_vector.T.todense(), index=vectorizer.get_feature_names_out(), columns=['TF-IDF'])
df.sort_values(by=['TF-IDF'], ascending=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X_train.shape

from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)

from sklearn.metrics import accuracy_score

# print('Accuracy (Decision Tree):', accuracy_score(y_test, y_pred))

import pickle

pickle.dump(vectorizer, open("tfidf_password_strength.pickle", "wb"))
pickle.dump(dt_clf, open("final_model.pickle", "wb"))

# with open("tfidf_password_strength.pickle", 'rb') as file:
#     saved_vectorizer = pickle.load(file)  # Load the vectorizer from the pickle file

# with open("final_model.pickle", 'rb') as file:
#     final_model = pickle.load(file)


def test_password_strength(password, vectorizer, model):
    X_password = np.array([password])  # Convert the password to a numpy array
    X_predict = vectorizer.transform(X_password)  # Transform the password using the loaded vectorizer
    y_pred = model.predict(X_predict)  # Predict the password strength using the loaded model
    return y_pred[0]  # Return the predicted password strength


# # Print the first password and its predicted strength
# password1 = 'abc123'
# strength1 = test_password_strength(password1, saved_vectorizer, final_model)
# print(f'Password: {password1}, Strength: {strength1}')
