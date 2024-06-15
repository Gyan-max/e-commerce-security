import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("mail_data.csv")
print(df)

data = df.where(pd.notnull(df), " ")
data.head()
data.info()
data.shape

data.loc[data['Category'] == 'spam', 'Category',] = 0
data.loc[data['Category'] == 'ham', 'Category',] = 1

x = data['Message']
y = data['Category']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print(x.shape)
print(x_train.shape)
print(x_test.shape)

print(y.shape)
print(y_train.shape)
print(y_test.shape)

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

# Fit the vectorizer only on the training data
x_train_features = feature_extraction.fit_transform(x_train)

# Transform the test data using the learned vocabulary
x_test_features = feature_extraction.transform(x_test)
x_train_features.shape[1]

# Save the vectorizer
with open("vectorizer.pickle", "wb") as f:
    pickle.dump(feature_extraction,f)

y_train = y_train.astype('int')
y_test = y_test.astype("int")

# Train the model only once
model = RandomForestClassifier()
model.fit(x_train_features, y_train)

# Evaluate the model on the training data
prediction_on_training_data = model.predict(x_train_features)
accuracy_on_training_data = accuracy_score(y_train, prediction_on_training_data)
print("Accuracy on training data : ", accuracy_on_training_data)

# Evaluate the model on the test data
prediction_on_test_data = model.predict(x_test_features)
accuracy_on_test_data = accuracy_score(y_test, prediction_on_test_data)
print("Accuracy on test data : ", accuracy_on_test_data)

# Get user input for email text
input_email = input("Enter the email text: ")
input_data = feature_extraction.transform([input_email])  # Transform the input email

prediction = model.predict(input_data)

print(prediction)

if prediction[0] == 1:
    print("Ham mail")
else:
    print("Spam mail")

# Save the trained model
with open("model.pickle", "wb") as f:
    pickle.dump(model, f)
