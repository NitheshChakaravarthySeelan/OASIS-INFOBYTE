import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

# Preprocess the data
data['v2'] = data['v2'].str.lower()
data['v2'] = data['v2'].str.replace('[^\w\s]','')
data['v2'] = data['v2'].str.replace('\s+', ' ')

# Descriptive statistics
print(data.describe())

# Frequency table
print(data['v1'].value_counts())

# Missing values
print(data.isnull().sum())

# Data visualization
plt.figure(figsize=(10,6))
sns.countplot(x='v1', data=data)
plt.title('Frequency of Spam and Ham Emails')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.show()

# Split the dataset into features (X) and target (y)
X = data['v2']
y = data['v1']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the text data into numerical vectors
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Make predictions on the testing set
predictions = model.predict(X_test_vectorized)

# Calculate the accuracy of the model
print("Accuracy:", accuracy_score(y_test, predictions))

# Get input from the user
email = input("Enter an email to check if it's spam or not: ")

# Vectorize the input email
email_vectorized = vectorizer.transform([email])

# Make a prediction on the input email
prediction = model.predict(email_vectorized)

# Print the prediction
if prediction[0] == 'spam':
    print("The email is spam.")
else:
    print("The email is not spam.")