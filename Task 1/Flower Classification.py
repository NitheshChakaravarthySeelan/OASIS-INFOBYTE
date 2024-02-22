import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset from iris.csv
iris_df = pd.read_csv("Iris.csv")

# Check for missing values
print("Missing Values:")
print(iris_df.isnull().sum())

# Print the head of the Iris dataset
print("Head of Iris Dataset:")
print(iris_df.head())

# Print the description of the Iris dataset
print("\nDescription of Iris Dataset:")
print(iris_df.describe())

# Print the frequency table of the Species column
print("\nFrequency Table of Species:")
print(iris_df['Species'].value_counts())

# Encode the target labels
label_encoder = LabelEncoder()
iris_df['Species'] = label_encoder.fit_transform(iris_df['Species'])

# Visualize the data using scatter plots
sns.pairplot(iris_df, hue='Species', diag_kind='kde')
plt.show()

# Separate features (X) and target labels (y)
X = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the number of nearest neighbors
n_neighbors = 5

# Train the KNN classifier
clf= KNeighborsClassifier(n_neighbors=n_neighbors)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Convert numerical predictions back to original labels
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Convert y_test to strings
y_test_labels = label_encoder.inverse_transform(y_test)

# Calculate accuracy
accuracy = accuracy_score(y_test_labels, y_pred_labels)

# Print accuracy
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test_labels, y_pred_labels))

# Input details from the user
sepal_length = float(input("Enter Sepal Length (cm): "))
sepal_width = float(input("Enter Sepal Width (cm): "))
petal_length = float(input("Enter Petal Length (cm): "))
petal_width = float(input("Enter Petal Width (cm): "))

# Make prediction for the user input
user_input = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=X.columns)
user_pred = clf.predict(user_input)
predicted_species = label_encoder.inverse_transform(user_pred)

# Display the predicted species
print("Predicted Species",predicted_species[0])