

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
car_data = pd.read_csv("car data.csv")

# Display the head of the dataset
print("Head of the dataset:")
print(car_data.head())

# Check for missing values
print("\nMissing values in the dataset:")
print(car_data.isnull().sum())

# Describe the data
print("\nDescription of the dataset:")
print(car_data.describe())

# Create a frequency table for categorical variables
print("\nFrequency table for Fuel_Type:")
print(car_data['Fuel_Type'].value_counts())

# Preprocess the data
car_data = pd.get_dummies(car_data, drop_first=True)

# Handle missing column name
car_data.rename(columns={'Driven_kms': 'Kms_Driven'}, inplace=True)

# Define features and target variable
X = car_data.drop(columns='Selling_Price')
y = car_data['Selling_Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Visualize the data
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()


# Define the predict_price function
def predict_price(year, present_price, kms_driven, owner, fuel_type, seller_type, transmission):
    fuel_type_index = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
    seller_type_index = {'Dealer': 0, 'Individual': 1}
    transmission_index = {'Manual': 0, 'Automatic': 1}
    
    fuel_type_encoded = [0, 0, 0]
    fuel_type_encoded[fuel_type_index.get(fuel_type)] = 1
    
    seller_type_encoded = [0, 0]
    seller_type_encoded[seller_type_index.get(seller_type)] = 1
    
    transmission_encoded = [0, 0]
    transmission_encoded[transmission_index.get(transmission)] = 1
    
    input_data = {
        'Year': [year],
        'Present_Price': [present_price],
        'Kms_Driven': [kms_driven],
        'Owner': [owner],
        'Fuel_Type_CNG': [fuel_type_encoded[2]],
        'Fuel_Type_Diesel': [fuel_type_encoded[1]],
        'Fuel_Type_Petrol': [fuel_type_encoded[0]],
        'Seller_Type_Individual': [seller_type_encoded[1]],
        'Transmission_Manual': [transmission_encoded[0]]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # Ensure all necessary features are included
    missing_features = set(X_train.columns) - set(input_df.columns)
    for feature in missing_features:
        input_df[feature] = 0
    
    # Reorder columns to match the order during training
    input_df = input_df[X_train.columns]
    
    predicted_price = model.predict(input_df)
    return predicted_price[0]

# Call the predict_price function with the desired input parameters
predicted_price = predict_price(2017, 10.21, 50000, 0, 'Diesel', 'Dealer', 'Manual')
print("Predicted Price:", predicted_price)
