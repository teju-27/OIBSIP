import pandas as pd  
import numpy as np  
import os  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_absolute_error, mean_squared_error  
from sklearn.preprocessing import StandardScaler  
file_path = "car_data.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Error: '{file_path}' not found in {os.getcwd()}")
df = pd.read_csv(file_path)
print("Preview of Dataset:")
print(f"Current Working Directory: {os.getcwd()}")
print(df.head())
df['Fuel_Type'] = df['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2})
df['Selling_type'] = df['Selling_type'].map({'Dealer': 0, 'Individual': 1})
df['Transmission'] = df['Transmission'].map({'Manual': 0, 'Automatic': 1})
df['Car_Age'] = 2025 - df['Year']
features = ['Car_Age', 'Present_Price', 'Driven_kms', 'Fuel_Type', 'Selling_type', 'Transmission', 'Owner']
target = 'Selling_Price'
for col in features + [target]:
    if col not in df.columns:
        raise ValueError(f"Missing required column: '{col}' in dataset")
df = df.dropna()  
X = df[features]  
y = df[target]  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression()  
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)  
mse = mean_squared_error(y_test, y_pred)  
rmse = np.sqrt(mse)  
print("\nModel Evaluation:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
new_car = np.array([[5, 7.5, 50000, 0, 0, 1, 0]]).reshape(1, -1)
new_car_scaled = scaler.transform(new_car)
predicted_price = model.predict(new_car_scaled)
print("\nPredicted Selling Price for the new car:")
print(f"${predicted_price[0]:,.2f}")