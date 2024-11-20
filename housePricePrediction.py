# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv(r'''D:\Programming\vsjupiter\AIML training\matlib\Bengaluru_House_Data.csv''')
# Display the first few rows
print(data.head())

# Preprocess the data
# Drop unnecessary columns
data = data.drop(columns=['area_type', 'society', 'availability', 'balcony'])

# Drop rows with missing values
data = data.dropna()

# Feature Engineering: Convert the 'size' column (e.g., '3 BHK') to numerical values
data['size'] = data['size'].apply(lambda x: int(x.split(' ')[0]))

# Convert 'total_sqft' to a numerical value (handle ranges by averaging)
def convert_sqft_to_num(x):
    try:
        return float(x)
    except:
        if '-' in x:
            tokens = x.split('-')
            return (float(tokens[0]) + float(tokens[1])) / 2
        return None

data['total_sqft'] = data['total_sqft'].apply(convert_sqft_to_num)
data = data.dropna(subset=['total_sqft'])  # Drop rows where conversion failed

# Select features and target variable
X = data[['total_sqft', 'bath', 'size']]
y = data['price']  # Target variable (price)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the StandardScaler to normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error and R2 Score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Plot actual vs predicted prices for a quick visualization
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.show()
