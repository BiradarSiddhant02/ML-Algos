import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import root_mean_squared_error, r2_score

# Step 1: Load the dataset
data = pd.read_csv('data.csv')

# Step 2: Preprocess the data
# Handle missing values (if any)
data.fillna(data.mean(), inplace=True)

# Define features and target variable
features = ['RAM', 'ROM', 'Mobile_Size', 'Primary_Cam', 'Selfi_Cam', 'Battery_Power']
target = 'Price'

# Extract features and target variable from the dataset
X = data[features]
y = data[target]

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a linear regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Print model coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
