import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Load dataset
data = pd.read_csv("dataset.csv")

# Step 2: Define features (X) and target (y)
X = data[["SquareFeet"]]  # Independent variable
y = data["Price"]         # Dependent variable

# Step 3: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 4: Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions on test data
y_pred = model.predict(X_test)

# Step 6: Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Step 7: Display results
print("=== Model Evaluation with RMSE ===")
print("Actual Prices:", list(y_test.values))
print("Predicted Prices:", [round(p, 2) for p in y_pred])
print("RMSE:", round(rmse, 2))

# Step 8: Show model equation
print(f"Model Equation: Price = {model.coef_[0]:.2f} * SquareFeet + {model.intercept_:.2f}")

# Step 9: Plot regression line
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", linewidth=2, label="Regression Line")
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.title("House Price Prediction using Linear Regression")
plt.legend()
plt.show()
