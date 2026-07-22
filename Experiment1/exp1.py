import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data["price"] = housing.target

x = data[['AveRooms']]
y = data['price'].values

# Split Data

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Scaling

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

# Gradient Descent

w = 0
b = 0

learning_rate = 0.01
epochs = 1000

n = len(x_train_scaled)

for i in range(epochs):
    y_pred = w * x_train_scaled.flatten() + b
    dw = (1 / n) * np.sum((y_pred - y_train) * x_train_scaled.flatten())
    db = (1 / n) * np.sum(y_pred - y_train)
    w = w - learning_rate * dw
    b = b - learning_rate * db

    if i % 100 == 0:
        cost = (1 / (2 * n)) * np.sum((y_pred - y_train) ** 2)
        print(f"Epoch {i}, Cost {cost:.4f}")


y_pred_gd = w * x_test_scaled.flatten() + b

print("\nGradient Descent:")
print("----------------")
print("Weight:", w)
print("Bias:", b)
print("MSE:", mean_squared_error(y_test, y_pred_gd))
print("R2 Score:", r2_score(y_test, y_pred_gd))

# Normal Equation

X_train_ne = np.c_[np.ones((len(X_train),1)), X_train]
X_test_ne = np.c_[np.ones((len(X_test),1)), X_test]

theta = np.linalg.inv(X_train_ne.T @ X_train_ne) @ X_train_ne.T @ y_train
y_pred_ne = X_test_ne @ theta

index=0
for i in theta:
    print (f"Theta {index}: ",i)
    index+=1


