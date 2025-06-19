# Sales Prediction Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")


print("Loading the dataset...")
df = pd.read_csv(r'C:\vs\CODSOFT\Sales Prediction\advertising.csv')


print("\nFirst 5 rows:")
print(df.head())

print("\nData Info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

df.dropna(inplace=True)


# Insight: TV and Radio ads correlate most with Sales; Newspaper less so.
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# Visualize the distribution of Sales
sns.pairplot(df)
plt.suptitle("Pairplot of Features", y=1.02)
plt.tight_layout()
plt.show()


X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


print("\nTraining Linear Regression model...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


y_pred_lr = lr_model.predict(X_test)
print("\nLinear Regression Predictions:{ y_pred_lr[:5]}")

# Evaluate Linear Regression model
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print("\nLinear Regression Evaluation:")
print(f"Mean Squared Error: {mse_lr:.2f}")
print(f"R2 Score: {r2_lr:.2f}")


print("\nTraining Random Forest Regressor model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=1)
rf_model.fit(X_train, y_train)


y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Predictions:", y_pred_rf[:5])

# Evaluate Random Forest model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("\nRandom Forest Evaluation:")
print(f"Mean Squared Error: {mse_rf:.2f}")
print(f"R2 Score: {r2_rf:.2f}")

# Visualize predictions vs actual
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_lr, color='green', label='Linear Regression')
plt.scatter(y_test, y_pred_rf, color='orange', label='Random Forest', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.legend()
plt.tight_layout()
plt.show()

# Predict function
def predict_sales(tv, radio, newspaper):
    input_data = np.array([[tv, radio, newspaper]])
    return rf_model.predict(input_data)[0]

# Example usage
print("\nExample Prediction:")
predicted_sales = predict_sales(230.1, 37.8, 69.2)
print(f"Predicted Sales: {predicted_sales:.2f}")

