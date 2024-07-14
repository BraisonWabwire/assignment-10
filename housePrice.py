import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
file_path = 'C:/Users/braisonW/Desktop/G/assignment#10/housePrices.csv'
data = pd.read_csv(file_path)

# Define a function to calculate adjusted R-squared
def adjusted_r2_score(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

# Define a function to create and evaluate a model
def create_and_evaluate_model(data, independent_vars, dependent_var='SalePrice'):
    X = data[independent_vars]
    y = data[dependent_var]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    adj_r2 = adjusted_r2_score(r2, X_test.shape[0], X_test.shape[1])
    
    # Print results
    print(f"Independent Variables: {independent_vars}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-Squared Value: {r2}")
    print(f"Adjusted R-Squared Value: {adj_r2}")
    print("-" * 50)
    
    return model, rmse, r2, adj_r2

# Model 1
independent_vars1 = ['Overall Qual', 'Total Liv Area', 'Garage Area']
model1, rmse1, r21, adj_r21 = create_and_evaluate_model(data, independent_vars1)

# Model 2
independent_vars2 = ['Overall Cond', 'Year Built', 'Total Bsmt SF']
model2, rmse2, r22, adj_r22 = create_and_evaluate_model(data, independent_vars2)

# Model 3
independent_vars3 = ['Full Bath', 'Half Bath', 'Bedroom AbvGr']
model3, rmse3, r23, adj_r23 = create_and_evaluate_model(data, independent_vars3)

# Ensure all models meet the criteria
models_meet_criteria = all([
    rmse1 <= 36000, adj_r21 > 0.78,
    rmse2 <= 36000, adj_r22 > 0.78,
    rmse3 <= 36000, adj_r23 > 0.78,
])

print("Do all models meet the criteria?", models_meet_criteria)
