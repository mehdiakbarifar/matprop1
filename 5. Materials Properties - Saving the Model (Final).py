import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data.csv')

# Create features as previously defined
df['feature1'] = df['H Cond'] ** 2
df['feature2'] = df['MFP Phonon'] * df['H Cond']
df['log_H_Cond'] = np.log(df['H Cond'] + 1)
df['log_Density'] = np.log(df['Density'] + 1)
df['log_H_Cond+ log_Density'] = np.log(df['Sp Heat'] + 1)
df['c1'] = df['feature1'] ** 0.5 + df['Sp Heat'] + df['feature2']
df['c2'] = df['feature1'] ** 2 * df['Sp Heat'] ** 3 + df['Density']

# Cluster features
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['H Cond', 'Sp Heat', 'Density']])

# Select key features
X = df.drop(columns=['Atm Mass', 'Element', 'MFP Phonon', 'Atm R', 'E Cond', 'Electrons', 'Sp Heat'])
y = df['E Cond']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Scale the target variable (optional if needed)
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

# Save the scalers
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(y_scaler, 'y_scaler.joblib')
joblib.dump(kmeans, 'kmeans.joblib')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Initialize and train the RandomForest model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'Initial MSE with simplified features is: {mse}')
print(f'Mean Absolute Error: {mae}')

# Perform cross-validation using KFold
kf = KFold(n_splits=10, shuffle=True, random_state=42)
mse_scores = []
for train_index, test_index in kf.split(X_scaled):
    X_train_fold, X_test_fold = X_scaled[train_index], X_scaled[test_index]
    y_train_fold, y_test_fold = y_scaled[train_index], y_scaled[test_index]
    
    model.fit(X_train_fold, y_train_fold)
    y_pred_fold = model.predict(X_test_fold)
    mse_scores.append(mean_squared_error(y_test_fold, y_pred_fold))

# Compute the average cross-validated MSE
cv_mse = np.mean(mse_scores)
print(f'Standard Cross-Validated MSE: {cv_mse}')

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [10, 20, 30, 50],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f'Best Hyperparameters: {best_params}')

# Train the final model with the best hyperparameters
final_model = RandomForestRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)

# Make final predictions and evaluate
y_pred = final_model.predict(X_test)
final_mse = mean_squared_error(y_test, y_pred)
print(f'Final MSE with simplified features: {final_mse}')

# Final feature importances
final_importance = final_model.feature_importances_
for feature, importance in zip(X.columns, final_importance):
    print(f'{feature}: {importance}')

# Perform cross-validation using KFold
kf = KFold(n_splits=10, shuffle=True, random_state=42)
mse_scores = []
for train_index, test_index in kf.split(X_scaled):
    X_train_fold, X_test_fold = X_scaled[train_index], X_scaled[test_index]
    y_train_fold, y_test_fold = y_scaled[train_index], y_scaled[test_index]
    
    final_model.fit(X_train_fold, y_train_fold)
    y_pred_fold = final_model.predict(X_test_fold)
    mse_scores.append(mean_squared_error(y_test_fold, y_pred_fold))

# Compute the average cross-validated MSE
cv_mse = np.mean(mse_scores)
print(f'Standard Cross-Validated MSE: {cv_mse}')

# Save the final model and KMeans objects
joblib.dump(final_model, 'electrical_conductivity_model.joblib')
joblib.dump(kmeans, 'kmeans.joblib')

# Save the final dataset with predictions
df.to_csv('final.csv')

# Plot predicted vs. actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Electrical Conductivity')
plt.ylabel('Predicted Electrical Conductivity')
plt.title('Predicted vs. Actual Electrical Conductivity')
plt.show()


#7877041621:AAGHM8hqQ55oNXKjoYyqm2Wz6VVciNqLm-Y
