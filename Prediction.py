import numpy as np
import joblib
import pandas as pd

# Load the scaler, y_scaler, model, and KMeans
scaler = joblib.load('scaler.joblib')
y_scaler = joblib.load('y_scaler.joblib')
model = joblib.load('electrical_conductivity_model.joblib')
kmeans = joblib.load('kmeans.joblib')

# Prompt the user to input fundamental properties
print("Please enter the following properties of the material:")

# Fundamental features
fundamental_features = ['Density', 'H Cond', 'Sp Heat', 'Atm Mass', 'MFP Phonon', 'Atm R', 'Electrons']

input_data = {}
for feature in fundamental_features:
    value = float(input(f"{feature}: "))
    input_data[feature] = value

# Calculate derived features
input_data['feature1'] = input_data['H Cond'] ** 2
input_data['feature2'] = input_data['MFP Phonon'] * input_data['H Cond']
input_data['log_H_Cond'] = np.log(input_data['H Cond'] + 1)
input_data['log_Density'] = np.log(input_data['Density'] + 1)
input_data['log_H_Cond+ log_Density'] = np.log(input_data['Sp Heat'] + 1)
input_data['c1'] = input_data['feature1'] ** 0.5 + input_data['Sp Heat'] + input_data['feature2']
input_data['c2'] = input_data['feature1'] ** 2 * input_data['Sp Heat'] ** 3 + input_data['Density']

# Prepare input for KMeans prediction
cluster_input = pd.DataFrame({
    'H Cond': [input_data['H Cond']],
    'Sp Heat': [input_data['Sp Heat']],
    'Density': [input_data['Density']]
})

# Predict the cluster
input_data['cluster'] = kmeans.predict(cluster_input)[0]

# Prepare the input features in the correct order
feature_names = [
    'Density', 'H Cond', 'feature1', 'feature2', 'log_H_Cond',
    'log_Density', 'log_H_Cond+ log_Density', 'c1', 'c2', 'cluster'
]

input_features = pd.DataFrame([input_data], columns=feature_names)

# Apply scaling to the input features
input_features_scaled = scaler.transform(input_features)

# Predict using the model
prediction_scaled = model.predict(input_features_scaled)
prediction = y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).ravel()

# Output the prediction
print(f"\nPredicted Electrical Conductivity: {prediction[0]}")
print('\nYour Input Data Was:')

# Verify derived features and scaling
for feature, value in input_data.items():
    print(f"{feature}: {value}")
