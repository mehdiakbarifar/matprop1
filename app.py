from flask import Flask, request
import numpy as np
import joblib
import pandas as pd
from telegram import Bot, Update
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, Filters

app = Flask(__name__)

# Load the models
scaler = joblib.load('scaler.joblib')
y_scaler = joblib.load('y_scaler.joblib')
model = joblib.load('electrical_conductivity_model.joblib')
kmeans = joblib.load('kmeans.joblib')

# Telegram Bot setup
TOKEN = "7877041621:AAGHM8hqQ55oNXkjoYyqm2Wz6VVciNqLm-Y"
bot = Bot(token=TOKEN)
dispatcher = Dispatcher(bot, None, use_context=True)

# Define a command handler to start the bot
def start(update, context):
    update.message.reply_text("Please enter the following properties of the material:\n"
                              "Density, H Cond, Sp Heat, Atm Mass, MFP Phonon, Atm R, Electrons")

# Define a message handler to process the input data
def handle_message(update, context):
    try:
        input_text = update.message.text.split(',')
        input_data = {key: float(value) for key, value in zip(
            ['Density', 'H Cond', 'Sp Heat', 'Atm Mass', 'MFP Phonon', 'Atm R', 'Electrons'], input_text)}

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
        update.message.reply_text(f"Predicted Electrical Conductivity: {prediction[0]}")

        # Verify derived features and scaling
        for feature, value in input_data.items():
            update.message.reply_text(f"{feature}: {value}")

    except Exception as e:
        update.message.reply_text("There was an error processing your input. Please make sure the data is in the correct format.")

# Add handlers
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

# Route to handle Telegram webhook
@app.route(f'/{TOKEN}', methods=['POST'])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return "ok"

if __name__ == '__main__':
    app.run(debug=True)
