from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def predict_price():
    if request.method == 'POST':
        room_number = float(request.form['room_number'])
        area = float(request.form['area'])
        has_parking = int(request.form['has_parking'])
        has_storage = int(request.form['has_storage'])
        has_balcony = int(request.form['has_balcony'])
        has_mamad = int(request.form['has_mamad'])
        type = request.form['type']
        City = request.form['City']
        has_elevator = int(request.form['has_elevator'])
        has_air_condition = int(request.form['has_air_condition'])
        handicap_friendly = int(request.form['handicap_friendly'])
        type = 1 if type == 'בית פרטי' else 0

        # Create a DataFrame with the input data
        data = pd.DataFrame({
            'room_number': [room_number],
            'Area': [area],
            'type': [type],
            'City': [City],
            'hasElevator': [has_elevator],
            'hasParking': [has_parking],
            'hasStorage': [has_storage],
            'hasAirCondition': [has_air_condition],
            'hasBalcony': [has_balcony],
            'hasMamad': [has_mamad],
            'handicapFriendly': [handicap_friendly]
        })

        predicted_price = model.predict(data)[0]

        return render_template('index.html', predicted_price=predicted_price)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
