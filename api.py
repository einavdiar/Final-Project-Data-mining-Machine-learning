from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the HTML form
    area = float(request.form['area'])
    num_of_images = int(request.form['num_of_images'])
    floor = int(request.form['floor'])
    total_floors = int(request.form['total_floors'])
    hasElevator = int(request.form['hasElevator'])
    hasParking = int(request.form['hasParking'])
    hasBars = int(request.form['hasBars'])
    hasStorage = int(request.form['hasStorage'])
    condition = request.form['condition']
    hasAirCondition = int(request.form['hasAirCondition'])
    hasBalcony = int(request.form['hasBalcony'])
    hasMamad = int(request.form['hasMamad'])
    HandicapFriendly = int(request.form['HandicapFriendly'])
    entrance_date = request.form['entrance_date']
    furniture = request.form['furniture']
    published_days = int(request.form['published_days'])

    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        'area': [area],
        'num_of_images': [num_of_images],
        'floor': [floor],
        'total_floors': [total_floors],
        'hasElevator': [hasElevator],
        'hasParking': [hasParking],
        'hasBars': [hasBars],
        'hasStorage': [hasStorage],
        'condition': [condition],
        'hasAirCondition': [hasAirCondition],
        'hasBalcony': [hasBalcony],
        'hasMamad': [hasMamad],
        'HandicapFriendly': [HandicapFriendly],
        'entrance_date': [entrance_date],
        'furniture': [furniture],
        'published_days': [published_days]
    })

    # Make the prediction
    prediction = model.predict(input_data)

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
