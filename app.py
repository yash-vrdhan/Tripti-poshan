from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

with open('model-2.pkl', 'rb') as file:
    model = pickle.load(file)

def preprocess_input(dayOfMonth, timeOfDay, dayOfWeek, typeOfMeal, isHoliday):

    dayOfMonth = float(dayOfMonth)
    isHoliday = float(isHoliday)

    input_data = {
        'Day': [dayOfMonth],
        'Time_of_Day': [timeOfDay],
        'Day_of_Week': [dayOfWeek],
        'Type_of_Meal': [typeOfMeal],
        'Holiday_Indicator': [isHoliday]
    }

    input_df = pd.DataFrame(input_data)

    input_df_encoded = pd.get_dummies(input_df)

    input_features = [
        'Day', 'Holiday_Indicator',
        'Time_of_Day_Breakfast', 'Time_of_Day_Dinner', 'Time_of_Day_Lunch',
        'Day_of_Week_Friday', 'Day_of_Week_Monday', 'Day_of_Week_Saturday', 'Day_of_Week_Sunday', 'Day_of_Week_Thursday', 'Day_of_Week_Tuesday', 'Day_of_Week_Wednesday',
        'Type_of_Meal_Biryani', 'Type_of_Meal_Chole Bhature', 'Type_of_Meal_Dosa', 'Type_of_Meal_Paneer Butter Masala', 'Type_of_Meal_Poha', 'Type_of_Meal_Rajma Chawal'
    ]

    input_df_processed = input_df_encoded.reindex(columns=input_features, fill_value=0)

    return input_df_processed.values

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        dayOfMonth = request.form['dayOfMonth']
        timeOfDay = request.form['timeOfDay']
        dayOfWeek = request.form['dayOfWeek']
        typeOfMeal = request.form['typeOfMeal']
        isHoliday = request.form['isHoliday']

        input_data = preprocess_input(dayOfMonth, timeOfDay, dayOfWeek, typeOfMeal, isHoliday)


        prediction = model.predict(input_data)


        return f'Prediction: {prediction[0]:.0f}'
    

if __name__ == '__main__':
    app.run(debug=True)

