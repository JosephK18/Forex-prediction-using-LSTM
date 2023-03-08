from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('joseph5.h5')  # Load the trained LSTM model

# Load the dataset and convert the timestamp column
data= pd.read_csv('EURUSD.CSV')
data['Gmt time'] = data['Gmt time'].apply(lambda x: int(datetime.timestamp(datetime.strptime(x, '%d.%m.%Y %H:%M:%S.%f'))))

array = data.to_numpy(dtype=float)

scaler = MinMaxScaler()  # Create a scaler object for normalizing the data
scaler.fit(array)  # Fit the scaler on the entire dataset

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/predict')
def predict():
    ...

@app.route('/predict-high-low-prices', methods=['POST'])
def predict_high_low_prices():
    data = request.form.to_dict()
    date_time_str = data['datetime']
    date_time_obj = datetime.strptime(date_time_str, '%d.%m.%Y %H:%M:%S.%f')
    time_steps = 24  # set the number of time steps
    input_data = data.loc[date_time_obj-datetime.timedelta(hours=time_steps-1):date_time_obj]
    input_data = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns, index=input_data.index)
    X_input = np.array(input_data).reshape((1, time_steps, input_data.shape[1]))
    y_pred = model.predict(X_input)[0]
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
    return render_template('index2.html', high_price=y_pred[0], low_price=y_pred[1])


if __name__ == '__main__':
    app.run(debug=True)
