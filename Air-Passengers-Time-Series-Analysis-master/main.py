import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    data['Month'] = pd.to_datetime(data['Month'])
    data.set_index('Month', inplace=True)
    return data

def forecast_arima(data, steps):
    model = ARIMA(data, order=(5,1,0))
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=steps)
    return forecast


def get_forecast_steps():
    while True:
        try:
            forecast_steps = int(input("Enter the number of steps to forecast: "))
            if forecast_steps <= 0:
                print("Please enter a positive integer.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.")
    return forecast_steps

def plot_forecast(data, forecast_df):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Passengers'], label='Original Data')
    plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast')
    plt.xlabel('Month')
    plt.ylabel('Number of Air Passengers')
    plt.title('Time Series Forecasting using ARIMA')
    plt.legend()
    plt.show()

def main():
    file_path = 'C:\\Users\\Admin\\Desktop\\R1\\Air-Passengers-Time-Series-Analysis-master\\Data\\AirPassengers.csv'

    data = load_data(file_path)
    data = preprocess_data(data)
    forecast_steps = get_forecast_steps()
    forecast = forecast_arima(data, forecast_steps)
    forecast_index = pd.date_range(start=data.index[-1], periods=forecast_steps+1, freq='M')[1:]
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecast'])
    plot_forecast(data, forecast_df)
    print("Forecasted passenger counts for the next {} months:".format(forecast_steps))
    print(forecast_df)

if __name__ == "__main__":
    main()
