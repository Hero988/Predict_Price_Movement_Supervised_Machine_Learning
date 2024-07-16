import os
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import pandas_ta as ta

import matplotlib.pyplot as plt
import seaborn as sns


import joblib

import xgboost as xgb


def fetch_fx_data_mt5(symbol, timeframe_str, start_date, end_date):

    # Define your MetaTrader 5 account number
    account_number = 7855545
    # Define your MetaTrader 5 password
    password = '5jEi%ppS'
    # Define the server name associated with your MT5 account
    server_name ='Eightcap-Demo'

    # Initialize MT5 connection; if it fails, print error message and exit
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    # Attempt to log in with the given account number, password, and server
    authorized = mt5.login(account_number, password=password, server=server_name)
    # If login fails, print error message, shut down MT5 connection, and exit
    if not authorized:
        print("login failed, error code =", mt5.last_error())
        mt5.shutdown()
        quit()
    # On successful login, print a confirmation message
    else:
        print("Connected to MetaTrader 5")

    # Set the timezone to Berlin, as MT5 times are in UTC
    timezone = pytz.timezone("Europe/Berlin")

    # Convert start and end dates to datetime objects, considering the timezone
    start_date = start_date.replace(tzinfo=timezone)
    end_date = end_date.replace(hour=23, minute=59, second=59, tzinfo=timezone)

    # Define a mapping from string representations of timeframes to MT5's timeframe constants
    timeframes = {
        '1H': mt5.TIMEFRAME_H1,
        'DAILY': mt5.TIMEFRAME_D1,
        '12H': mt5.TIMEFRAME_H12,
        '2H': mt5.TIMEFRAME_H2,
        '3H': mt5.TIMEFRAME_H3,
        '4H': mt5.TIMEFRAME_H4,
        '6H': mt5.TIMEFRAME_H6,
        '8H': mt5.TIMEFRAME_H8,
        '1M': mt5.TIMEFRAME_M1,
        '10M': mt5.TIMEFRAME_M10,
        '12M': mt5.TIMEFRAME_M12,
        '15M': mt5.TIMEFRAME_M15,
        '2M': mt5.TIMEFRAME_M2,
        '20M': mt5.TIMEFRAME_M20,
        '3M': mt5.TIMEFRAME_M3,
        '30M': mt5.TIMEFRAME_M30,
        '4M': mt5.TIMEFRAME_M4,
        '5M': mt5.TIMEFRAME_M5,
        '6M': mt5.TIMEFRAME_M6,
        '1MN': mt5.TIMEFRAME_MN1,
        '1W': mt5.TIMEFRAME_W1
    }

    # Retrieve the MT5 constant for the requested timeframe
    timeframe = timeframes.get(timeframe_str)
    # If the requested timeframe is invalid, print error message, shut down MT5, and return None
    if timeframe is None:
        print(f"Invalid timeframe: {timeframe_str}")
        mt5.shutdown()
        return None

    # Fetch the rates for the given symbol and timeframe within the start and end dates
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

    # If no rates were fetched, print error message, shut down MT5, and return None
    if rates is None:
        print("No rates retrieved, error code =", mt5.last_error())
        mt5.shutdown()
        return None
    
    # Convert the fetched rates into a Pandas DataFrame
    rates_frame = pd.DataFrame(rates)
    # Convert the 'time' column from UNIX timestamps to human-readable dates
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')

    # Set the 'time' column as the DataFrame index and ensure its format is proper for datetime
    rates_frame.set_index('time', inplace=True)
    rates_frame.index = pd.to_datetime(rates_frame.index, format="%Y-%m-%d")

    # Check if 'tick_volume' column is present in the fetched data
    if 'tick_volume' not in rates_frame.columns:
        print("tick_volume is not in the fetched data. Ensure it's included in the API call.")
    else:
        print("tick_volume is included in the data.")
    
    # Shut down the MT5 connection before returning the data
    mt5.shutdown()
    
    # Return the prepared DataFrame containing the rates
    return rates_frame

def calculate_indicators(data):
    bollinger_length=12
    bollinger_std_dev=1.5

    # Calculate the 50-period simple moving average of the 'close' price
    data['SMA_50'] = ta.sma(data['close'], length=50)
    # Calculate the 200-period simple moving average of the 'close' price
    data['SMA_200'] = ta.sma(data['close'], length=200)
    
    # Calculate the 50-period exponential moving average of the 'close' price
    data['EMA_50'] = ta.ema(data['close'], length=50)
    # Calculate the 200-period exponential moving average of the 'close' price
    data['EMA_200'] = ta.ema(data['close'], length=200)

    # Calculate the 9-period exponential moving average for scalping strategies
    data['EMA_9'] = ta.ema(data['close'], length=9)
    # Calculate the 21-period exponential moving average for scalping strategies
    data['EMA_21'] = ta.ema(data['close'], length=21)
    
    # Generate original Bollinger Bands with a 20-period SMA and 2 standard deviations
    original_bollinger = ta.bbands(data['close'], length=20, std=2)
    # The 20-period simple moving average for the middle band
    data['SMA_20'] = ta.sma(data['close'], length=20)
    # Upper and lower bands from the original Bollinger Bands calculation
    data['Upper Band'] = original_bollinger['BBU_20_2.0']
    data['Lower Band'] = original_bollinger['BBL_20_2.0']

    # Generate updated Bollinger Bands for scalping with custom length and standard deviation
    updated_bollinger = ta.bbands(data['close'], length=bollinger_length, std=bollinger_std_dev)
    # Assign lower, middle, and upper bands for scalping
    data['Lower Band Scalping'], data['Middle Band Scalping'], data['Upper Band Scalping'] = updated_bollinger['BBL_'+str(bollinger_length)+'_'+str(bollinger_std_dev)], ta.sma(data['close'], length=bollinger_length), updated_bollinger['BBU_'+str(bollinger_length)+'_'+str(bollinger_std_dev)]
    
    # Calculate the MACD indicator and its signal line
    macd = ta.macd(data['close'])
    data['MACD'] = macd['MACD_12_26_9']
    data['Signal_Line'] = macd['MACDs_12_26_9']
    
    # Calculate the Relative Strength Index (RSI) with the specified window length
    data[f'RSI_14'] = ta.rsi(data['close'], length=14).round(2)

    # Calculate the Stochastic Oscillator
    stoch = ta.stoch(data['high'], data['low'], data['close'])
    data['Stoch_%K'] = stoch['STOCHk_14_3_3']
    data['Stoch_%D'] = stoch['STOCHd_14_3_3']

    data['close_price_previous'] = data['close'].shift(1)
    
    data['close_price_percentage_change'] = data['close'].pct_change() * 100
    data['close_price_previous_percentage_change'] = data['close_price_percentage_change'].shift(1)
    
    # Time-based features
    data['day_of_week'] = data.index.dayofweek
    data['month_of_year'] = data.index.month

    # Remove the last row of the DataFrame
    data = data.drop(data.tail(1).index)

    # Remove rows with missing values
    data = data.dropna()

    # Return the data with added indicators
    return data

def save_fx_data(pair, timeframe, directory):
    # Retrieve and store the current date
    current_date = str(datetime.now().date())

    # Hardcoded start date for strategy evaluation
    strategy_start_date_all = "1971-01-04"
    # Use the current date as the end date for strategy evaluation
    strategy_end_date_all = current_date

    # Convert string representation of dates to datetime objects for further processing
    start_date_all = datetime.strptime(strategy_start_date_all, "%Y-%m-%d")
    end_date_all = datetime.strptime(strategy_end_date_all, "%Y-%m-%d")

    # Fetch and prepare the FX data for the specified currency pair and timeframe
    eur_usd_data = fetch_fx_data_mt5(pair, timeframe, start_date_all, end_date_all)

    eur_usd_data = calculate_indicators(eur_usd_data)

    # Check if data was successfully fetched
    if eur_usd_data is not None:
        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save the DataFrame to a CSV file inside the specified directory
        filename = os.path.join(directory, f"{pair}_{timeframe}_data.csv")
        eur_usd_data.to_csv(filename)
        print(f"Data has been saved to {filename}")
    else:
        print("Failed to fetch the data.")

def add_price_movement_labels(df):
    # Initialize the 'movement' column with 'no_change'
    df['movement'] = 'no_change'
    
    # Iterate over the DataFrame to determine the movement
    for i in range(len(df) - 1):
        if df.at[i, 'close'] < df.at[i + 1, 'close']:
            df.at[i, 'movement'] = 'up'
        elif df.at[i, 'close'] > df.at[i + 1, 'close']:
            df.at[i, 'movement'] = 'down'
    
    # The last row's movement is set to 'no_change' since there's no future data to compare
    df.at[len(df) - 1, 'movement'] = 'no_change'
    
    return df

def process_csv(file_path, output_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Add price movement labels
    df = add_price_movement_labels(df)
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_path, index=False)
    print(f"Processed data has been saved to {output_path}")

def load_and_preprocess_data(file_path):
    # Load the data from CSV file
    df = pd.read_csv(file_path)

    # Convert 'time' column to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Check for missing values
    if df.isnull().values.any():
        print("Data contains missing values. Filling missing values with forward fill method.")
        df.fillna(method='ffill', inplace=True)
    
    # Encode categorical variables
    global label_encoder
    label_encoder = LabelEncoder()
    df['movement'] = label_encoder.fit_transform(df['movement'])
    
    return df

def select_features_and_target(df):
    # Define the feature columns and the target column
    feature_cols = [col for col in df.columns if col not in ['movement', 'time']]
    target_col = 'movement'
    
    # Split the data into features (X) and target (y)
    X = df[feature_cols]
    y = df[target_col]
    
    return X, y

def train_and_evaluate_model(X, y, model_to_use):
    # Use TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    if model_to_use == 'random_forest':
        # Initialize the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_to_use == 'xgboost':
        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    elif model_to_use == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=42)

    
    fold = 1
    for train_index, test_index in tscv.split(X):
        print(f"Training fold {fold}...")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        print(f"Fold {fold} results:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
        
        fold += 1
    
    return model

def save_model(model, filename):
    # Save the model to a file
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    # Load the model from a file
    return joblib.load(filename)

def predict_movement(model, X):
    # Make predictions using the model
    return model.predict(X)

def get_data_for_prediction_for_date(file_path, date):
    # Load the output CSV file
    df = pd.read_csv(file_path)
    
    # Convert 'time' column to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Filter the data for the specified date
    result = df[df['time'] == date]
    
    if not result.empty:
        print(result)
    else:
        print("No data available for the specified date.")

    return result

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    print(f"Directory '{directory}' is ready.")

def menu():
    print("\nMenu:")
    print("1. Train model and Test Model")
    print("2. Predict movement for a specific date")
    return input("Enter your choice: ")

if __name__ == "__main__":
    pair = input("Enter the currency pair (e.g., GBPUSD, EURUSD): ").strip().upper()
    timeframe = input("Enter the timeframe (e.g., Daily, 1H): ").strip().upper()

    print("Choose the model to use:")
    print("1. Random Forest")
    print("2. XGBoost")
    print("3. Logistic Regression")
    model_choice_bool = input("Enter your choice (1 or 2): ").strip()

    if model_choice_bool == '1':
        model_choice = 'random_forest'
    elif model_choice_bool == '2':
        model_choice = 'xgboost'
    elif model_choice_bool == '3':
        model_choice = 'logistic_regression'

    # Create a new directory to save everything
    directory = f'{pair}_{timeframe}_{model_choice}'
    create_directory(directory)

    # Input CSV file path
    input_file_path = os.path.join(directory, f'{pair}_{timeframe}_data.csv')
    # Output CSV file path
    output_file_path = os.path.join(directory, f'{pair}_{timeframe}_labeled_data.csv')

    model_file_path = os.path.join(directory, f'{pair}_{timeframe}_price_movement_model.joblib')

    while True:
        choice = menu()
        if choice == '1':
            if not os.path.exists(input_file_path):
                # If the file does not exist, fetch and save the FX data
                save_fx_data(pair, timeframe, directory)

            # Process the CSV file and add labels
            process_csv(input_file_path, output_file_path)

            # Load and preprocess the data
            df = load_and_preprocess_data(output_file_path)
            
            # Select features and target
            X, y = select_features_and_target(df)
            
            # Train and evaluate the model using TimeSeriesSplit
            model = train_and_evaluate_model(X, y, model_choice)

            # Save the trained model
            save_model(model, model_file_path)

            # Load the model for testing
            model = load_model(model_file_path)

            # Input for testing
            #test_start_date = input("Enter the start date for testing (YYYY-MM-DD): ").strip()
            #test_end_date = input("Enter the end date for testing (YYYY-MM-DD): ").strip()

            test_start_date_string = '2024-01-01'
            test_end_date_string = '2024-07-01'

            # Load and preprocess the data
            df = load_and_preprocess_data(output_file_path)
            
            # Select features and target
            X, y = select_features_and_target(df)

            # Convert input dates to datetime
            test_start_date = pd.to_datetime(test_start_date_string)
            test_end_date = pd.to_datetime(test_end_date_string)
            
            # Select test data based on 'time' column
            test_data = df[(df['time'] >= test_start_date) & (df['time'] <= test_end_date)]

            if not test_data.empty:
                X_test = test_data[[col for col in test_data.columns if col not in ['movement', 'time']]]
                y_test = test_data['movement']
                
                # Make predictions
                y_pred = predict_movement(model, X_test)

                # Add the predictions to the test_data DataFrame
                test_data['predicted'] = y_pred

                # Decode the 'movement' and 'predicted' columns
                test_data['movement'] = label_encoder.inverse_transform(test_data['movement'])
                test_data['predicted'] = label_encoder.inverse_transform(test_data['predicted'])
                
                # Select the relevant columns and save to a new CSV file
                output_columns = ['time', 'close', 'movement', 'predicted']
                output_data = test_data[output_columns]
                # Save the DataFrame to a CSV file inside the specified directory
                output_filename = f'predicted_movement_for_{pair}_{timeframe}_for_dates_{test_start_date_string}_to_{test_end_date_string}.csv'
                output_filepath = os.path.join(directory, output_filename)
                output_data.to_csv(output_filepath, index=False)
                
                # Evaluate predictions
                print("Testing results:")
                print("Accuracy:", accuracy_score(y_test, y_pred))
                print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
                print("Classification Report:\n", classification_report(y_test, y_pred))

                # Filter out 'no change' from y_test and y_pred
                valid_indices = y_test != label_encoder.transform(['no_change'])[0]
                filtered_y_test = y_test[valid_indices]
                filtered_y_pred = y_pred[valid_indices]

                # Exclude 'no change' from the labels
                filtered_labels = [label for label in label_encoder.classes_ if label != 'no_change']

                # Visualize Confusion Matrix
                cm = confusion_matrix(filtered_y_test, filtered_y_pred, labels=label_encoder.transform(filtered_labels))
                plt.figure(figsize=(10, 7))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=filtered_labels, yticklabels=filtered_labels)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title(f'Confusion Matrix: {test_start_date_string} - {test_end_date_string}')
                # Save the confusion matrix image to the directory
                confusion_matrix_image_path = os.path.join(directory, 'confusion_matrix.png')
                plt.savefig(confusion_matrix_image_path)
                
            else:
                print("No data available for the given date range.")
            break

        if choice == '2':
            # Load the model for testing
            model = load_model(model_file_path)

            load_and_preprocess_data(output_file_path)

            # Input for date-based prediction
            prediction_date = input("Enter the date for prediction (YYYY-MM-DD): ").strip()
            prediction_date = pd.to_datetime(prediction_date)
            
            # Get the data for the specified date
            data_for_prediction = get_data_for_prediction_for_date(input_file_path, prediction_date)
            
            if data_for_prediction is not None:
                X_predict = data_for_prediction[[col for col in data_for_prediction.columns if col not in ['movement', 'time']]]
                # Make a prediction using the model
                predicted_movement = predict_movement(model, X_predict)
                predicted_movement = label_encoder.inverse_transform(predicted_movement)
                print(f"Predicted movement for {prediction_date.date()}: {predicted_movement[0]}")
            break




    

    



