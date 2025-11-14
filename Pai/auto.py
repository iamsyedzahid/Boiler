import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. CONFIGURATION ---
TICKER = 'BTC-USD'
MODEL_PATH = 'ridge_model.joblib'
SCALER_PATH = 'scaler.joblib'

# Parameters needed for feature calculation
VOLATILITY_WINDOW = 7 
SMA_WINDOW = 14

def fetch_and_calculate_latest_features(ticker: str) -> dict:
    """
    Fetches the necessary historical data and calculates the three required features 
    for the model's prediction.
    """
    # We need enough data to calculate the 14-day SMA and the 7-day volatility.
    # Fetch data for the last 50 days to ensure all rolling window calculations are valid.
    end_date = datetime.now()
    start_date = end_date - timedelta(days=50)

    print(f"Fetching recent data for {ticker}...")
    try:
        # Fetch OHLCV data
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty or len(df) < SMA_WINDOW + 2:
            raise ValueError("Insufficient data fetched.")
            
        df['Daily_Return'] = df['Close'].pct_change()

        # Calculate the 7-day volatility for the *current* day (this will be Lagged_Volatility)
        df['Volatility_Target'] = df['Daily_Return'].rolling(window=VOLATILITY_WINDOW).std().shift(-1)
        
        # 1. 14-day Simple Moving Average (SMA_14)
        df['SMA_14'] = df['Close'].rolling(window=SMA_WINDOW).mean()

        # 2. 1-Day Lagged Returns (The return from the second-to-last day, since we shift)
        df['Lagged_Return'] = df['Daily_Return'].shift(1)
        
        # 3. Lagged Volatility (The 7-day vol. calculated for the previous day)
        df['Lagged_Volatility'] = df['Volatility_Target'].shift(1)

        # Drop NaNs and take the LAST complete row of features
        features = df[['SMA_14', 'Lagged_Return', 'Lagged_Volatility']].dropna().iloc[-1]
        
        # Prepare the data dictionary
        latest_features = {
            'SMA_14': [features['SMA_14']],
            'Lagged_Return': [features['Lagged_Return']],
            'Lagged_Volatility': [features['Lagged_Volatility']]
        }
        
        print("Feature calculation complete.")
        return latest_features

    except Exception as e:
        print(f"Error during data fetching/calculation: {e}")
        return None


def predict_tomorrow_volatility(new_features: dict):
    """
    Loads the saved model and scaler to predict volatility for new input features.
    """
    try:
        # Load the Model and Scaler
        scaler = joblib.load(SCALER_PATH)
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {MODEL_PATH} or {SCALER_PATH}. Ensure files are in the same folder.")
        return

    # Prepare the Data for the Model
    X_new = pd.DataFrame(new_features)
    
    # Ensure compatibility with the scaler by using .values
    X_new_scaled = scaler.transform(X_new.values) 
    
    # Make the Prediction
    prediction = model.predict(X_new_scaled)
    
    # --- Output ---
    print("\n--- Volatility Prediction Input & Result ---")
    print(f"Prediction Date: {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}")
    
    # --- FIX THE FORMATTING ERRORS HERE ---
    # Access the first element [0] and explicitly convert it to float()
    # (The FutureWarning will likely persist due to internal library code, but the TypeError will be fixed)
    print(f"Current SMA_14:          {float(new_features['SMA_14'][0].iloc[0]):.2f}")
    print(f"Yesterday's Return:      {float(new_features['Lagged_Return'][0].iloc[0])*100:.3f}%")
    print(f"Today's Volatility (Lag): {float(new_features['Lagged_Volatility'][0].iloc[0]):.6f}")
    # --- END FIX ---
    
    print("-" * 37)
    print(f"Predicted 7-Day Volatility: {prediction[0]:.6f}")
    print(f"Interpretation: Market uncertainty predicted at {prediction[0]*100:.4f}% std dev.")
    print("--------------------------------------------")

if __name__ == "__main__":
    latest_features = fetch_and_calculate_latest_features(TICKER)
    
    if latest_features:
        predict_tomorrow_volatility(latest_features)