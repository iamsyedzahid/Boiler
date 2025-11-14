import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple 
from datetime import datetime

# --- 1. Configuration (Updated Scope) ---
TICKER = 'BTC-USD'
# Using 2013 start date to capture all available history (yfinance starts BTC around 2014)
START_DATE = '2013-01-01' 
# End date set to the current date for the most up-to-date model
END_DATE = datetime.now().strftime('%Y-%m-%d')
VOLATILITY_WINDOW = 30 # Long-term volatility target
SMA_WINDOW = 14
LAG_DAYS = 1
TEST_SIZE = 0.2
MODEL_PATH = 'ridge_model.joblib'
SCALER_PATH = 'scaler.joblib'


# --- 2. Data Fetcher ---
class DataFetcher:
    def fetch_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """ Fetches historical OHLCV data using yfinance. """
        print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        try:
            # Note: YF.download() defaults auto_adjust=True now, which is generally good
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                raise ValueError("Fetched DataFrame is empty.")
            print("Data fetching successful.")
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()


# --- 3. Feature Engineer ---
class FeatureEngineer:
    def create_features(self, df: pd.DataFrame, window_size: int) -> Tuple[pd.DataFrame, pd.Series]:
        """ Calculates returns, 30-day volatility (target), SMA, and lagged features. """
        df = df.copy()

        # 1. Daily Returns
        df['Daily_Return'] = df['Close'].pct_change()

        # 2. Target Variable (Y): Rolling Volatility (Shifted to predict the *next* day)
        df['Volatility_Target'] = df['Daily_Return'].rolling(window=window_size).std().shift(-1)

        # 3. Predictor Feature (X) 1: 14-day Simple Moving Average (SMA)
        df['SMA_14'] = df['Close'].rolling(window=SMA_WINDOW).mean()

        # 4. Predictor Feature (X) 2: 1-day Lagged Returns
        df['Lagged_Return'] = df['Daily_Return'].shift(LAG_DAYS)
        
        # 5. Predictor Feature (X) 3: Current Volatility (Lagged Target)
        df['Lagged_Volatility'] = df['Volatility_Target'].shift(LAG_DAYS)

        # Define Features (X) and Target (Y) and drop NaNs
        X = df[['SMA_14', 'Lagged_Return', 'Lagged_Volatility']]
        Y = df['Volatility_Target']

        data = pd.concat([X, Y], axis=1).dropna()
        
        X_clean = data.drop(columns=['Volatility_Target'])
        Y_clean = data['Volatility_Target']

        print(f"Feature engineering complete. Total samples: {len(X_clean)}")
        return X_clean, Y_clean


# --- 4. Model Pipeline ---
class ModelPipeline:
    def split_data(self, X: pd.DataFrame, Y: pd.Series, test_size: float = TEST_SIZE) -> Tuple:
        """ Splits data into training and testing sets sequentially (time-series). """
        test_samples = int(len(X) * test_size)
        split_idx = len(X) - test_samples

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        Y_train, Y_test = Y.iloc[:split_idx], Y.iloc[split_idx:]
        
        print(f"Data split: Train samples={len(X_train)}, Test samples={len(X_test)}")
        return X_train, X_test, Y_train, Y_test

    def train_model(self, X_train: pd.DataFrame, Y_train: pd.Series, model_path: str, scaler_path: str):
        """ Scales features, trains Ridge Regressor, and saves model/scaler. """
        
        scaler = StandardScaler()
        # FIX: Fit on the NumPy array (.values) to prevent UserWarning
        X_train_scaled = scaler.fit_transform(X_train.values)
        joblib.dump(scaler, scaler_path)
        
        model = Ridge(alpha=1.0) 
        model.fit(X_train_scaled, Y_train)
        joblib.dump(model, model_path)
        
        print(f"Model and Scaler saved to {model_path} and {scaler_path}")
    
    def evaluate_model(self, X_test: pd.DataFrame, Y_test: pd.Series, model_path: str, scaler_path: str) -> Tuple[dict, pd.Series, pd.Series]:
        """ Loads saved model/scaler, makes predictions, and calculates metrics. """
        try:
            scaler = joblib.load(scaler_path)
            model = joblib.load(model_path)
        except FileNotFoundError:
            return {"Error": "Model or Scaler file not found."}, Y_test, pd.Series([])

        # FIX: Transform using the NumPy array (.values) 
        X_test_scaled = scaler.transform(X_test.values)
        Y_pred = model.predict(X_test_scaled)
        
        metrics = {
            "MAE (Mean Absolute Error)": mean_absolute_error(Y_test, Y_pred),
            "RMSE (Root Mean Square Error)": np.sqrt(mean_squared_error(Y_test, Y_pred)),
            "R2_Score": r2_score(Y_test, Y_pred)
        }
        
        Y_pred_series = pd.Series(Y_pred, index=Y_test.index)
        
        return metrics, Y_test, Y_pred_series


# --- 5. Main Execution Block ---
def run_pipeline():
    """ Orchestrates the entire ML pipeline. """
    fetcher = DataFetcher()
    engineer = FeatureEngineer()
    pipeline = ModelPipeline()

    # 1. Fetch Data
    raw_data = fetcher.fetch_data(TICKER, START_DATE, END_DATE)
    if raw_data.empty: return

    # 2. Feature Engineering (using the 30-day window)
    X, Y = engineer.create_features(raw_data, VOLATILITY_WINDOW)
    
    # 3. Data Splitting
    X_train, X_test, Y_train, Y_test = pipeline.split_data(X, Y, test_size=TEST_SIZE)

    # 4. Model Training and Persistence
    pipeline.train_model(X_train, Y_train, MODEL_PATH, SCALER_PATH)

    # 5. Model Evaluation
    metrics, Y_actual_test, Y_pred_test = pipeline.evaluate_model(X_test, Y_test, MODEL_PATH, SCALER_PATH)

    # Print Final Results
    print("\n--- Final Model Evaluation Results (Test Set) ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    print("-------------------------------------------------")
    
    # 6. Visualization
    plt.figure(figsize=(12, 6))
    Y_actual_test.plot(label=f'Actual Volatility ($\sigma_{VOLATILITY_WINDOW}$)', color='blue', linewidth=1.5)
    Y_pred_test.plot(label=f'Predicted Volatility ($\sigma_{VOLATILITY_WINDOW}$)', color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    plt.title(f'{TICKER} {VOLATILITY_WINDOW}-Day Volatility Prediction (Actual vs. Predicted on Test Set)')
    plt.ylabel(f'{VOLATILITY_WINDOW}-Day Rolling Volatility (Std Dev)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_pipeline()