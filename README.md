# LSTM-BlackScholes-OptionPricing

This project implements a hybrid model to predict European call option prices and their Greeks (Delta, Gamma, Theta, Vega) using the Black-Scholes formula and Long Short-Term Memory (LSTM) networks. Historical market data is fetched from Yahoo Finance to train and evaluate the model.

## Features
- **Black-Scholes Pricing**: Calculate European call option prices and Greeks.
- **LSTM-based Forecasting**: Predict future option prices and Greeks using LSTM.
- **Market Data Integration**: Download and preprocess SPY data, including daily returns and volatility.
- **Error Analysis**: Compare real vs predicted Greeks with visualizations.
  
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/LSTM-BlackScholes-OptionPricing.git
   cd LSTM-BlackScholes-OptionPricing
   
2. Install the required dependencies:
    ```bash
   pip3 install -r requirements.txt
   
## Usage
1. Run the script to download market data, train the model, and generate predictions:
   ```bash
   python3 main.py
2. Output graphs and figures will be saved in the figures/ directory.

## Requirements
Python 3.x
TensorFlow/Keras
scikit-learn
pandas
numpy
matplotlib
yfinance
scipy

## Results
1. Model Evaluation: Metrics such as MSE, MAE, MAPE, and R² are used to assess model performance.
2. Visualizations: Graphs comparing real vs predicted option prices and Greeks are generated.
