import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import yfinance as yf
import os
import sys
from scipy.stats import norm
import warnings
import tensorflow as tf
warnings.filterwarnings("ignore")

# Versione di Python
if sys.version_info[0] < 3:
    raise Exception("Python 3 è richiesto per eseguire questo script.")

# Verifica Directory - altrimenti crearla
if not os.path.exists("figures"):
    os.makedirs("figures")

# Calcolo prezzo opzione call europea usando Black-Scholes
def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Calcolo delle opzioni greche
def black_scholes_greeks(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return delta, gamma, theta, vega

# Scarico dei dati di mercato
def get_market_data(start_date='2010-01-01', end_date='2024-01-01'):
    spy = yf.download('SPY', start=start_date, end=end_date)
    spy['Returns'] = spy['Adj Close'].pct_change()  # Funzione per il calcolo dei rendimenti giornalieri
    
    # Calcolare la volatilità annualizzata a 30 giorni
    spy['Volatility'] = spy['Returns'].rolling(window=30).std() * np.sqrt(252)
    spy['T'] = (pd.to_datetime(end_date) - spy.index).days / 365.0  # Tempo alla scadenza in anni
    
    # Tasso di interesse (approssimato) senza rischio storico dai Treasury Bond
    spy['r'] = 0.01  # Placeholder per contesto Hp (in comtesto reale bisognerebbe usare dati storici)
    
    # Strike Price approssimato
    K = spy['Adj Close'].mean()
    spy['K'] = K
    spy.dropna(inplace=True)
    
    # Modello Black-Scholes per calcolo prezzo opzioni call
    spy['Call_Price'] = black_scholes_call_price(spy['Adj Close'], spy['K'], spy['T'], spy['r'], spy['Volatility'])
    
    # Greche con Black-Scholes
    spy['Delta'], spy['Gamma'], spy['Theta'], spy['Vega'] = black_scholes_greeks(spy['Adj Close'], spy['K'], spy['T'], spy['r'], spy['Volatility'])

    return spy

# Finestre temporali per LSTM
def create_windowed_data(X, y, window_size=30):
    X_windowed, y_windowed = [], []
    for i in range(window_size, len(X)):
        X_windowed.append(X[i-window_size:i])
        y_windowed.append(y[i])
    return np.array(X_windowed), np.array(y_windowed)

# Compilazione modello LSTM
def create_lstm_model(input_shape, output_size):
    model = Sequential()
    model.add(LSTM(100, activation='tanh', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(50, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(output_size)) 
    model.compile(optimizer=Adam(learning_rate=0.001), loss=tf.keras.losses.Huber())
    return model

# Visualizzazione Greche
def plot_greeks(data, predicted_greeks):
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(data.index, data['Delta'], label='Delta Reale', color='blue')
    plt.plot(data.index[window_size:], predicted_greeks['Delta'], label='Delta Predetto', color='cyan', linestyle='--')
    plt.xlabel('Data')
    plt.ylabel('Delta')
    plt.title('Andamento Delta nel Tempo')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(data.index, data['Gamma'], label='Gamma Reale', color='green')
    plt.plot(data.index[window_size:], predicted_greeks['Gamma'], label='Gamma Predetto', color='lightgreen', linestyle='--')
    plt.xlabel('Data')
    plt.ylabel('Gamma')
    plt.title('Andamento Gamma nel Tempo')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(data.index, data['Theta'], label='Theta Reale', color='red')
    plt.plot(data.index[window_size:], predicted_greeks['Theta'], label='Theta Predetto', color='orange', linestyle='--')
    plt.xlabel('Data')
    plt.ylabel('Theta')
    plt.title('Andamento Theta nel Tempo')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(data.index, data['Vega'], label='Vega Reale', color='purple')
    plt.plot(data.index[window_size:], predicted_greeks['Vega'], label='Vega Predetto', color='magenta', linestyle='--')
    plt.xlabel('Data')
    plt.ylabel('Vega')
    plt.title('Andamento Vega nel Tempo')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("figures/greeks_over_time_with_predictions.png")
    plt.show()

# Scarico Dati di mercato
data = get_market_data()

# Separazione features e target
X = data[['Adj Close', 'K', 'T', 'r', 'Volatility', 'Delta', 'Gamma', 'Theta', 'Vega']]
y = data[['Call_Price', 'Delta', 'Gamma', 'Theta', 'Vega']]

# Standardizzazione features del target
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Suddivisione del dataset
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_scaled[train_index], y_scaled[test_index]

# Finestre temporali LSTM
window_size = 30
X_train_windowed, y_train_windowed = create_windowed_data(X_train, y_train, window_size)
X_test_windowed, y_test_windowed = create_windowed_data(X_test, y_test, window_size)

# Addestramento LSTM
model = create_lstm_model((X_train_windowed.shape[1], X_train_windowed.shape[2]), y_train_windowed.shape[1])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train_windowed, y_train_windowed, epochs=200, batch_size=32, verbose=1,
                    validation_data=(X_test_windowed, y_test_windowed), callbacks=[early_stopping])

# Predizione su tutto dataset
X_full_windowed, _ = create_windowed_data(X_scaled, y_scaled, window_size)
y_full_pred_scaled = model.predict(X_full_windowed)

# Inversione scala previsioni
y_full_pred = scaler_y.inverse_transform(y_full_pred_scaled)

# Separazione previsioni delle greche
predicted_greeks = pd.DataFrame(y_full_pred, columns=['Call_Price', 'Delta', 'Gamma', 'Theta', 'Vega'])

# Valutazione del modello
mse = mean_squared_error(y[window_size:], y_full_pred)
mae = mean_absolute_error(y[window_size:], y_full_pred)
mape = mean_absolute_percentage_error(y[window_size:], y_full_pred)
r2 = r2_score(y[window_size:], y_full_pred)

print(f"MSE: {mse}, MAE: {mae}, MAPE: {mape}, R^2: {r2}")

# Grafico prezzi (predetti e reali a confronto)
plt.figure(figsize=(14, 7))
plt.plot(data.index[window_size:], y['Call_Price'][window_size:], label='Prezzo Call Reale (Black-Scholes)', color='blue')
plt.plot(data.index[window_size:], predicted_greeks['Call_Price'], label='Prezzo Call Predetto', color='red', linestyle='--')
plt.xlabel('Data')
plt.ylabel('Prezzo Call')
plt.title('Andamento Temporale dei Prezzi Call Reali e Predetti (2010-2024)')
plt.legend()
plt.grid(True)
plt.savefig("figures/andamento_temporale_reale_vs_predetto_2010_2024.png")
plt.show()

# Grafico Greche
plot_greeks(data, predicted_greeks)

# Grafico errore Greche (predette e reali)
plt.figure(figsize=(14, 10))

try:
    plt.subplot(2, 2, 1)
    plt.plot(data.index[window_size:], data['Delta'].iloc[window_size:] - predicted_greeks['Delta'].values, label='Errore Delta', color='blue')
    plt.xlabel('Data')
    plt.ylabel('Errore Delta')
    plt.title('Errore Delta nel Tempo')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(data.index[window_size:], data['Gamma'].iloc[window_size:] - predicted_greeks['Gamma'].values, label='Errore Gamma', color='green')
    plt.xlabel('Data')
    plt.ylabel('Errore Gamma')
    plt.title('Errore Gamma nel Tempo')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(data.index[window_size:], data['Theta'].iloc[window_size:] - predicted_greeks['Theta'].values, label='Errore Theta', color='red')
    plt.xlabel('Data')
    plt.ylabel('Errore Theta')
    plt.title('Errore Theta nel Tempo')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(data.index[window_size:], data['Vega'].iloc[window_size:] - predicted_greeks['Vega'].values, label='Errore Vega', color='purple')
    plt.xlabel('Data')
    plt.ylabel('Errore Vega')
    plt.title('Errore Vega nel Tempo')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("figures/error_greeks_over_time.png")
    plt.show()
except Exception as e:
    print(f"Errore durante la creazione dei grafici degli errori delle greche nel tempo: {e}")

# Grafico Greche Distribuzione errori
plt.figure(figsize=(14, 10))

try:
    plt.subplot(2, 2, 1)
    plt.hist((data['Delta'].iloc[window_size:] - predicted_greeks['Delta'].values), bins=30, color='blue', alpha=0.7, label='Errore Delta')
    plt.xlabel('Errore Delta')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione Errore Delta')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.hist((data['Gamma'].iloc[window_size:] - predicted_greeks['Gamma'].values), bins=30, color='green', alpha=0.7, label='Errore Gamma')
    plt.xlabel('Errore Gamma')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione Errore Gamma')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.hist((data['Theta'].iloc[window_size:] - predicted_greeks['Theta'].values), bins=30, color='red', alpha=0.7, label='Errore Theta')
    plt.xlabel('Errore Theta')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione Errore Theta')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.hist((data['Vega'].iloc[window_size:] - predicted_greeks['Vega'].values), bins=30, color='purple', alpha=0.7, label='Errore Vega')
    plt.xlabel('Errore Vega')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione Errore Vega')
    plt.legend()

    plt.tight_layout()
    plt.savefig("figures/distribution_error_greeks.png")
    plt.show()
except Exception as e:
    print(f"Errore durante la creazione dei grafici delle distribuzioni degli errori delle greche: {e}")

# Grafico Greche Errori Assoluti Medi nel tempo
plt.figure(figsize=(14, 7))

try:
    plt.plot(data.index[window_size:], np.abs(data['Delta'].iloc[window_size:] - predicted_greeks['Delta'].values), label='Errore Assoluto Delta', color='blue')
    plt.plot(data.index[window_size:], np.abs(data['Gamma'].iloc[window_size:] - predicted_greeks['Gamma'].values), label='Errore Assoluto Gamma', color='green')
    plt.plot(data.index[window_size:], np.abs(data['Theta'].iloc[window_size:] - predicted_greeks['Theta'].values), label='Errore Assoluto Theta', color='red')
    plt.plot(data.index[window_size:], np.abs(data['Vega'].iloc[window_size:] - predicted_greeks['Vega'].values), label='Errore Assoluto Vega', color='purple')

    plt.xlabel('Data')
    plt.ylabel('Errore Assoluto Medio')
    plt.title('Errore Assoluto Medio nel Tempo per ciascuna Greca')
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/mae_greeks_over_time.png")
    plt.show()
except Exception as e:
    print(f"Errore durante la creazione del grafico degli errori assoluti medi nel tempo: {e}")