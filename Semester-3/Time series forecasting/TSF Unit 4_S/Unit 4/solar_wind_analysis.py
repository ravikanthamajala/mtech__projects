# ---------------------------
# 0. Imports & settings
# ---------------------------
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# Optional: pmdarima for auto_arima (if installed)
try:
    import pmdarima as pm
except Exception:
    pm = None

# Optional: Prophet
try:
    from prophet import Prophet
except Exception:
    try:
        from fbprophet import Prophet
    except Exception:
        Prophet = None

# Optional: TensorFlow/Keras for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
except Exception:
    tf = None

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12,5)

# ---------------------------
# 1. Load the Airline Passengers dataset
#    We use the well-known CSV from GitHub (J. Brownlee)
# ---------------------------
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
df.index.freq = 'MS'   # Monthly start frequency
df.columns = ['Passengers']
series = df['Passengers'].astype(float)
print("Loaded Airline Passengers dataset. Range:", series.index.min().date(), "->", series.index.max().date())
display(series.head())

# ---------------------------
# 2. Extensive EDA & plots
# ---------------------------
# 2.1 Basic line plot
plt.figure()
plt.plot(series, marker='o')
plt.title('Airline Passengers - Monthly')
plt.ylabel('Passengers (1000s)')
plt.show()

# 2.2 Histogram + KDE
plt.figure()
sns.histplot(series, kde=True, bins=20)
plt.title('Distribution: Passengers')
plt.show()

# 2.3 Summary statistics
print(series.describe())

# 2.4 Missing / zeros
print("Missing values:", series.isna().sum())

# 2.5 Yearly boxplot (trend confirmation)
df_box = series.to_frame().copy()
df_box['Year'] = df_box.index.year
df_box['MonthNum'] = df_box.index.month
df_box['Month'] = df_box.index.month_name()
plt.figure(figsize=(14,6))
sns.boxplot(x='Year', y='Passengers', data=df_box)
plt.title('Yearly Boxplot (trend check)')
plt.show()

# 2.6 Monthly boxplot (seasonality check) - order months correctly
month_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
plt.figure(figsize=(14,6))
sns.boxplot(x='Month', y='Passengers', data=df_box, order=month_order)
plt.title('Monthly Boxplot (seasonality check)')
plt.xticks(rotation=30)
plt.show()

# 2.7 Monthplot / Quarterplot (statsmodels helper)
fig, axes = plt.subplots(2,1, figsize=(12,8))
sm.graphics.tsa.month_plot(series, ax=axes[0])
sm.graphics.tsa.quarter_plot(series, ax=axes[1])
plt.suptitle('Month plot and Quarter plot')
plt.show()

# 2.8 Rolling statistics (12-month)
rolling_mean = series.rolling(12).mean()
rolling_std  = series.rolling(12).std()
plt.figure()
plt.plot(series, label='Original')
plt.plot(rolling_mean, label='Rolling Mean (12)')
plt.plot(rolling_std, label='Rolling Std (12)')
plt.legend()
plt.title('Rolling Mean & Std Dev (12)')
plt.show()

# ---------------------------
# 3. Decomposition (additive & STL)
# ---------------------------
# Additive decomposition (for demonstration - series is multiplicative, but we'll show both)
decomp_add = seasonal_decompose(series, model='additive', period=12)
decomp_add.plot()
plt.suptitle('Additive Decomposition (12-month)')
plt.show()

# Multiplicative decomposition (more appropriate for this dataset)
decomp_mul = seasonal_decompose(series, model='multiplicative', period=12)
decomp_mul.plot()
plt.suptitle('Multiplicative Decomposition (12-month)')
plt.show()

# STL decomposition
stl = STL(series, period=12, robust=True).fit()
stl.plot()
plt.suptitle('STL decomposition')
plt.show()

# ---------------------------
# 4. Stationarity tests (ADF)
# ---------------------------
def adf_print(x, name="series"):
    res = adfuller(x.dropna())
    print(f"ADF for {name}: statistic={res[0]:.4f}, p-value={res[1]:.4f}")
    for k, v in res[4].items():
        print(f"  critical value {k}: {v:.4f}")
    print()

adf_print(series, "Raw series")
adf_print(np.log(series).diff().dropna(), "log(series) first diff")

# ---------------------------
# 5. ACF / PACF plots (raw, differenced, seasonal diff)
# ---------------------------
plot_acf(series.dropna(), lags=48)
plt.title('ACF - Raw')
plt.show()
plot_pacf(series.dropna(), lags=48, method='ywm')
plt.title('PACF - Raw')
plt.show()

# First difference
series_diff1 = series.diff().dropna()
plot_acf(series_diff1, lags=48)
plt.title('ACF - 1st Difference')
plt.show()
plot_pacf(series_diff1, lags=48, method='ywm')
plt.title('PACF - 1st Difference')
plt.show()

# Seasonal difference (lag=12)
series_seas_diff = series.diff(12).dropna()
plot_acf(series_seas_diff, lags=48)
plt.title('ACF - Seasonal Difference (12)')
plt.show()
plot_pacf(series_seas_diff, lags=48, method='ywm')
plt.title('PACF - Seasonal Difference (12)')
plt.show()

# Seasonal + first difference
series_sd_fd = series.diff(12).diff().dropna()
plot_acf(series_sd_fd, lags=48)
plt.title('ACF - Seasonal(12) + 1st Difference')
plt.show()
plot_pacf(series_sd_fd, lags=48, method='ywm')
plt.title('PACF - Seasonal(12) + 1st Difference')
plt.show()

# ---------------------------
# 6. Train/test split (last 24 months as test)
# ---------------------------
test_periods = 24
train = series.iloc[:-test_periods]
test  = series.iloc[-test_periods:]
print("Train:", train.index.min().date(), "to", train.index.max().date(), "n=", len(train))
print("Test:", test.index.min().date(), "to", test.index.max().date(), "n=", len(test))

# ---------------------------
# 7. Exponential Smoothing family
#    - SES, Holt, Holt-Winters (add & mul)
# ---------------------------
# 7.1 Simple Exponential Smoothing (SES)
ses_model = SimpleExpSmoothing(train).fit(optimized=True)
ses_fore = ses_model.forecast(len(test))
print("SES RMSE:", sqrt(mean_squared_error(test, ses_fore)))

# 7.2 Holt's linear (trend)
holt_model = Holt(train).fit(optimized=True)
holt_fore = holt_model.forecast(len(test))
print("Holt RMSE:", sqrt(mean_squared_error(test, holt_fore)))

# 7.3 Holt-Winters (additive seasonality, additive trend)
hw_add = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit(optimized=True)
hw_add_fore = hw_add.forecast(len(test))
print("HW (add) RMSE:", sqrt(mean_squared_error(test, hw_add_fore)))

# 7.4 Holt-Winters (multiplicative seasonal, additive trend)
hw_mul = ExponentialSmoothing(train, trend='add', seasonal='mul', seasonal_periods=12).fit(optimized=True)
hw_mul_fore = hw_mul.forecast(len(test))
print("HW (mul) RMSE:", sqrt(mean_squared_error(test, hw_mul_fore)))

# 7.5 Plot these forecasts
plt.figure()
train.plot(label='train')
test.plot(label='test')
ses_fore.plot(label='SES')
holt_fore.plot(label='Holt')
hw_add_fore.plot(label='HW_add')
hw_mul_fore.plot(label='HW_mul')
plt.legend()
plt.title('Exponential Smoothing family forecasts')
plt.show()

# ---------------------------
# 8. Holt-Winters hyperparameter grid search (alpha, beta, gamma)
#    Keep grid small for demo; extend for production.
# ---------------------------
alphas = [None, 0.2, 0.4, 0.6]
betas = [None, 0.05, 0.2]
gammas = [None, 0.05, 0.2]
trends = ['add', None]
seasonals = ['add', 'mul']

hw_results = []
best_hw = None
best_rmse = np.inf

for trend in trends:
    for seasonal in seasonals:
        for a in alphas:
            for b in betas:
                for g in gammas:
                    try:
                        model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal, seasonal_periods=12)
                        fit = model.fit(smoothing_level=a, smoothing_slope=b, smoothing_seasonal=g, optimized=(a is None))
                        pred = fit.forecast(len(test))
                        rmse = sqrt(mean_squared_error(test, pred))
                        hw_results.append({'trend':trend, 'seasonal':seasonal, 'alpha':a, 'beta':b, 'gamma':g, 'rmse':rmse})
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_hw = {'fit':fit, 'config':(trend, seasonal, a, b, g), 'rmse':rmse}
                    except Exception as e:
                        continue

hw_df = pd.DataFrame(hw_results).sort_values('rmse').reset_index(drop=True)
print("Top 6 Holt-Winters configs by RMSE:")
display(hw_df.head(6))
print("Best HW RMSE:", best_hw['rmse'], "config:", best_hw['config'])

# plot best hw
best_hw_fore = best_hw['fit'].forecast(len(test))
plt.figure()
train.plot(label='train')
test.plot(label='test')
best_hw_fore.plot(label='Best_HW')
plt.legend(); plt.title('Best Holt-Winters forecast (grid search)')
plt.show()

# ---------------------------
# 9. ARIMA / SARIMA modeling & manual grid-search
#    We'll use log transform + seasonal diff guidance.
# ---------------------------
# Transform: take log to stabilize variance
series_log = np.log(series)
train_log = series_log.iloc[:-test_periods]
test_log = series_log.iloc[-test_periods:]

# Manual small grid for (p,d,q) and seasonal (P,D,Q,12)
import itertools
p = q = range(0,3)
d = range(0,2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(P, D, Q, 12) for P in range(0,2) for D in range(0,2) for Q in range(0,2)]

best_aic = np.inf
best_order = None
best_seasonal = None
best_sarima_res = None
sarima_results = []

for order in pdq:
    for seasonal in seasonal_pdq:
        try:
            mod = SARIMAX(train_log, order=order, seasonal_order=seasonal,
                          enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False, maxiter=50)
            aic = res.aic
            pred_log = res.get_forecast(steps=len(test_log)).predicted_mean
            pred = np.exp(pred_log)  # back to original scale
            rmse = sqrt(mean_squared_error(test, pred))
            sarima_results.append({'order':order, 'seasonal':seasonal, 'aic':aic, 'rmse':rmse})
            if aic < best_aic:
                best_aic = aic
                best_order = order
                best_seasonal = seasonal
                best_sarima_res = res
        except Exception as e:
            continue

sarima_df = pd.DataFrame(sarima_results).sort_values('aic').reset_index(drop=True)
print("Top SARIMA candidates by AIC:")
display(sarima_df.head(6))
print("Best SARIMA order:", best_order, "seasonal:", best_seasonal, "AIC:", best_aic)

# Forecast & evaluate best SARIMA
if best_sarima_res is not None:
    pred_log = best_sarima_res.get_forecast(steps=len(test_log)).predicted_mean
    sarima_fore = np.exp(pred_log)
    print("Best SARIMA RMSE:", sqrt(mean_squared_error(test, sarima_fore)))
    plt.figure()
    train.plot(label='train')
    test.plot(label='test')
    sarima_fore.plot(label='SARIMA_best')
    plt.legend(); plt.title('Best SARIMA forecast')
    plt.show()
    # Diagnostics
    best_sarima_res.plot_diagnostics(figsize=(12,8))
    plt.show()

# ---------------------------
# 10. (Optional) auto_arima with pmdarima if available
# ---------------------------
if pm is not None:
    try:
        auto = pm.auto_arima(train_log, seasonal=True, m=12, stepwise=True, suppress_warnings=True, trace=True, error_action='ignore')
        print("pmdarima auto_arima order:", auto.order, "seasonal_order:", auto.seasonal_order)
        auto_pred_log = pd.Series(auto.predict(n_periods=len(test_log)), index=test_log.index)
        auto_pred = np.exp(auto_pred_log)
        print("auto_arima RMSE:", sqrt(mean_squared_error(test, auto_pred)))
    except Exception as e:
        print("auto_arima failed:", e)
else:
    print("pmdarima not installed; skipping auto_arima.")

# ---------------------------
# 11. Prophet (if installed)
# ---------------------------
if Prophet is not None:
    # Prophet needs ds,y columns (ds datetime, y numeric)
    prophet_df = series.reset_index().rename(columns={'Month':'ds','Passengers':'y'})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    # fit on train only
    prophet_train_df = prophet_df.iloc[:-test_periods]
    m.fit(prophet_train_df)
    future = m.make_future_dataframe(periods=test_periods, freq='MS')
    forecast = m.predict(future)
    prop_pred = forecast.set_index('ds')['yhat'][-test_periods:]
    prop_pred.index = test.index  # align index
    print("Prophet RMSE:", sqrt(mean_squared_error(test, prop_pred)))
    # Plot
    plt.figure()
    train.plot(label='train'); test.plot(label='test'); prop_pred.plot(label='Prophet')
    plt.legend(); plt.title('Prophet forecast')
    plt.show()
else:
    print("Prophet not installed; skip Prophet modeling.")

# ---------------------------
# 12. RNN / LSTM model (multi-layer) — simple demo
# ---------------------------
if tf is not None:
    # Prepare dataset: we'll use log(series) scaled [0,1] and create lag features (window=12)
    data = series.values.reshape(-1,1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # create supervised dataset with lag=12 (use last 24 months as test)
    lag = 12
    X, y = [], []
    for i in range(len(data_scaled)-lag):
        X.append(data_scaled[i:i+lag, 0])
        y.append(data_scaled[i+lag, 0])
    X = np.array(X)
    y = np.array(y)
    # reshape for LSTM: [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # train/test split corresponding to earlier split of last 24 months
    train_X = X[:-test_periods]
    train_y = y[:-test_periods]
    test_X = X[-test_periods:]
    test_y = y[-test_periods:]

    print("LSTM shapes:", train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # Build multi-layer LSTM
    model = Sequential()
    model.add(LSTM(100, activation='tanh', return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(LSTM(50, activation='tanh', return_sequences=True))
    model.add(LSTM(25, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Model summary
    model.summary()

    # Train (small epochs for demo; increase for production)
    history = model.fit(train_X, train_y, epochs=50, batch_size=16, validation_data=(test_X, test_y), verbose=1)

    # Forecast - one-step predictions for test set
    preds = model.predict(test_X)
    preds_inv = scaler.inverse_transform(preds)  # back to original scale
    test_y_inv = scaler.inverse_transform(test_y.reshape(-1,1))
    # create pandas series for plotting aligned to test index
    lstm_pred_series = pd.Series(preds_inv.flatten(), index=test.index)
    lstm_true_series = pd.Series(test_y_inv.flatten(), index=test.index)
    print("LSTM RMSE:", sqrt(mean_squared_error(lstm_true_series, lstm_pred_series)))

    plt.figure()
    train.plot(label='train')
    test.plot(label='test')
    lstm_pred_series.plot(label='LSTM')
    plt.legend(); plt.title('LSTM Forecast (test set predictions)')
    plt.show()
else:
    print("TensorFlow not installed; skipping LSTM demo. To run it install TensorFlow (pip install tensorflow).")

# ---------------------------
# 13. Model comparison (collect RMSEs)
# ---------------------------
results = []

# SES
results.append({'model':'SES', 'rmse': sqrt(mean_squared_error(test, ses_fore)), 'mae': mean_absolute_error(test, ses_fore)})

# Holt
results.append({'model':'Holt', 'rmse': sqrt(mean_squared_error(test, holt_fore)), 'mae': mean_absolute_error(test, holt_fore)})

# HW add/mul and best hw
results.append({'model':'HW_add', 'rmse': sqrt(mean_squared_error(test, hw_add_fore)), 'mae': mean_absolute_error(test, hw_add_fore)})
results.append({'model':'HW_mul', 'rmse': sqrt(mean_squared_error(test, hw_mul_fore)), 'mae': mean_absolute_error(test, hw_mul_fore)})
results.append({'model':'HW_best_grid', 'rmse': best_hw['rmse'], 'mae': None})

# SARIMA best
if best_sarima_res is not None:
    results.append({'model':'SARIMA_best', 'rmse': sqrt(mean_squared_error(test, sarima_fore)), 'mae': mean_absolute_error(test, sarima_fore)})

# pmdarima auto_arima if used
if pm is not None and 'auto_pred' in locals():
    results.append({'model':'pmdarima_auto', 'rmse': sqrt(mean_squared_error(test, auto_pred)), 'mae': mean_absolute_error(test, auto_pred)})

# Prophet
if Prophet is not None and 'prop_pred' in locals():
    results.append({'model':'Prophet', 'rmse': sqrt(mean_squared_error(test, prop_pred)), 'mae': mean_absolute_error(test, prop_pred)})

# LSTM
if tf is not None and 'lstm_pred_series' in locals():
    results.append({'model':'LSTM', 'rmse': sqrt(mean_squared_error(lstm_true_series, lstm_pred_series)), 'mae': mean_absolute_error(lstm_true_series, lstm_pred_series)})

comp_df = pd.DataFrame(results).sort_values('rmse').reset_index(drop=True)
print("Model comparison on test set (sorted by RMSE):")
display(comp_df)

# ---------------------------
# 14. Save key outputs (optional)
# ---------------------------
comp_df.to_csv('airline_model_comparison.csv', index=False)
print("Saved model comparison to airline_model_comparison.csv")

# End of script
