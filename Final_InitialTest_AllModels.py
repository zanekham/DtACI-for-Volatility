###############################################################################
# Both CursorAI and OpenAI's GPT-o1 and GPT-4o models were used to improve the
# efficiency of the codes developed for this project. In this program one 
# notable use was changing how the ML and GARCH models were defined, which
# dramatically improved the run time of this code. 
###############################################################################

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from arch import arch_model

plt.ioff() 

###############################################################################
# Parameters
###############################################################################
zoom_days = 100
alpha = 0.2
gamma_candidates = np.array([0.005, 0.01, 0.05, 0.1])
sigma = 0.05
eta = 0.01
stock_ticker = "BA"
start_date = "2012-01-01"

plt.rcParams.update({'font.size': 26})

# Downloading Data

print("Downloading stock data...")
data = yf.download(stock_ticker, start=start_date, auto_adjust=False)

returns = data['Adj Close'].pct_change() * 100
if isinstance(returns, pd.DataFrame):
    returns = returns.squeeze()

if returns.empty:
    raise ValueError("The 'returns' series is empty. Please check the stock data.")

df = pd.DataFrame({'returns': returns})
window_sizes = [5, 10, 15, 20, 25]
for window in window_sizes:
    df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()

df['realized_volatility'] = df['returns'].rolling(window=5).std().shift(-1)

feature_columns = [f'volatility_{window}' for window in window_sizes]
df.dropna(subset=feature_columns + ['realized_volatility'], inplace=True)

X = df[feature_columns]
y = df['realized_volatility']
dates = df.index 

X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)
cleaned_dates = dates 

# Define LSTM Model

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape))
    model.add(Dense(1))
    return model

# Define other ML Models

def train_and_predict_model(model_name, model_fn, X, y, cleaned_dates):
    forecasted_volatility = []
    realized_volatility = []
    conformity_scores = []
    pred_indices = []

    # Introduce 5-Fold Cross Validation

    tscv = TimeSeriesSplit(n_splits=5)
    print(f"\nTraining {model_name} model...")
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Feature Engineering and Scaling

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if model_name == 'LSTM':
            X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
            X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

        model = model_fn()

        # LSTM Params
        if model_name == 'LSTM':
            model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
            es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
            model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test),
                      epochs=50, batch_size=32, verbose=0, callbacks=[es])
            y_pred = model.predict(X_test_scaled).ravel()
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            if model_name == 'LinearRegression':
                print(f"Coefficients for {model_name}: {model.coef_}")
                print(f"Intercept for {model_name}: {model.intercept_}")

        forecasted_volatility.extend(y_pred)
        realized_volatility.extend(y_test)
        conformity_scores.extend(np.abs(y_pred - y_test))
        pred_indices.extend(test_index)

    forecasted_volatility = np.array(forecasted_volatility)
    realized_volatility = np.array(realized_volatility)
    conformity_scores = np.array(conformity_scores)

    min_length = min(len(forecasted_volatility), len(realized_volatility))
    forecasted_volatility = forecasted_volatility[:min_length]
    realized_volatility = realized_volatility[:min_length]
    conformity_scores = conformity_scores[:min_length]
    pred_indices = np.array(pred_indices)[:min_length]

    pred_dates = cleaned_dates[pred_indices]

    # Evaluation Metrics
 
    mse = mean_squared_error(realized_volatility, forecasted_volatility)
    mae = mean_absolute_error(realized_volatility, forecasted_volatility)
    r2 = r2_score(realized_volatility, forecasted_volatility)

    return forecasted_volatility, realized_volatility, conformity_scores, (mse, mae, r2), pred_dates

# Define GARCH Model

def train_and_predict_garch(X, y, returns_series, cleaned_dates):
    forecasted_volatility = []
    realized_volatility = []
    conformity_scores = []
    pred_indices = []

    tscv = TimeSeriesSplit(n_splits=5)
    print("\nTraining GARCH(1,1) model...")
    for train_index, test_index in tscv.split(X):
        train_returns = returns_series.iloc[train_index]
        test_returns = returns_series.iloc[test_index]
        test_realized = y.iloc[test_index]

        current_history = train_returns.copy()
        preds = []
        for i, d_idx in enumerate(test_index):
            # Fit GARCH each day
            am = arch_model(current_history, p=1, q=1, vol='GARCH', dist='normal', mean='zero')
            res = am.fit(disp='off')
            fcast = res.forecast(horizon=1)
            vol_pred = fcast.variance.iloc[-1, 0] ** 0.5
            preds.append(vol_pred)

            # Update history with new day
            current_history = pd.concat([current_history, pd.Series(test_returns.iloc[i],
                                                        index=[test_returns.index[i]])])

        preds = np.array(preds)
        test_realized_array = test_realized.values
        forecasted_volatility.extend(preds)
        realized_volatility.extend(test_realized_array)
        conformity_scores.extend(np.abs(preds - test_realized_array))
        pred_indices.extend(test_index)

    forecasted_volatility = np.array(forecasted_volatility)
    realized_volatility = np.array(realized_volatility)
    conformity_scores = np.array(conformity_scores)
    pred_indices = np.array(pred_indices)

    min_length = min(len(forecasted_volatility), len(realized_volatility))
    forecasted_volatility = forecasted_volatility[:min_length]
    realized_volatility = realized_volatility[:min_length]
    conformity_scores = conformity_scores[:min_length]
    pred_indices = pred_indices[:min_length]

    pred_dates = cleaned_dates[pred_indices]

    # Evaluation Metrics

    mse = mean_squared_error(realized_volatility, forecasted_volatility)
    mae = mean_absolute_error(realized_volatility, forecasted_volatility)
    r2 = r2_score(realized_volatility, forecasted_volatility)

    return forecasted_volatility, realized_volatility, conformity_scores, (mse, mae, r2), pred_dates

# Define DtACI ALgorithm and Plot figures for each Model

def run_dtaci_and_plot(model_name, forecasted_volatility, realized_volatility, conformity_scores, zoom_days, pred_dates):
    num_experts = len(gamma_candidates)
    weights = np.ones(num_experts)
    alpha_t = np.full(num_experts, alpha)

    miscoverages = []
    prediction_intervals = []
    times_outside = 0

    print(f"\nRunning DtACI algorithm for {model_name}...")
    for t in range(len(forecasted_volatility)):
        vol = forecasted_volatility[t]
        true_vol = realized_volatility[t]

        if t > 0:
            conformity_scores_t = np.abs(forecasted_volatility[:t] - realized_volatility[:t])
        else:
            conformity_scores_t = np.array([])

        weights = np.clip(weights, 1e-10, None)
        probabilities = weights / np.sum(weights)
        expert_idx = np.random.choice(num_experts, p=probabilities)

        threshold = np.clip(alpha_t[expert_idx], 0, 1)
        if len(conformity_scores_t) > 0:
            percentile_value = np.clip((1 - threshold) * 100, 0, 100)
            quantile = np.percentile(conformity_scores_t, percentile_value)
        else:
            quantile = vol * 0.1

        lower_bound = max(0, vol - quantile)
        upper_bound = vol + quantile
        prediction_intervals.append((lower_bound, upper_bound))

        if true_vol < lower_bound or true_vol > upper_bound:
            times_outside += 1
            miscoverage = 1
        else:
            miscoverage = 0

        losses = np.abs(alpha_t - miscoverage)
        weights = weights * np.exp(-eta * losses)
        weights = np.clip(weights, 1e-10, None)
        weights = (1 - sigma) * weights + (sigma / num_experts) * np.sum(weights)

        alpha_t[expert_idx] += gamma_candidates[expert_idx] * (alpha - miscoverage)
        alpha_t = np.clip(alpha_t, 0, 1)

        miscoverages.append(miscoverage)

    print(f"Generating plots for {model_name}...")
    time_axis = pred_dates
    lower_bounds, upper_bounds = zip(*prediction_intervals)
    zoom_start = max(0, len(time_axis) - zoom_days)

    # For LinearRegression and GARCH, 4 separate figures (for report)
    if model_name in ["LinearRegression", "GARCH"]:
        # Plot 1: Full period forecast
        fig_full = plt.figure(figsize=(10, 6))
        ax_full = fig_full.add_subplot(111)
        ax_full.plot(time_axis, forecasted_volatility, label="Forecasted Volatility", alpha=0.7)
        ax_full.plot(time_axis, realized_volatility, label="Realized Volatility", alpha=0.7)
        ax_full.fill_between(time_axis, lower_bounds, upper_bounds, color='gray', alpha=0.3, label="Prediction Intervals")
        ax_full.legend()
        ax_full.set_title(f"{stock_ticker} Volatility Forecast with DtACI ({model_name} - Full Period)")
        ax_full.set_xlabel("Date")
        ax_full.set_ylabel("Volatility (%)")

        # Plot 2: Zoomed forecast (with 2-column legend, override legend font size to 14)
        fig_zoom = plt.figure(figsize=(10, 6))
        ax_zoom = fig_zoom.add_subplot(111)
        ax_zoom.plot(time_axis[zoom_start:], forecasted_volatility[zoom_start:], label="Forecasted Volatility", alpha=0.7)
        ax_zoom.plot(time_axis[zoom_start:], realized_volatility[zoom_start:], label="Realized Volatility", alpha=0.7)
        ax_zoom.fill_between(
            time_axis[zoom_start:],
            np.array(lower_bounds)[zoom_start:],
            np.array(upper_bounds)[zoom_start:],
            color='gray',
            alpha=0.3,
            label="Prediction Intervals"
        )
        # leg_zoom = ax_zoom.legend(ncol=2)
        # for text in leg_zoom.get_texts():
        #     text.set_fontsize(24)
        ax_zoom.set_xlabel("Date")
        ax_zoom.set_ylabel("Volatility (%)")

        # Plot 3: Miscoverage rate full period
        miscoverage_rates = np.cumsum(miscoverages) / np.arange(1, len(miscoverages) + 1)
        fig_mis_full = plt.figure(figsize=(10, 6))
        ax_mis_full = fig_mis_full.add_subplot(111)
        ax_mis_full.plot(time_axis, miscoverage_rates, label="Empirical Miscoverage Rate")
        ax_mis_full.axhline(alpha, color="red", linestyle="--", label=f"Target Miscoverage Rate ({alpha})")
        ax_mis_full.legend()
        ax_mis_full.set_title(f"Miscoverage Rate Over Time (Full Period - {model_name})")
        ax_mis_full.set_xlabel("Date")
        ax_mis_full.set_ylabel("Miscoverage Rate")

        # Plot 4: Miscoverage rate zoomed (with 2-column legend, override legend font size to 14)
        fig_mis_zoom = plt.figure(figsize=(10, 6))
        ax_mis_zoom = fig_mis_zoom.add_subplot(111)
        ax_mis_zoom.plot(time_axis[zoom_start:], miscoverage_rates[zoom_start:], label="Empirical Miscoverage Rate")
        ax_mis_zoom.axhline(alpha, color="red", linestyle="--", label=f"Target Miscoverage Rate ({alpha})")
        # leg_mis_zoom = ax_mis_zoom.legend(ncol=2)
        # for text in leg_mis_zoom.get_texts():
        #     text.set_fontsize(24)
        ax_mis_zoom.set_xlabel("Date")
        ax_mis_zoom.set_ylabel("Miscoverage Rate")

        plt.tight_layout()
        final_miscoverage = times_outside / len(miscoverages)
        final_miscoverage_zoom = np.mean(miscoverages[zoom_start:])
        print(f"\n{model_name} Results:")
        print(f"Number of times outside intervals: {times_outside}")
        print(f"Final miscoverage rate: {final_miscoverage:.4f}")
        print(f"Final miscoverage rate (last {zoom_days} days): {final_miscoverage_zoom:.4f}")

        return prediction_intervals, miscoverages, final_miscoverage, final_miscoverage_zoom, [
            fig_full, fig_zoom, fig_mis_full, fig_mis_zoom
        ]
    else:
        # For all other models, 4 tile subplots showing intervals and miscoverage rate over the full time period and zoomend in
        fig, axs = plt.subplots(2, 2, figsize=(20, 12))
        axs = axs.flatten()

        axs[0].plot(time_axis, forecasted_volatility, label="Forecasted Volatility", alpha=0.7)
        axs[0].plot(time_axis, realized_volatility, label="Realized Volatility", alpha=0.7)
        axs[0].fill_between(time_axis, lower_bounds, upper_bounds, color='gray', alpha=0.3, label="Prediction Intervals")
        axs[0].legend()
        axs[0].set_title(f"{stock_ticker} Volatility Forecast with DtACI ({model_name} - Full Period)")
        axs[0].set_xlabel("Date")
        axs[0].set_ylabel("Volatility (%)")

        axs[1].plot(time_axis[zoom_start:], forecasted_volatility[zoom_start:], label="Forecasted Volatility", alpha=0.7)
        axs[1].plot(time_axis[zoom_start:], realized_volatility[zoom_start:], label="Realized Volatility", alpha=0.7)
        axs[1].fill_between(
            time_axis[zoom_start:],
            np.array(lower_bounds)[zoom_start:],
            np.array(upper_bounds)[zoom_start:],
            color='gray',
            alpha=0.3,
            label="Prediction Intervals"
        )
        axs[1].legend()
        axs[1].set_title(f"Last {zoom_days} Days ({model_name})")
        axs[1].set_xlabel("Date")
        axs[1].set_ylabel("Volatility (%)")

        miscoverage_rates = np.cumsum(miscoverages) / np.arange(1, len(miscoverages) + 1)
        axs[2].plot(time_axis, miscoverage_rates, label="Empirical Miscoverage Rate")
        axs[2].axhline(alpha, color="red", linestyle="--", label=f"Target Miscoverage Rate ({alpha})")
        axs[2].legend()
        axs[2].set_title(f"Miscoverage Rate Over Time (Full Period - {model_name})")
        axs[2].set_xlabel("Date")
        axs[2].set_ylabel("Miscoverage Rate")

        axs[3].plot(time_axis[zoom_start:], miscoverage_rates[zoom_start:], label="Empirical Miscoverage Rate")
        axs[3].axhline(alpha, color="red", linestyle="--", label=f"Target Miscoverage Rate ({alpha})")
        axs[3].legend()
        axs[3].set_title(f"Miscoverage Rate (Last {zoom_days} Days - {model_name})")
        axs[3].set_xlabel("Date")
        axs[3].set_ylabel("Miscoverage Rate")

        plt.tight_layout()
        final_miscoverage = times_outside / len(miscoverages)
        final_miscoverage_zoom = np.mean(miscoverages[zoom_start:])
        print(f"\n{model_name} Results:")
        print(f"Number of times outside intervals: {times_outside}")
        print(f"Final miscoverage rate: {final_miscoverage:.4f}")
        print(f"Final miscoverage rate (last {zoom_days} days): {final_miscoverage_zoom:.4f}")

        return prediction_intervals, miscoverages, final_miscoverage, final_miscoverage_zoom, fig
    
# Run the ML models for the given stock predicting the next day's volatilty

def forecast_tomorrow_volatility(rf_model, X, y, df, feature_columns, conformity_scores):
    latest_data = df.iloc[-max(window_sizes):]

    tomorrow_features = {}
    for window in window_sizes:
        if len(latest_data['returns']) >= window:
            vol = latest_data['returns'][-window:].std()
            tomorrow_features[f'volatility_{window}'] = vol
        else:
            raise ValueError(f"Not enough data to compute volatility_{window} for tomorrow's prediction.")

    tomorrow_features_df = pd.DataFrame([tomorrow_features])

    scaler = StandardScaler()
    scaler.fit(X)
    tomorrow_features_scaled = scaler.transform(tomorrow_features_df)

    tomorrow_forecast = rf_model.predict(tomorrow_features_scaled)[0]
    print(f"\nForecasted volatility for tomorrow: {tomorrow_forecast:.4f}%")

    threshold = alpha
    percentile_value = np.clip((1 - threshold) * 100, 0, 100)
    quantile = np.percentile(conformity_scores, percentile_value)
    lower_bound = max(0, tomorrow_forecast - quantile)
    upper_bound = tomorrow_forecast + quantile
    print(f"Prediction interval for tomorrow's volatility: [{lower_bound:.4f}%, {upper_bound:.4f}%]")

# Define the list of ML models to be used

models = {
    "RandomForest": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": lambda: xgb.XGBRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": lambda: GradientBoostingRegressor(n_estimators=100, random_state=42),
    "LightGBM": lambda: lgb.LGBMRegressor(n_estimators=100, random_state=42),
    "LinearRegression": lambda: LinearRegression(),
    "LSTM": lambda: create_lstm_model(input_shape=(1, len(feature_columns))),
    "GARCH": None
}

returns_series = df['returns']

all_results = {}
all_metrics = {}
all_figs = []

# Run the defined models

for mname, mfn in models.items():
    if mname == "GARCH":
        f_vol, r_vol, conf_scores, metrics, pred_dates = train_and_predict_garch(X, y, returns_series, cleaned_dates)
    else:
        f_vol, r_vol, conf_scores, metrics, pred_dates = train_and_predict_model(mname, mfn, X, y, cleaned_dates)

    intervals, miscoverages, final_mcov, final_mcov_zoom, fig = run_dtaci_and_plot(
        mname, f_vol, r_vol, conf_scores, zoom_days, pred_dates
    )
    
    all_results[mname] = {
        "forecasted_volatility": f_vol,
        "realized_volatility": r_vol,
        "conformity_scores": conf_scores,
        "intervals": intervals,
        "miscoverages": miscoverages,
        "final_mcov": final_mcov,
        "final_mcov_zoom": final_mcov_zoom,
        "pred_dates": pred_dates
    }
    all_metrics[mname] = metrics  # (mse, mae, r2)
    
    if isinstance(fig, list):
        all_figs.extend(fig)
    else:
        all_figs.append(fig)

# Generate Summary Plot

print("\nGenerating summary overlay plot of all models (last 100 days)...")
common_length = min(len(res["pred_dates"]) for res in all_results.values())
plot_length = min(zoom_days, common_length)

with plt.rc_context({'font.size': 30}):
    fig_summary = plt.figure(figsize=(15,10))
    some_model = list(all_results.keys())[0]
    pred_dates_ref = all_results[some_model]["pred_dates"]
    zoom_start_ref = len(pred_dates_ref) - plot_length

    # Plot realized volatility (baseline)
    realized_volatility_ref = all_results[some_model]["realized_volatility"]
    plt.plot(
        pred_dates_ref[zoom_start_ref:],
        realized_volatility_ref[zoom_start_ref:],
        label="Realized Volatility",
        color='black',
        linewidth=2
    )
    # Only plot forecast lines for selected models for report
    allowed_models = ["LinearRegression", "GARCH", "GradientBoosting", "XGBoost", "LightGBM"]
    colors = ['blue', 'red', 'green', 'purple', 'orange'] 
    for i, m in enumerate(allowed_models):
        if m in all_results:
            pred_dates_curr = all_results[m]["pred_dates"]
            f_vol = all_results[m]["forecasted_volatility"]
            zoom_start_curr = len(pred_dates_curr) - plot_length
            plt.plot(
                pred_dates_curr[zoom_start_curr:],
                f_vol[zoom_start_curr:],
                label=f"{m} Forecast",
                alpha=0.7,
                color=colors[i % len(colors)]
            )
    plt.xlabel("Date")
    plt.ylabel("Volatility (%)")
    leg_summary = plt.legend(ncol=2)
    for text in leg_summary.get_texts():
        text.set_fontsize(24)
    plt.tight_layout()
# --- END MODIFIED SUMMARY OVERLAY PLOT ---

all_figs.append(fig_summary)

# Metrics table
method_list = []
mse_list = []
mae_list = []
r2_list = []
avg_interval_size_list = []
achieved_coverage_list = []

for mname, res in all_results.items():
    f_vol = res["forecasted_volatility"]
    intervals = res["intervals"]
    miscoverages = res["miscoverages"]
    mse, mae, r2 = all_metrics[mname]

    interval_sizes = [ub - lb for (lb, ub) in intervals]
    avg_interval_size = np.mean(interval_sizes)
    coverage = 1 - (sum(miscoverages) / len(miscoverages))

    method_list.append(mname)
    mse_list.append(mse)
    mae_list.append(mae)
    r2_list.append(r2)
    avg_interval_size_list.append(avg_interval_size)
    achieved_coverage_list.append(coverage)

metrics_df = pd.DataFrame({
    "Model": method_list,
    "MSE": mse_list,
    "MAE": mae_list,
    "R2": r2_list,
    "Avg Interval Size": avg_interval_size_list,
    "Achieved Coverage": achieved_coverage_list
})
print("\nFinal Metrics Table:")
print(metrics_df)

for fig in all_figs:
    if isinstance(fig, list):
        for f in fig:
            f.show()
    else:
        fig.show()

plt.show(block=True)