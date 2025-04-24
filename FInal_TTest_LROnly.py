###############################################################################
# Both CursorAI and OpenAI's GPT-o1 and GPT-4o models were used to improve the
# efficiency of the codes developed for this project. In this program one 
# notable use was the scenario structure which made the code more efficient
# and run far more quickly. 
###############################################################################

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import datetime
from arch import arch_model

plt.ioff() 

###############################################################################
# Parameters
###############################################################################
zoom_days = 100
alpha = 0.1
gamma_candidates = np.array([0.005, 0.01, 0.05, 0.1])
sigma = 0.05
eta = 0.01

start_date = "2010-01-01"
window_sizes = [5, 10, 15, 20, 25, 100, 252]

vix_ticker = "^VIX"
aero_ticker = "PPA"  # Aerospace & Defense ETF

# Download check
def safe_squeeze_to_series(series_or_df):
    if series_or_df.empty:
        raise ValueError("Downloaded data is empty. Check your date range or ticker.")
    if isinstance(series_or_df, pd.DataFrame):
        if series_or_df.shape[1] == 1:
            return series_or_df.squeeze()
        else:
            raise ValueError("Expected a single-column DataFrame but got multiple columns.")
    return series_or_df

# T-Test Function 
def evaluate_feature_importance(X, y):
    X_const = sm.add_constant(X)
    ols_model = sm.OLS(y, X_const).fit()
    # The summary2 table returns a DataFrame with the statistics.
    feat_imp_df = ols_model.summary2().tables[1]
    return feat_imp_df

# Only Linear Regression for this program
models_dict = {
    "LinearRegression": lambda: LinearRegression()
}

# Defining and Training LR model for calculation, T-Test and plotting 
def train_and_predict_model(model_name, model_fn, X, y, dates_index):
    
    forecasted_volatility = []
    realized_volatility = []
    conformity_scores = []
    pred_indices = []
    
    tscv = TimeSeriesSplit(n_splits=5)
    print(f"\nTraining {model_name} model...")
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = model_fn()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        print(f"Coefficients for {model_name}: {model.coef_}")
        print(f"Intercept for {model_name}: {model.intercept_}")
        
        forecasted_volatility.extend(y_pred)
        realized_volatility.extend(y_test)
        conformity_scores.extend(np.abs(y_pred - y_test))
        pred_indices.extend(test_index)
    
    forecasted_volatility = np.array(forecasted_volatility)
    realized_volatility = np.array(realized_volatility)
    conformity_scores = np.array(conformity_scores)
    
    min_len = min(len(forecasted_volatility), len(realized_volatility))
    forecasted_volatility = forecasted_volatility[:min_len]
    realized_volatility = realized_volatility[:min_len]
    conformity_scores = conformity_scores[:min_len]
    pred_indices = np.array(pred_indices)[:min_len]
    
    pred_dates = dates_index[pred_indices]

    # Eval Metrics
    
    mse = mean_squared_error(realized_volatility, forecasted_volatility)
    mae = mean_absolute_error(realized_volatility, forecasted_volatility)
    r2 = r2_score(realized_volatility, forecasted_volatility)
    
    return forecasted_volatility, realized_volatility, conformity_scores, (mse, mae, r2), pred_dates


# Main DtACI Algorithm
def run_dtaci_and_plot(
    model_name,
    forecasted_volatility,
    realized_volatility,
    conformity_scores,
    zoom_days,
    pred_dates,
    scenario_label="Scenario"
):
    print(f"\nRunning DtACI algorithm for {model_name} ({scenario_label})...")
    
    num_experts = len(gamma_candidates)
    weights = np.ones(num_experts)
    alpha_t = np.full(num_experts, alpha)
    
    miscoverages = []
    prediction_intervals = []
    times_outside = 0
    
    for t in range(len(forecasted_volatility)):
        vol = forecasted_volatility[t]
        true_vol = realized_volatility[t]
        if t > 0:
            conf_scores_t = np.abs(forecasted_volatility[:t] - realized_volatility[:t])
        else:
            conf_scores_t = np.array([])
        
        weights = np.clip(weights, 1e-10, None)
        probabilities = weights / np.sum(weights)
        expert_idx = np.random.choice(num_experts, p=probabilities)
        threshold = np.clip(alpha_t[expert_idx], 0, 1)
        
        if len(conf_scores_t) > 0:
            perc_val = np.clip((1 - threshold) * 100, 0, 100)
            quantile = np.percentile(conf_scores_t, perc_val)
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
        weights *= np.exp(-eta * losses)
        weights = np.clip(weights, 1e-10, None)
        weights = (1 - sigma) * weights + (sigma / num_experts) * np.sum(weights)
        alpha_t[expert_idx] += gamma_candidates[expert_idx] * (alpha - miscoverage)
        alpha_t = np.clip(alpha_t, 0, 1)
        miscoverages.append(miscoverage)

    # Generating 4-tile plots for DtACI intervals, showing intervals and miscoverage rate over the full time period and zoomend in 
    
    print(f"Generating plots for {model_name} ({scenario_label})...")
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))
    axs = axs.flatten()
    
    time_axis = pred_dates
    lower_bounds, upper_bounds = zip(*prediction_intervals)
    zoom_start = max(0, len(time_axis) - zoom_days)
    
    # Full view
    axs[0].plot(time_axis, forecasted_volatility, label="Forecasted Vol", alpha=0.7)
    axs[0].plot(time_axis, realized_volatility, label="Realized Vol", alpha=0.7)
    axs[0].fill_between(time_axis, lower_bounds, upper_bounds, color='gray', alpha=0.3, label="Prediction Intervals")
    axs[0].legend()
    axs[0].set_title(f"{model_name} - {scenario_label} (Full)")
    axs[0].set_xlabel("Date")
    axs[0].set_ylabel("Volatility (%)")
    
    # Zoomed view
    axs[1].plot(time_axis[zoom_start:], forecasted_volatility[zoom_start:], label="Forecasted", alpha=0.7)
    axs[1].plot(time_axis[zoom_start:], realized_volatility[zoom_start:], label="Realized", alpha=0.7)
    axs[1].fill_between(time_axis[zoom_start:], np.array(lower_bounds)[zoom_start:], np.array(upper_bounds)[zoom_start:], color='gray', alpha=0.3, label="Intervals")
    axs[1].legend()
    axs[1].set_title(f"{model_name} - {scenario_label} (Last {zoom_days} days)")
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel("Volatility (%)")
    
    # Miscoverage full view
    miscoverage_rates = np.cumsum(miscoverages) / np.arange(1, len(miscoverages) + 1)
    axs[2].plot(time_axis, miscoverage_rates, label="Empirical Miscoverage")
    axs[2].axhline(alpha, color="red", linestyle="--", label=f"Target = {alpha}")
    axs[2].legend()
    axs[2].set_title(f"{model_name} - Miscoverage (Full) - {scenario_label}")
    axs[2].set_xlabel("Date")
    axs[2].set_ylabel("Miscoverage Rate")
    
    # Miscoverage zoomed view
    axs[3].plot(time_axis[zoom_start:], miscoverage_rates[zoom_start:], label="Empirical Miscoverage")
    axs[3].axhline(alpha, color="red", linestyle="--", label=f"Target = {alpha}")
    axs[3].legend()
    axs[3].set_title(f"{model_name} - Miscoverage (Last {zoom_days}) - {scenario_label}")
    axs[3].set_xlabel("Date")
    axs[3].set_ylabel("Miscoverage Rate")
    
    plt.tight_layout()
    
    final_miscoverage = times_outside / len(miscoverages)
    final_miscoverage_zoom = np.mean(miscoverages[zoom_start:])
    
    print(f"\n{model_name} ({scenario_label}) Results:")
    print(f"Times outside intervals: {times_outside}")
    print(f"Final miscoverage rate: {final_miscoverage:.4f}")
    print(f"Final miscoverage rate (last {zoom_days} days): {final_miscoverage_zoom:.4f}")
    
    return prediction_intervals, miscoverages, final_miscoverage, final_miscoverage_zoom, fig

# Builds the specific scenario required, accounting for the VIX, chosen ETF and stock
def run_scenario(
    ticker,
    scenario_label,
    use_vix=False,
    use_aero=False,
    show_scenario_plots=True
):

    print(f"\n=== RUNNING SCENARIO: [{scenario_label}] for Ticker: {ticker} ===")
    
    # 1) Download main ticker
    data_main = yf.download(ticker, start=start_date, auto_adjust=False)
    returns_main = data_main['Adj Close'].pct_change() * 100
    returns_main = safe_squeeze_to_series(returns_main)
    if returns_main.empty:
        raise ValueError(f"{ticker} returns are empty for scenario: {scenario_label}")
    
    df = pd.DataFrame({'returns_main': returns_main})
    for w in window_sizes:
        df[f'main_vol_{w}'] = df['returns_main'].rolling(w).std()
    df['realized_volatility'] = df['returns_main'].rolling(5).std().shift(-1)
    
    # 2) Optionally add VIX data
    if use_vix:
        print(f"Downloading VIX data ({vix_ticker})...")
        data_vix = yf.download(vix_ticker, start=start_date)
        vix_close = data_vix['Close']
        vix_close = safe_squeeze_to_series(vix_close)
        vix_close = vix_close.reindex(df.index, method='ffill')
        df['vix'] = vix_close
    
    # 3) Optionally add Aerospace ETF data
    if use_aero:
        print(f"Downloading Aerospace & Defense ETF ({aero_ticker})...")
        data_ita = yf.download(aero_ticker, start=start_date)
        if 'Adj Close' in data_ita.columns:
            returns_ita = data_ita['Adj Close'].pct_change() * 100
        else:
            returns_ita = data_ita['Close'].pct_change() * 100
        returns_ita = safe_squeeze_to_series(returns_ita)
        returns_ita = returns_ita.reindex(df.index, method='ffill')
        df['returns_ita'] = returns_ita
        for w in window_sizes:
            df[f'ita_vol_{w}'] = df['returns_ita'].rolling(w).std()
    
    # 4) Build feature columns
    feature_cols = [f'main_vol_{w}' for w in window_sizes]
    if use_vix:
        feature_cols.append('vix')
    if use_aero:
        feature_cols += [f'ita_vol_{w}' for w in window_sizes]
    
    df.dropna(subset=feature_cols + ['realized_volatility'], inplace=True)
    if df.empty:
        raise ValueError(f"No data left after dropna for scenario: {scenario_label} (ticker: {ticker})")
    
    X = df[feature_cols]
    y = df['realized_volatility']
    scenario_dates = df.index
    
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    if len(X) < 6:
        raise ValueError(
            f"Scenario '{scenario_label}', Ticker '{ticker}': only {len(X)} points left. Not enough for 5-split TSCV."
        )
    
    # Use T-Test Function
    feat_imp_df = evaluate_feature_importance(X, y)
    print("\nFeature Importance for Linear Regression:")
    print(feat_imp_df)
    

    scenario_results = {}
    scenario_metrics = {}
    scenario_figs = []
    
    # Build model dict (only fro LinearRegression)
    scenario_models = {}
    for mname, builder in models_dict.items():
        scenario_models[mname] = builder
    
    for mname, fn in scenario_models.items():
        f_vol, r_vol, conf_scores, metrics, pred_dates = train_and_predict_model(
            mname, fn, X, y, scenario_dates
        )
        intervals, miscoverages, final_mcov, final_mcov_zoom, fig = run_dtaci_and_plot(
            mname,
            f_vol,
            r_vol,
            conf_scores,
            zoom_days,
            pred_dates,
            scenario_label=scenario_label
        )
        scenario_figs.append(fig)
        scenario_results[mname] = {
            "forecasted_volatility": f_vol,
            "realized_volatility": r_vol,
            "conformity_scores": conf_scores,
            "intervals": intervals,
            "miscoverages": miscoverages,
            "final_mcov": final_mcov,
            "final_mcov_zoom": final_mcov_zoom,
            "pred_dates": pred_dates
        }
        scenario_metrics[mname] = metrics
    
    # 6) Summary overlay (only one model exists)
    if show_scenario_plots:
        print(f"\nGenerating summary overlay (Scenario: {scenario_label}, Ticker: {ticker})...")
        common_len = len(list(scenario_results.values())[0]["pred_dates"])
        plot_len = min(zoom_days, common_len)
        
        fig_summary = plt.figure(figsize=(15, 8))
        first_model = list(scenario_results.keys())[0]
        pred_dates_ref = scenario_results[first_model]["pred_dates"]
        zoom_start_ref = len(pred_dates_ref) - plot_len
        
        plt.plot(
            pred_dates_ref[zoom_start_ref:], scenario_results[first_model]["realized_volatility"][zoom_start_ref:],
            label="Realized Volatility", color='black', linewidth=2
        )
        plt.plot(
            pred_dates_ref[zoom_start_ref:], scenario_results[first_model]["forecasted_volatility"][zoom_start_ref:],
            label=f"{first_model} Forecast", alpha=0.7, color='blue'
        )
        plt.title(f"Scenario: {scenario_label} - Forecast (Last {zoom_days} days) for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Volatility (%)")
        plt.legend()
        plt.tight_layout()
        scenario_figs.append(fig_summary)
    
    # 7) Summary of metrics in a DataFrame
    meth_list = []
    mse_list = []
    mae_list = []
    r2_list = []
    avg_int_list = []
    coverage_list = []
    
    for mname, res in scenario_results.items():
        f_vol = res["forecasted_volatility"]
        intervals = res["intervals"]
        miscover = res["miscoverages"]
        (mse, mae, r2) = scenario_metrics[mname]
        
        int_sizes = [ub - lb for (lb, ub) in intervals]
        avg_int = np.mean(int_sizes)
        coverage = 1 - (sum(miscover) / len(miscover))
        
        meth_list.append(mname)
        mse_list.append(mse)
        mae_list.append(mae)
        r2_list.append(r2)
        avg_int_list.append(avg_int)
        coverage_list.append(coverage)
    
    df_metrics = pd.DataFrame({
        "Model": meth_list,
        "MSE": mse_list,
        "MAE": mae_list,
        "R2": r2_list,
        "Avg Interval Size": avg_int_list,
        "Achieved Coverage": coverage_list
    })
    print(f"\nFinal Metrics Table (Scenario: {scenario_label}, Ticker: {ticker}):")
    print(df_metrics)
    
    if show_scenario_plots:
        for f in scenario_figs:
            f.show()
    
    return df_metrics, feat_imp_df

# Runs the built Scenatio for all of the stocks in TICKERS
if __name__ == "__main__":
    TICKERS = ["BA", "LMT", "GE", "RTX", "NOC"]  # Stocks Considered
    
    # Builds a Summary which includes the ETF, window sizes and parametrs used for refence when viewing results.
    df_summary = pd.DataFrame({
        "Parameter": [
            "Conformal alpha",
            "Gamma candidates",
            "sigma",
            "eta",
            "Aerospace ETF",
            "Window sizes",
            "Date/time"
        ],
        "Value": [
            alpha,
            gamma_candidates,
            sigma,
            eta,
            aero_ticker,
            window_sizes,
            str(datetime.datetime.now())
        ]
    })

    # Establishes an Excel file to save the results to.
    
    excel_name = "Final_LR_File.xlsx"
    with pd.ExcelWriter(excel_name) as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        
        # Runs the scenario for each ticker.
        for ticker in TICKERS:
            print(f"\n\n============== Processing Ticker: {ticker} ==============")
            
            # SCENARIO A: Windows Only
            df_a, feat_imp_a = run_scenario(
                ticker=ticker,
                scenario_label="Windows Only",
                use_vix=False,
                use_aero=False,
                show_scenario_plots=True
            )
            df_a["Scenario"] = "Windows Only"
            
            # SCENARIO B: Windows + VIX
            df_b, feat_imp_b = run_scenario(
                ticker=ticker,
                scenario_label="Windows + VIX",
                use_vix=True,
                use_aero=False,
                show_scenario_plots=True
            )
            df_b["Scenario"] = "Windows + VIX"
            
            # SCENARIO C: Windows + ITA
            df_c, feat_imp_c = run_scenario(
                ticker=ticker,
                scenario_label="Windows + ITA",
                use_vix=False,
                use_aero=True,
                show_scenario_plots=True
            )
            df_c["Scenario"] = "Windows + ITA"
            
            # SCENARIO D: Windows + VIX + ITA
            df_d, feat_imp_d = run_scenario(
                ticker=ticker,
                scenario_label="Windows + VIX + ITA",
                use_vix=True,
                use_aero=True,
                show_scenario_plots=True
            )
            df_d["Scenario"] = "Windows + VIX + ITA"
            
            combined_metrics = pd.concat([df_a, df_b, df_c, df_d], ignore_index=True)
            combined_metrics.to_excel(writer, sheet_name=ticker, index=False)
            
            # Write results for each scenario into it's own Excel sheet.
            feat_imp_a.to_excel(writer, sheet_name=f"{ticker}_WindowsOnly_Features")
            feat_imp_b.to_excel(writer, sheet_name=f"{ticker}_WindowsVIX_Features")
            feat_imp_c.to_excel(writer, sheet_name=f"{ticker}_WindowsITA_Features")
            feat_imp_d.to_excel(writer, sheet_name=f"{ticker}_WindowsVIXITA_Features")
    
    print(f"\nAll results (4 scenarios per stock) saved to '{excel_name}'!")
    print("Done. Showing all plots now...")
    plt.show(block=True)
