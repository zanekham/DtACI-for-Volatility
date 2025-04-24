###############################################################################
# Both CursorAI and OpenAI's GPT-o1 and GPT-4o models were used to improve the
# efficiency of the codes developed for this project. In this program one 
# notable use was improving how the data is stored after each cumulative window
# iteration, allowing the program to run for efficiently.
###############################################################################

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.ioff() 

###############################################################################
# Parameters
###############################################################################
ticker = "BA"
start_date = "2010-01-01"

alpha = 0.1
gamma_candidates = np.array([0.005, 0.01, 0.05, 0.1])
sigma = 0.05
eta = 0.01

all_window_sizes = np.arange(5, 200, 5)  # 5, 10, 15, ..., 250

# Number of repetitions for averaging results
num_reps = 20

# Download Data and engineer features
print(f"Downloading {ticker} data from {start_date}...")
data = yf.download(ticker, start=start_date, auto_adjust=False)

if 'Adj Close' not in data.columns:
    raise ValueError("Missing 'Adj Close' column in downloaded data.")

returns = data['Adj Close'].pct_change() * 100
returns.dropna(inplace=True)

if isinstance(returns, pd.Series):
    df_original = returns.to_frame(name='returns')
elif isinstance(returns, pd.DataFrame):
    if returns.shape[1] == 1:
        df_original = returns.copy()
        df_original.columns = ['returns']
    else:
        raise ValueError("Expected a single column of returns, but got multiple columns.")

# Define LR Model with 5 fold cross validation
def train_and_predict_linear(X, y):
   
    forecasted_volatility = []
    realized_volatility = []
    conformity_scores = []
    
    tscv = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()
    model = LinearRegression()
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Scaling features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model and predict
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        forecasted_volatility.extend(y_pred)
        realized_volatility.extend(y_test)
        conformity_scores.extend(np.abs(y_pred - y_test))
    
    forecasted_volatility = np.array(forecasted_volatility)
    realized_volatility = np.array(realized_volatility)
    conformity_scores = np.array(conformity_scores)
    
    # Compute metrics
    mse = mean_squared_error(realized_volatility, forecasted_volatility)
    mae = mean_absolute_error(realized_volatility, forecasted_volatility)
    r2 = r2_score(realized_volatility, forecasted_volatility)
    
    return forecasted_volatility, realized_volatility, conformity_scores, (mse, mae, r2)

# Defines DtACI Algorithm 
def run_dtaci(forecasted_volatility, realized_volatility, conformity_scores):
    
    num_experts = len(gamma_candidates)
    weights = np.ones(num_experts)
    alpha_t = np.full(num_experts, alpha)
    
    prediction_intervals = []
    
    for t in range(len(forecasted_volatility)):
        vol = forecasted_volatility[t]
        true_vol = realized_volatility[t]
        
        # Use past conformity scores up to time t-1
        if t > 0:
            conf_scores_t = np.abs(forecasted_volatility[:t] - realized_volatility[:t])
        else:
            conf_scores_t = np.array([])
        
        weights = np.clip(weights, 1e-10, None)
        probabilities = weights / np.sum(weights)
        expert_idx = np.random.choice(num_experts, p=probabilities)
        threshold = np.clip(alpha_t[expert_idx], 0, 1)
        
        # Calculate quantile from past conformity scores
        if len(conf_scores_t) > 0:
            quantile = np.percentile(conf_scores_t, (1 - threshold) * 100)
        else:
            quantile = vol * 0.1
        
        lower_bound = max(0, vol - quantile)
        upper_bound = vol + quantile
        prediction_intervals.append((lower_bound, upper_bound))
        
        # Determine miscoverage (1 if true volatility is outside the interval, else 0)
        miscoverage = 1 if (true_vol < lower_bound or true_vol > upper_bound) else 0
        
        # Update the expert’s alpha using gamma_candidates and the miscoverage indicator
        alpha_t[expert_idx] += gamma_candidates[expert_idx] * (alpha - miscoverage)
        alpha_t = np.clip(alpha_t, 0, 1)
    
    avg_interval_size = np.mean([ub - lb for lb, ub in prediction_intervals])
    return avg_interval_size

# Store Results Over Repetitions

num_experiments = len(all_window_sizes)
avg_interval_all = np.zeros((num_reps, num_experiments))
comp_time_all = np.zeros((num_reps, num_experiments))
max_window_values = all_window_sizes.copy()

# Run the program within each loop

for rep in range(num_reps):
    # Current Iteration
    print(f"\n=== Repetition {rep+1} of {num_reps} ===")
    for i in range(1, num_experiments + 1):
        current_windows = list(all_window_sizes[:i])
        df = df_original.copy()
        
        for w in current_windows:
            df[f'vol_{w}'] = df['returns'].rolling(window=w).std()
        
        df['realized_volatility'] = df['returns'].rolling(5).std().shift(-1)
        required_cols = [f'vol_{w}' for w in current_windows] + ['realized_volatility']
        df.dropna(subset=required_cols, inplace=True)
        
        feature_cols = [f'vol_{w}' for w in current_windows]
        X = df[feature_cols]
        y = df['realized_volatility']
        
        # Measure time for training and DtACI interval calculation
        start_time = time.time()
        f_vol, r_vol, conf_scores, metrics = train_and_predict_linear(X, y)
        avg_interval = run_dtaci(f_vol, r_vol, conf_scores)
        elapsed_time = time.time() - start_time
        
        # Save the results for this window set
        avg_interval_all[rep, i-1] = avg_interval
        comp_time_all[rep, i-1] = elapsed_time

# Compute averages across repetitions
mean_avg_intervals = avg_interval_all.mean(axis=0)
mean_comp_times = comp_time_all.mean(axis=0)

#Plot Results
plt.figure(figsize=(12, 4))
plt.plot(max_window_values, mean_avg_intervals, marker='o', linestyle='-')
plt.xlabel("Maximum Window Size in Feature Set", fontsize=29)
plt.ylabel("Average DtACI Interval Size", fontsize=29)
plt.xticks(fontsize=29)
plt.yticks(fontsize=29)
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(max_window_values, mean_comp_times, marker='o', linestyle='-', color='red')
plt.xlabel("Maximum Window Size in Feature Set", fontsize=29)
plt.ylabel("Computation Time (seconds)", fontsize=29)
plt.xticks(fontsize=29)
plt.yticks(fontsize=29)
plt.grid(True)
plt.show()
