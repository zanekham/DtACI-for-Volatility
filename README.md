# 📈 Volatility Forecasting with Machine Learning and DtACI

This repository can be used to recreate the experiments conducted in **"Forecasted Aerospace Volatility with DtACI Intervals"** by **Zane Khambatta**

This project presents a hybrid framework for forecasting realized volatility of a financial time series (Aerospace stock) using a combination of:

- Machine Learning models (Random Forest, XGBoost, LightGBM, LSTM, etc.)
- Traditional econometric models (GARCH)

The uncertainity surrounding thes epredictions are then quantified with Conformal Prediction via **DtACI** to generate valid prediction intervals.

The system is designed to evaluate models using **time-series-aware cross-validation**, and dynamically adjust confidence intervals using the **Dynamically Tuned Adaptive Conformal Inference (DtACI)** algorithm.

---

## 🔍 Overview

### Features:
- Download and preprocess stock return data from Yahoo Finance
- Generate volatility-based features
- Predict next-day volatility using:
  - Random Forest
  - XGBoost
  - Gradient Boosting
  - LightGBM
  - Linear Regression
  - LSTM (Keras)
  - GARCH(1,1) (ARCH package)
- Apply **DtACI** to build adaptive prediction intervals around forecasts
- Visualize:
  - Forecasts vs. Realized Volatility
  - Prediction intervals
  - Empirical miscoverage rates
- Summarise performance across all models with RMSE, MAE, R², coverage
- Explore the effects of using volatility proxies and industry specific information to enhance forcasting
- Evaluate Feature importance with a statistical T-test
---

## 📁 Repository Structure:
- **Final_InitialTest_AllModels:** Code for implementing DtACI on the forecasted volatilty of a specified Aerospace Stock using multiple ML and Economtric models.
- **FInal_TTest_LROnly:** Code for a further investigation into the effects of using the VIX and an Aerospace ETF to enhance Linear Regression forecasting, along with evaluation of feature importance with a T-Test.
- **Final_Window_Finder:** Code for an averaged iterative test which determines the most accurate and efficient combination of rolling volatility windows to forecast volatility.
