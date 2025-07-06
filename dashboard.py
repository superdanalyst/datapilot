# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 17:00:20 2025

@author: manthis
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgb

st.set_page_config(layout="wide")
st.title("Data Pilot")

# === Sidebar: Choose data source ===
st.sidebar.title("📁 Data Source")
data_source = st.sidebar.radio("Select Data Source", ["Built-in Dataset", "Upload Your Own"])

merged_df = None
uploaded_df = None

# === Built-in dataset setup ===
if data_source == "Built-in Dataset":
    # File paths
    stocks_file = "AllStocks2.xlsx"
    trade_file = "ImportsExports.xlsx"

    # Load imports/exports
    imports_df = pd.read_excel(trade_file, sheet_name="Quarterly Imports")
    exports_df = pd.read_excel(trade_file, sheet_name="Quarterly Exports")
    imports_monthly_df = pd.read_excel(trade_file, sheet_name="Monthly Imports")
    exports_monthly_df = pd.read_excel(trade_file, sheet_name="Monthly Exports")

    imports_monthly_df["Date"] = pd.to_datetime(imports_monthly_df["Date"])
    exports_monthly_df["Period"] = pd.to_datetime(exports_monthly_df["Date"])

    imports_df["Date"] = imports_df.index if "Date" not in imports_df.columns else imports_df["Date"].astype(str)
    exports_df["Date"] = exports_df.index if "Date" not in exports_df.columns else exports_df["Date"].astype(str)

    import_options = [col for col in imports_df.columns if col != "Date"]
    export_options = [col for col in exports_df.columns if col != "Date"]

    # Ticker selection
    xls = pd.ExcelFile(stocks_file)
    tickers = xls.sheet_names
    ticker_selected = st.selectbox("Select a Ticker", tickers)
    import_selected = st.multiselect("Select Import Categories", import_options)
    export_selected = st.multiselect("Select Export Categories", export_options)

    # Load and merge data
    stock_df = pd.read_excel(stocks_file, sheet_name=ticker_selected)
    stock_df["Date"] = stock_df["Date"].astype(str)
    merged_df = stock_df.copy()

    if import_selected:
        selected_imports = imports_df[["Date"] + import_selected]
        merged_df = merged_df.merge(selected_imports, on="Date", how="left")

    if export_selected:
        selected_exports = exports_df[["Date"] + export_selected]
        merged_df = merged_df.merge(selected_exports, on="Date", how="left")

# === Uploaded dataset setup ===
else:
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        file_ext = uploaded_file.name.split(".")[-1].lower()

        try:
            if file_ext == "csv":
                uploaded_df = pd.read_csv(uploaded_file)
                merged_df = uploaded_df.copy()
                st.success("✅ CSV file uploaded successfully.")

            elif file_ext == "xlsx":
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names

                selected_sheet = st.selectbox("Select sheet to load", sheet_names)

                if selected_sheet:
                    uploaded_df = excel_file.parse(selected_sheet)
                    if uploaded_df.empty:
                        st.error("Selected sheet is empty.")
                    else:
                        uploaded_df.dropna(how="all", axis=1, inplace=True)
                        merged_df = uploaded_df.copy()
                        st.success(f"✅ Excel file loaded from sheet: `{selected_sheet}`")

        except Exception as e:
            st.error(f"Failed to read file: {e}")


# === Show table + YoY & Lag logic ===
if merged_df is not None:
    # Ensure Date exists and is datetime if possible
    if "Date" in merged_df.columns:
        try:
            merged_df["Date_dt"] = pd.to_datetime(merged_df["Date"], errors="coerce")
            merged_df = merged_df.sort_values("Date_dt")
        except Exception:
            st.warning("⚠️ Date column could not be parsed.")

    # Create a two-column layout
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("Combined Data")
        st.dataframe(merged_df)

    with right_col:
        st.subheader("Add Year-over-Year (YoY) % Change Columns")

        # YoY on numerics
        yoy_candidates = merged_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        yoy_selected = st.multiselect("Select numeric columns to compute YoY % Change", yoy_candidates)

        if yoy_selected:
            for col in yoy_selected:
                new_col = f"{col} YoY % Change"
                merged_df[new_col] = merged_df[col].pct_change(periods=4) * 100
                merged_df[new_col] = merged_df[new_col].round(2)
            st.success("YoY % change columns added.")

        st.markdown("---")
        st.subheader("Create Lagged Variables")
        lag_candidates = merged_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        lag_selected = st.multiselect("Select numeric columns to create lag features", lag_candidates)
        lag_amount = st.slider("Number of quarters to lag by", 1, 4, 1)

        if lag_selected:
            for col in lag_selected:
                lag_col = f"Lag_{lag_amount}_{col}"
                merged_df[lag_col] = merged_df[col].shift(lag_amount).round(2)
            st.success(f"Lagged columns (by {lag_amount} quarters) added.")

else:
    st.warning("Please upload or select a dataset to continue.")
    st.stop()


# ========= Integrated EDA Starts Here =========
# Tab Navigation (no sidebar)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Final Dataframe", 
    "🔎 Summary & Visuals", 
    "📈 Correlation & OLS", 
    "⚡ XGBoost", 
    "🔁 Seasonality",
    "📅 General Forecasting"

])

# Common: Numeric columns
numeric_columns = merged_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# === Tab 1: Combined Data ===
with tab1:
    st.dataframe(merged_df, use_container_width=True)

# === Tab 2: Summary & Visuals ===
with tab2:
    summary_option = st.radio(
        "Choose a summary option:",
        ["Data Dimensions", "Field Descriptions", "Summary Statistics"],
        horizontal=True
    )

    if summary_option == 'Field Descriptions':
        fd = merged_df.dtypes.reset_index().rename(columns={'index': 'Field Name', 0: 'Field Type'})
        st.dataframe(fd, use_container_width=True, hide_index=True)
    elif summary_option == 'Summary Statistics':
        desc = merged_df.describe(include='all').round(2)
        nulls = merged_df.isnull().sum().rename("Missing Values")
        st.dataframe(pd.concat([nulls, desc], axis=1), use_container_width=True)
    else:
        st.write(f"Shape of data: {merged_df.shape}")

    st.divider()
    
# === Tab 3: Correlation & OLS ===
with tab3:
    if len(numeric_columns) >= 2:
        # Step 1 – Dropdowns to select Y-axis left and right
        left_y = st.selectbox("Select Left Y-Axis Variable", numeric_columns, index=0, key="left_y")
        right_y = st.selectbox("Select Right Y-Axis Variable", numeric_columns, index=1, key="right_y")

        # Step 2 – Chart (left) and correlation (right)
        st.markdown("### Dual-Axis Chart and Correlation")
        chart_col, corr_col = st.columns([2, 1])

        with chart_col:
            valid_df = merged_df[[left_y, right_y]].replace([np.inf, -np.inf], np.nan).dropna()
            if valid_df.empty:
                st.warning("Not enough data to plot after removing NaNs/Infs.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=valid_df.index, y=valid_df[left_y], name=left_y, yaxis="y1"))
                fig.add_trace(go.Scatter(x=valid_df.index, y=valid_df[right_y], name=right_y, yaxis="y2"))
                
                fig.update_layout(
                    xaxis=dict(title="Index"),
                    yaxis=dict(title=left_y),
                    yaxis2=dict(title=right_y, overlaying='y', side='right'),
                    legend=dict(x=0, y=1),
                    margin=dict(t=30, b=30)
                )
                st.plotly_chart(fig, use_container_width=True)


        with corr_col:
            st.markdown("### Correlation")
            corr_val = merged_df[left_y].corr(merged_df[right_y])
            st.metric(f"Corr({left_y}, {right_y})", f"{corr_val:.4f}")

        # Step 3 – OLS Regression Section
        st.markdown("---")
        st.subheader("OLS Regression")

        # Step 4 – Select Y and X variables + Train/Test Split toggle (on the left), Chart (on the right)
        selectors_col, chart_col = st.columns([1, 2])
        
        with selectors_col:
            y_var = st.selectbox("Select Dependent (Y) Variable", numeric_columns, key="ols_y")
            x_vars = st.multiselect("Select Independent Variable(s)", [col for col in numeric_columns if col != y_var], key="ols_x")
            split_index = st.slider(
                "Train/Test Split (for Regression)", 
                1, 
                len(merged_df) - 1, 
                int(len(merged_df) * 0.8), 
                key="ols_split"
            )
        
        with chart_col:
            if x_vars:
                vis_df = merged_df[[y_var] + [x_vars[0]]].replace([np.inf, -np.inf], np.nan).dropna()
                if vis_df.empty:
                    st.warning("Insufficient data to visualize.")
                else:
                    fig = go.Figure()
        
                    # Add primary y-axis trace
                    fig.add_trace(go.Scatter(
                        x=vis_df.index,
                        y=vis_df[y_var],
                        name=y_var,
                        yaxis="y1",
                        line=dict(color="blue")
                    ))
        
                    # Add secondary y-axis trace
                    fig.add_trace(go.Scatter(
                        x=vis_df.index,
                        y=vis_df[x_vars[0]],
                        name=x_vars[0],
                        yaxis="y2",
                        line=dict(color="pink")
                    ))
        
                    # Add vertical line for train/test split
                    fig.add_vline(
                        x=vis_df.index[split_index],
                        line=dict(color="white", dash="dash"),
                        annotation_text="Train/Test Split",
                        annotation_position="top left"
                    )
        
                    fig.update_layout(
                        xaxis=dict(title="Index"),
                        yaxis=dict(title=y_var),
                        yaxis2=dict(
                            title=x_vars[0],
                            overlaying="y",
                            side="right"
                        ),
                        legend=dict(x=0.01, y=0.99),
                        margin=dict(t=40, b=40)
                    )
        
                    st.plotly_chart(fig, use_container_width=True)
    

    # ========== Full-Width Analysis Output ========== #
    if x_vars:
        reg_df = merged_df[[y_var] + x_vars].replace([np.inf, -np.inf], np.nan).dropna()
        if reg_df.empty:
            st.warning("Insufficient data after cleaning.")
        else:
            train_data = reg_df.iloc[:split_index]
            test_data = reg_df.iloc[split_index:]
    
            if train_data.empty or test_data.empty:
                st.error("Train or test data is empty after the split. Adjust the split point.")
            else:
                # --- Correlation Output ---
                st.markdown("### Correlation with Dependent Variable")
                corr_metrics = [(x, reg_df[y_var].corr(reg_df[x])) for x in x_vars]
                for var, val in corr_metrics:
                    st.metric(f"{var}", f"{val:.4f}")
    
                # --- VIF Check ---
                if all(abs(corr) >= 0.3 for _, corr in corr_metrics):
                    X_train_vif = sm.add_constant(train_data[x_vars])
                    vif_data = pd.DataFrame({
                        "Variable": ["const"] + x_vars,
                        "VIF": [variance_inflation_factor(X_train_vif.values, i) for i in range(X_train_vif.shape[1])]
                    })
                    st.markdown("### Variance Inflation Factor (VIF)")
                    st.dataframe(vif_data)
    
                    if vif_data["VIF"][1:].max() > 5:
                        st.error("High multicollinearity detected (VIF > 5).")
                    else:
                        try:
                            # --- Fit and Predict ---
                            y_train = train_data[y_var]
                            X_train = sm.add_constant(train_data[x_vars])
                            model = sm.OLS(y_train, X_train).fit()
    
                            X_test = sm.add_constant(test_data[x_vars])
                            y_test = test_data[y_var]
                            y_pred = model.predict(X_test)
    
                            # --- Regression Metrics ---
                            st.markdown("### OLS Regression Results")
                            st.metric("R² (Test)", f"{r2_score(y_test, y_pred):.4f}")
                            st.metric("RMSE", f"{mean_squared_error(y_test, y_pred, squared=False):.4f}")
    
                            # --- Model Summary ---
                            with st.expander("Show OLS Summary"):
                                st.text(model.summary())
                                
                            # --- User Prediction Input ---
                            ########### New Addition ###############
                            st.markdown("### Predict Upcoming Quarter (e.g., Q2 2025)")
                            
                            st.info("Enter estimated values for the independent variables to forecast the next quarter.")
                            
                            future_input = {}
                            for var in x_vars:
                                val = st.number_input(
                                    f"Estimated value for `{var}`", 
                                    value=float(reg_df[var].mean()), 
                                    key=f"future_{var}"
                                )
                                future_input[var] = val
                            
                            if st.button("Predict Next Quarter Value"):
                                try:
                                    future_df = pd.DataFrame([future_input])
                                    # Ensure same columns and order as training set
                                    future_df = future_df[x_vars]
                                    X_future = sm.add_constant(future_df, has_constant='add')  # Matches training
                            
                                    future_prediction = model.predict(X_future)[0]
                                    st.success(f"🔮 Predicted `{y_var}` for next quarter: **{future_prediction:.2f}**")
                                except Exception as e:
                                    st.error(f"Prediction failed: {e}")

    
    
                        except Exception as e:
                            st.error(f"OLS regression failed: {e}")
                else:
                    st.error("One or more X variables have weak correlation (|corr| < 0.3) with Y.")
    else:
        st.info("Please select at least one independent variable.")
    


with tab4:
    st.subheader("XGBoost Regression")

    y_xgb = st.selectbox("Select Dependent Variable", numeric_columns, key="xgb_y")
    x_xgb = st.multiselect("Select Independent Variables", [col for col in numeric_columns if col != y_xgb], key="xgb_x")

    if x_xgb:
        # Train/Test split slider
        split_idx = st.slider("Train/Test Split", 1, len(merged_df)-1, int(len(merged_df) * 0.8), key="xgb_split")

        # Parameter selection
        st.markdown("#### XGBoost Hyperparameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.number_input("n_estimators", value=100, min_value=10, max_value=1000, step=10)
        with col2:
            learning_rate = st.number_input("learning_rate", value=0.1, min_value=0.001, max_value=1.0, step=0.01, format="%.3f")
        with col3:
            max_depth = st.number_input("max_depth", value=3, min_value=1, max_value=10, step=1)

        # Data prep
        model_df = merged_df[[y_xgb] + x_xgb].replace([np.inf, -np.inf], np.nan).dropna()
        if len(model_df) < split_idx + 1:
            st.warning("Not enough data after cleaning and split.")
        else:
            train = model_df.iloc[:split_idx]
            test = model_df.iloc[split_idx:]

            X_train = train[x_xgb]
            y_train = train[y_xgb]
            X_test = test[x_xgb]
            y_test = test[y_xgb]

            # Model training
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                objective='reg:squarederror'
            )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # Evaluation
            st.markdown("### Model Performance")
            st.metric("R² (Test)", f"{r2_score(y_test, preds):.4f}")
            st.metric("RMSE", f"{mean_squared_error(y_test, preds, squared=False):.4f}")

            # Forecasting input
            st.markdown("### Forecast Next Quarter")

            future_vals = {}
            for var in x_xgb:
                future_vals[var] = st.number_input(f"Estimated value for `{var}`", value=float(model_df[var].mean()), key=f"xgb_input_{var}")
            if st.button("Predict with XGBoost"):
                try:
                    input_df = pd.DataFrame([future_vals])
                    forecast = model.predict(input_df)[0]
                    st.success(f"🔮 Predicted `{y_xgb}`: **{forecast:.2f}**")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
    else:
        st.info("Please select at least one independent variable.")

# === Tab 5: Monthly Seasonality & Forecasting ===
with tab5:
    st.subheader("Monthly Seasonality Forecast & Quarterly Estimation")

    selected_col = None  # <-- Add this line to prevent NameError

    if data_source == "Upload Your Own":
        st.info("📁 Monthly forecasting is only available when using the built-in dataset.")
    else:
        monthly_type = st.radio("Select Monthly Data Type", ["Imports", "Exports"], horizontal=True)
        monthly_df = imports_monthly_df if monthly_type == "Imports" else exports_monthly_df
        quarterly_df = imports_df if monthly_type == "Imports" else exports_df

        monthly_cols = [col for col in monthly_df.columns if col != "Date"]
        selected_col = st.selectbox("Select Monthly Variable", monthly_cols)

        forecast_method = st.selectbox("Select Forecasting Method", ["SARIMAX", "Holt-Winters (Seasonal)"])

        # (keep rest of tab4 forecasting logic unchanged below this point)

    if selected_col:
        data = monthly_df[["Date", selected_col]].dropna().copy()
        data = data.sort_values("Date")
        data.set_index("Date", inplace=True)
        ts = data[selected_col]

        freq = 12  # monthly seasonality

        try:
            result = seasonal_decompose(ts, model='additive', period=freq)

            # Plot decomposition
            # Create 4 vertically stacked subplots
            fig = make_subplots(
                rows=4, cols=1, shared_xaxes=True,
                subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
                vertical_spacing=0.05
            )
            
            fig.add_trace(go.Scatter(x=ts.index, y=result.observed, name="Observed"), row=1, col=1)
            fig.add_trace(go.Scatter(x=ts.index, y=result.trend, name="Trend"), row=2, col=1)
            fig.add_trace(go.Scatter(x=ts.index, y=result.seasonal, name="Seasonal"), row=3, col=1)
            fig.add_trace(go.Scatter(x=ts.index, y=result.resid, name="Residual", line=dict(color="purple")), row=4, col=1)
            
            fig.update_layout(
                height=900,
                showlegend=False,
                margin=dict(t=40, b=40),
                title_text="Seasonal Decomposition"
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Forecast next 3 months using selected method
            forecast_index = pd.date_range(ts.index[-1] + pd.DateOffset(months=1), periods=3, freq='MS')

            if forecast_method == "SARIMAX":
                model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, freq))
                fit_model = model.fit(disp=False)
                forecast_values = fit_model.get_forecast(steps=3).predicted_mean.round(2)

            elif forecast_method == "Holt-Winters (Seasonal)":
                model = ExponentialSmoothing(ts,  trend='add', seasonal='add', seasonal_periods=freq)
                fit_model = model.fit()
                forecast_values = fit_model.forecast(3).round(2)

            forecast_df = pd.DataFrame({
                "Date": forecast_index,
                "Forecast": forecast_values
            }).set_index("Date")

            st.markdown("### 3-Month Forecast")
            st.dataframe(forecast_df)

            # Determine current quarter
            last_date = ts.index.max()
            year = last_date.year
            month = last_date.month
            if month in [1, 2, 3]:
                cq = f"Q1 {year}"
                quarter_months = [1, 2, 3]
            elif month in [4, 5, 6]:
                cq = f"Q2 {year}"
                quarter_months = [4, 5, 6]
            elif month in [7, 8, 9]:
                cq = f"Q3 {year}"
                quarter_months = [7, 8, 9]
            else:
                cq = f"Q4 {year}"
                quarter_months = [10, 11, 12]


            # Gather available actuals + forecast
            quarter_dates = [pd.Timestamp(year, m, 1) for m in quarter_months]
            actuals = ts.reindex(quarter_dates).dropna()
            needed = [d for d in quarter_dates if d not in actuals.index]
            forecast_needed = forecast_df.reindex(needed).dropna()

            est_quarterly_total = actuals.sum() + forecast_needed["Forecast"].sum()
            st.metric(f"Estimated Total for {cq}", f"{est_quarterly_total:.2f}")

            # Combine with historical quarterly actuals
            quarterly_col = selected_col if selected_col in quarterly_df.columns else None
            if quarterly_col:
                q_actuals = quarterly_df[["Date", quarterly_col]].dropna()
                
                # Only try to convert Date if it's not already quarterly string like 'Q1 2020'
                try:
                    q_actuals["Date"] = pd.to_datetime(q_actuals["Date"])
                    q_actuals["Quarter"] = q_actuals["Date"].dt.to_Date("Q").astype(str)
                except Exception:
                    q_actuals["Quarter"] = q_actuals["Date"]  # Assume already quarterly strings
            
                q_actuals = q_actuals.groupby("Quarter")[quarterly_col].sum().reset_index()
            else:
                q_actuals = pd.DataFrame(columns=["Quarter", "Total"])


            # Append forecasted quarter
            forecast_row = pd.DataFrame({"Quarter": [cq], quarterly_col: [est_quarterly_total]})
            plot_df = pd.concat([q_actuals, forecast_row])

            # Bar plot
            # Create two columns side by side
            bar_col, data_col = st.columns([2, 1])  # Wider chart, smaller data
            
            with bar_col:
                st.markdown("### Quarterly Totals with Forecast")
            
                # Plotly Bar Chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=plot_df["Quarter"],
                    y=plot_df[quarterly_col],
                    name="Quarterly Total",
                    marker=dict(color='rgba(100, 149, 237, 0.8)')
                ))
            
                fig.update_layout(
                    xaxis_title="Quarter",
                    yaxis_title="Total Value",
                    title=f"{selected_col} – Quarterly Actuals and Forecasted",
                    xaxis_tickangle=-90,
                    margin=dict(t=50, b=50),
                    height=400
                )
            
                st.plotly_chart(fig, use_container_width=True)
            
            with data_col:
                st.markdown("### Forecast & Estimate Summary")
                st.dataframe(plot_df.set_index("Quarter"), use_container_width=True)



        except Exception as e:
            st.error(f"Seasonal decomposition or forecasting failed: {e}")


with tab6:
    st.subheader("General Time Series Forecasting")

    if merged_df is None or merged_df.empty:
        st.warning("No data available.")
        st.stop()

    # Select date column
    potential_dates = [col for col in merged_df.columns if "date" in col.lower()]
    date_col = st.selectbox("Select the Date Column", potential_dates)

    # Select value column to forecast
    value_col = st.selectbox("Select Column to Forecast", [col for col in merged_df.columns if col != date_col and merged_df[col].dtype != "object"])

    # Frequency
    freq = st.radio("Select Time Frequency", ["Monthly", "Quarterly"], horizontal=True)
    periods_ahead = st.slider("Forecast Periods Ahead", 1, 12 if freq == "Monthly" else 8, 3)

    # Model choice
    algo = st.selectbox("Forecasting Method", ["SARIMAX", "Holt-Winters (Seasonal)"])

    if date_col and value_col:
        try:
            ts_df = merged_df[[date_col, value_col]].dropna().copy()
            ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors='coerce')
            ts_df = ts_df.dropna(subset=[date_col])
            ts_df = ts_df.sort_values(date_col).set_index(date_col)

            # Infer frequency if needed
            ts = ts_df[value_col]
            if freq == "Monthly":
                ts = ts.resample("MS").mean()
                seasonal_periods = 12
            else:
                ts = ts.resample("QS").mean()
                seasonal_periods = 4

            # Forecasting
            if algo == "SARIMAX":
                model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_periods))
                fit_model = model.fit(disp=False)
                forecast = fit_model.get_forecast(steps=periods_ahead).predicted_mean
            else:
                model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
                fit_model = model.fit()
                forecast = fit_model.forecast(periods_ahead)

            forecast_df = pd.DataFrame({f"Forecast_{value_col}": forecast})
            st.write(forecast_df)

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts.index, y=ts, name="Historical"))
            fig.add_trace(go.Scatter(x=forecast.index, y=forecast, name="Forecast"))
            fig.update_layout(title=f"Forecast for {value_col}", xaxis_title="Date", yaxis_title=value_col)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Forecasting failed: {e}")
