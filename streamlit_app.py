import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Multi-Stock ML Prediction Dashboard", layout="wide")
st.title("📈 Multi-Stock ML Prediction Dashboard")

# Caching style config
plt.rcParams['figure.dpi'] = 120
sns.set_style('whitegrid')

@st.cache_data
def load_data():
    df = pd.read_csv('Multi_Stock_SMA_Dashboard_Dataset.csv', skiprows=1)
    df.columns = ['Date','AAPL','MSFT','GOOGL','AMZN','TSLA','META','NVDA','NFLX','JPM','JNJ','XOM','WMT']
    df = df.dropna()
    for col in df.columns[1:]:
        df[col] = df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip().astype(float)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

df = load_data()
stocks = ['AAPL','MSFT','GOOGL','AMZN','TSLA','META','NVDA','NFLX','JPM','JNJ','XOM','WMT']

def make_features(series, lookback=10):
    X, y = [], []
    for i in range(lookback, len(series)):
        window = series[i - lookback:i]
        feat = list(window)
        feat.append(np.mean(window))
        feat.append(np.mean(series[max(0, i-20):i]))
        feat.append(np.mean(series[max(0, i-50):i]))
        feat.append(window[-1] - window[0])
        feat.append(window[-1] / window[0] - 1 if window[0] != 0 else 0)
        X.append(feat)
        y.append(series[i])
    return np.array(X), np.array(y)

@st.cache_resource
def train_models(_df, stocks, lookback=10):
    results = {}
    for stock in stocks:
        series = _df[stock].values
        X, y = make_features(series, lookback=lookback)
        
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        window = list(series[-lookback:])
        all_vals = series.tolist()
        future_preds = []
        for _ in range(5):
            feat = window[-lookback:]
            hist = all_vals + future_preds
            f = list(feat)
            f.append(np.mean(feat))
            f.append(np.mean(hist[-20:]))
            f.append(np.mean(hist[-50:]))
            f.append(feat[-1] - feat[0])
            f.append((feat[-1] / feat[0] - 1) if feat[0] != 0 else 0)
            nxt = model.predict([f])[0]
            future_preds.append(nxt)
            window.append(nxt)
            
        current_price = series[-1]
        pred_5day = future_preds[-1]
        change_pct = (pred_5day - current_price) / current_price * 100
        signal = 'BUY' if change_pct > 1 else ('SELL' if change_pct < -1 else 'HOLD')
        
        results[stock] = {
            'model': model, 'y_test': y_test, 'y_pred': y_pred,
            'current': current_price, 'next5': future_preds,
            'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape,
            'change_pct': change_pct, 'signal': signal
        }
    return results

with st.spinner("Training models for all stocks (this is cached so it will be fast after the first run)..."):
    results = train_models(df, stocks)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview & Data", "Model Evaluation Portfolio", "Predictions & Forecasting"])

if page == "Overview & Data":
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
        st.write(f"**Date Range:** {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    with st.expander("Show Raw Data"):
        st.dataframe(df)
        
    st.subheader("Price Trends (2024)")
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab20.colors
    for i, stock in enumerate(stocks):
        ax.plot(df['Date'], df[stock], label=stock, linewidth=1.2, color=colors[i])
    ax.set_ylabel("Price")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)
    
    st.subheader("Normalized Performance (Base=100)")
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    for i, stock in enumerate(stocks):
        normalized = (df[stock] / df[stock].iloc[0]) * 100
        ax2.plot(df['Date'], normalized, label=stock, linewidth=1.2, color=colors[i])
    ax2.axhline(100, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig2)

    st.subheader("Correlation Matrix")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    corr = df[stocks].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax3, mask=mask, vmin=-1, vmax=1)
    st.pyplot(fig3)

elif page == "Model Evaluation Portfolio":
    st.header("Evaluation Metrics")
    
    summary = []
    for s in stocks:
        summary.append({
            'Stock': s,
            'MAE': results[s]['mae'],
            'RMSE': results[s]['rmse'],
            'R²': results[s]['r2'],
            'MAPE (%)': results[s]['mape']
        })
    metrics_df = pd.DataFrame(summary).set_index('Stock')
    st.dataframe(metrics_df.style.highlight_max(subset=['R²'], color='lightgreen').highlight_min(subset=['MAPE (%)', 'MAE', 'RMSE'], color='lightgreen'))
    
    st.subheader("Metrics Comparison")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    mapes = [results[s]['mape'] for s in stocks]
    r2s = [results[s]['r2'] for s in stocks]
    maes = [results[s]['mae'] for s in stocks]
    bar_colors = ['#3498db' for _ in stocks]
    
    axes[0].bar(stocks, mapes, color=bar_colors)
    axes[0].set_title('MAPE (%)')
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(stocks, r2s, color=bar_colors)
    axes[1].axhline(0, color='gray', linestyle='--')
    axes[1].set_title('R² Score')
    axes[1].tick_params(axis='x', rotation=45)
    
    axes[2].bar(stocks, maes, color=bar_colors)
    axes[2].set_title('MAE ($)')
    axes[2].tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    
    st.subheader("Actual vs Predicted & Residuals")
    selected_stock = st.selectbox("Select a stock for detailed view:", stocks)
    
    r = results[selected_stock]
    
    col1, col2 = st.columns(2)
    with col1:
        fig_actual, ax_actual = plt.subplots(figsize=(8, 4))
        # ensure idx is a list of sequential integers of correct length
        idx = range(len(r['y_test']))
        ax_actual.plot(idx, r['y_test'], label='Actual', color='#2980b9')
        ax_actual.plot(idx, r['y_pred'], label='Predicted', color='#e74c3c', linestyle='--')
        ax_actual.set_title(f"{selected_stock} Actual vs Predicted")
        ax_actual.legend()
        st.pyplot(fig_actual)
        
    with col2:
        fig_res, ax_res = plt.subplots(figsize=(8, 4))
        residuals = r['y_test'] - r['y_pred']
        ax_res.scatter(r['y_pred'], residuals, alpha=0.5, color='#9b59b6')
        ax_res.axhline(0, color='red', linestyle='--')
        ax_res.set_title(f"{selected_stock} Residuals")
        st.pyplot(fig_res)

elif page == "Predictions & Forecasting":
    st.header("5-Day Machine Learning Forecast")
    
    forecast_data = []
    for s in stocks:
        forecast_data.append({
            'Stock': s,
            'Current Price ($)': results[s]['current'],
            'Day 1 ($)': results[s]['next5'][0],
            'Day 3 ($)': results[s]['next5'][2],
            'Day 5 ($)': results[s]['next5'][4],
            '5d Change (%)': results[s]['change_pct'],
            'Signal': results[s]['signal']
        })
    f_df = pd.DataFrame(forecast_data).set_index('Stock')
    
    def color_signal(val):
        color = 'green' if val == 'BUY' else 'red' if val == 'SELL' else 'orange'
        return f'color: {color}; font-weight: bold'
        
    st.dataframe(f_df.style.map(color_signal, subset=['Signal']).format(precision=2))
    
    st.subheader("5-Day Predicted Change (%)")
    fig, ax = plt.subplots(figsize=(10, 5))
    changes = [results[s]['change_pct'] for s in stocks]
    bar_colors = ['#27ae60' if c > 1 else '#e74c3c' if c < -1 else '#f39c12' for c in changes]
    
    bars = ax.bar(stocks, changes, color=bar_colors)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(1, color='#27ae60', linestyle='--', label='Buy threshold (+1%)')
    ax.axhline(-1, color='#e74c3c', linestyle='--', label='Sell threshold (-1%)')
    ax.legend()
    st.pyplot(fig)
    
    st.subheader("Individual Stock Forecast Viewer")
    pred_stock = st.selectbox("Select a stock:", stocks)
    r = results[pred_stock]
    
    fig_pred, ax_pred = plt.subplots(figsize=(10, 4))
    recent_actual = df[pred_stock].values[-30:] # Last 30 days
    
    # Plot recent actuals
    ax_pred.plot(range(len(recent_actual)), recent_actual, label='Recent Actual', marker='o', color='#2980b9')
    
    # Plot forecasts
    fcast_idx = range(len(recent_actual)-1, len(recent_actual) + 5)
    fcast_y = [recent_actual[-1]] + r['next5']
    ax_pred.plot(fcast_idx, fcast_y, label='5d Forecast', marker='o', color='#27ae60', linestyle=':')
    ax_pred.set_title(f"{pred_stock} 5-Day Forecast (Signal: {r['signal']})")
    ax_pred.legend()
    st.pyplot(fig_pred)
