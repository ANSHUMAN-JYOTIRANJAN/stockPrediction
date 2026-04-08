# # Install required libraries if not already present
# !pip install scikit-learn pandas numpy matplotlib seaborn --quiet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style('whitegrid')

print('✅ All libraries imported successfully')

# Load dataset - update path if needed
df = pd.read_csv('Multi_Stock_SMA_Dashboard_Dataset.csv', skiprows=1)
df.columns = ['Date','AAPL','MSFT','GOOGL','AMZN','TSLA','META','NVDA','NFLX','JPM','JNJ','XOM','WMT']
df = df.dropna()

# Remove $ signs and commas, convert to float
for col in df.columns[1:]:
    df[col] = df[col].str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip().astype(float)

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f'Dataset shape: {df.shape}')
print(f'Date range: {df["Date"].min().date()} → {df["Date"].max().date()}')
df.head()

# Summary statistics
df.describe().round(2)

stocks = ['AAPL','MSFT','GOOGL','AMZN','TSLA','META','NVDA','NFLX','JPM','JNJ','XOM','WMT']

fig, axes = plt.subplots(4, 3, figsize=(16, 12))
axes = axes.flatten()
colors = plt.cm.tab20.colors

for i, stock in enumerate(stocks):
    ax = axes[i]
    ax.plot(df['Date'], df[stock], color=colors[i], linewidth=1.2)
    ax.set_title(stock, fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

plt.suptitle('2024 Daily Closing Prices — All Stocks', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()

# Normalized price performance (base = 100)
fig, ax = plt.subplots(figsize=(14, 6))
for i, stock in enumerate(stocks):
    normalized = (df[stock] / df[stock].iloc[0]) * 100
    ax.plot(df['Date'], normalized, label=stock, linewidth=1.4, color=colors[i])

ax.axhline(100, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax.set_title('Normalized Price Performance (Jan 2 = 100)', fontsize=13, fontweight='bold')
ax.set_ylabel('Indexed Price')
ax.legend(loc='upper left', fontsize=8, ncol=2)
plt.tight_layout()
plt.show()

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[stocks].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax,
            mask=mask, linewidths=0.5, vmin=-1, vmax=1,
            annot_kws={'size': 9})
ax.set_title('Stock Price Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

def make_features(series, lookback=10):
    """
    Build feature matrix from price series.
    Features:
      - Last 10 closing prices (lookback window)
      - SMA-10, SMA-20, SMA-50
      - 10-day momentum (absolute & percent)
    """
    X, y = [], []
    for i in range(lookback, len(series)):
        window = series[i - lookback:i]
        feat = list(window)
        feat.append(np.mean(window))                          # SMA-10
        feat.append(np.mean(series[max(0, i-20):i]))          # SMA-20
        feat.append(np.mean(series[max(0, i-50):i]))          # SMA-50
        feat.append(window[-1] - window[0])                   # Momentum (abs)
        feat.append(window[-1] / window[0] - 1)               # Momentum (%)
        X.append(feat)
        y.append(series[i])
    return np.array(X), np.array(y)

print('✅ Feature engineering function defined')
print('Features per sample: 15 (10 lag prices + SMA10 + SMA20 + SMA50 + momentum_abs + momentum_pct)')

LOOKBACK = 10
results = {}

for stock in stocks:
    series = df[stock].values
    X, y = make_features(series, lookback=LOOKBACK)

    # 80/20 train-test split (time-ordered)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train Gradient Boosting model
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # 5-day future forecast (iterative rollout)
    window = list(series[-LOOKBACK:])
    all_vals = series.tolist()
    future_preds = []
    for _ in range(5):
        feat = window[-LOOKBACK:]
        hist = all_vals + future_preds
        f = list(feat)
        f.append(np.mean(feat))
        f.append(np.mean(hist[-20:]))
        f.append(np.mean(hist[-50:]))
        f.append(feat[-1] - feat[0])
        f.append(feat[-1] / feat[0] - 1)
        nxt = model.predict([f])[0]
        future_preds.append(nxt)
        window.append(nxt)

    current_price = series[-1]
    pred_5day     = future_preds[-1]
    change_pct    = (pred_5day - current_price) / current_price * 100
    signal        = 'BUY' if change_pct > 1 else ('SELL' if change_pct < -1 else 'HOLD')

    results[stock] = {
        'model': model,
        'y_test': y_test,
        'y_pred': y_pred,
        'current': current_price,
        'next5': future_preds,
        'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape,
        'change_pct': change_pct, 'signal': signal
    }
    print(f'{stock:5s} | MAE={mae:6.2f} | RMSE={rmse:6.2f} | R²={r2:+.3f} | MAPE={mape:5.2f}% | Signal: {signal}')

print('\n✅ All models trained successfully')

# Build summary dataframe
summary = pd.DataFrame([
    {
        'Stock': s,
        'Current Price ($)': round(results[s]['current'], 2),
        'MAE': round(results[s]['mae'], 2),
        'RMSE': round(results[s]['rmse'], 2),
        'R²': round(results[s]['r2'], 3),
        'MAPE (%)': round(results[s]['mape'], 2),
        '5d Forecast ($)': round(results[s]['next5'][-1], 2),
        'Change (%)': round(results[s]['change_pct'], 2),
        'Signal': results[s]['signal']
    } for s in stocks
])

def color_signal(val):
    if val == 'BUY':  return 'background-color: #d4edda; color: #155724; font-weight: bold'
    if val == 'SELL': return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
    return 'background-color: #fff3cd; color: #856404; font-weight: bold'

def color_change(val):
    if val > 0:  return 'color: #155724; font-weight: bold'
    if val < 0:  return 'color: #721c24; font-weight: bold'
    return ''

summary.style \
    .applymap(color_signal, subset=['Signal']) \
    .applymap(color_change, subset=['Change (%)']) \
    .format({'Current Price ($)': '${:.2f}', '5d Forecast ($)': '${:.2f}', 'Change (%)': '{:+.2f}%'})

# Metrics comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

mapes  = [results[s]['mape'] for s in stocks]
r2s    = [results[s]['r2']   for s in stocks]
maes   = [results[s]['mae']  for s in stocks]

bar_colors = ['#2ecc71' if results[s]['signal'] == 'BUY'
               else '#e74c3c' if results[s]['signal'] == 'SELL'
               else '#f39c12' for s in stocks]

axes[0].bar(stocks, mapes, color=bar_colors, edgecolor='white', linewidth=0.5)
axes[0].set_title('MAPE (%) — lower is better', fontweight='bold')
axes[0].set_ylabel('MAPE (%)')
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(stocks, r2s, color=bar_colors, edgecolor='white', linewidth=0.5)
axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
axes[1].set_title('R² Score — higher is better', fontweight='bold')
axes[1].set_ylabel('R²')
axes[1].tick_params(axis='x', rotation=45)

axes[2].bar(stocks, maes, color=bar_colors, edgecolor='white', linewidth=0.5)
axes[2].set_title('MAE ($) — lower is better', fontweight='bold')
axes[2].set_ylabel('MAE ($)')
axes[2].tick_params(axis='x', rotation=45)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='BUY signal'),
    Patch(facecolor='#e74c3c', label='SELL signal'),
    Patch(facecolor='#f39c12', label='HOLD signal')
]
fig.legend(handles=legend_elements, loc='upper right', fontsize=9)
plt.suptitle('Model Performance Metrics by Stock', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(4, 3, figsize=(18, 14))
axes = axes.flatten()

for i, stock in enumerate(stocks):
    r   = results[stock]
    ax  = axes[i]
    idx = range(len(r['y_test']))

    ax.plot(idx, r['y_test'], label='Actual',    color='#2980b9', linewidth=1.4)
    ax.plot(idx, r['y_pred'], label='Predicted', color='#e74c3c', linewidth=1.2, linestyle='--')

    # Forecast extension
    fcast_idx = range(len(r['y_test']) - 1, len(r['y_test']) + 5)
    fcast_y   = [r['y_test'][-1]] + r['next5']
    ax.plot(fcast_idx, fcast_y, color='#27ae60', linewidth=1.6,
            linestyle=':', marker='o', markersize=3, label='5d forecast')

    signal_color = '#27ae60' if r['signal'] == 'BUY' else '#e74c3c' if r['signal'] == 'SELL' else '#f39c12'
    ax.set_title(f"{stock}  [{r['signal']}]  MAPE={r['mape']:.1f}%",
                 fontsize=10, fontweight='bold', color=signal_color)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.tick_params(labelsize=8)
    if i == 0:
        ax.legend(fontsize=7, loc='upper left')

plt.suptitle('Actual vs Predicted Closing Prices (Test Set + 5-Day Forecast)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# 5-day forecast table
forecast_df = pd.DataFrame(index=stocks)
forecast_df['Current ($)'] = [round(results[s]['current'], 2) for s in stocks]

for d in range(1, 6):
    forecast_df[f'Day {d} ($)'] = [round(results[s]['next5'][d-1], 2) for s in stocks]

forecast_df['5d Change (%)'] = [round(results[s]['change_pct'], 2) for s in stocks]
forecast_df['Signal'] = [results[s]['signal'] for s in stocks]

def style_row(row):
    if row['Signal'] == 'BUY':  return ['background-color: #d4edda'] * len(row)
    if row['Signal'] == 'SELL': return ['background-color: #f8d7da'] * len(row)
    return ['background-color: #fff3cd'] * len(row)

forecast_df.style.apply(style_row, axis=1)

# 5-day forecast bar chart
changes = [results[s]['change_pct'] for s in stocks]
bar_colors = ['#27ae60' if c > 1 else '#e74c3c' if c < -1 else '#f39c12' for c in changes]

fig, ax = plt.subplots(figsize=(13, 5))
bars = ax.bar(stocks, changes, color=bar_colors, edgecolor='white', linewidth=0.5, zorder=3)
ax.axhline(0, color='black', linewidth=0.8)
ax.axhline(1,  color='#27ae60', linestyle='--', linewidth=0.7, alpha=0.6, label='Buy threshold (+1%)')
ax.axhline(-1, color='#e74c3c', linestyle='--', linewidth=0.7, alpha=0.6, label='Sell threshold (-1%)')

for bar, val in zip(bars, changes):
    ypos = bar.get_height() + 0.2 if val >= 0 else bar.get_height() - 0.7
    ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:+.1f}%',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_title('5-Day Predicted Price Change by Stock', fontsize=13, fontweight='bold')
ax.set_ylabel('Predicted Change (%)')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.4, zorder=0)
plt.tight_layout()
plt.show()

feature_names = ([f'Lag_{i+1}' for i in range(LOOKBACK)] +
                 ['SMA_10', 'SMA_20', 'SMA_50', 'Momentum_Abs', 'Momentum_Pct'])

top_stocks = ['META', 'NVDA', 'JPM', 'WMT']  # best R² models

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for i, stock in enumerate(top_stocks):
    imp = results[stock]['model'].feature_importances_
    sorted_idx = np.argsort(imp)[::-1]
    axes[i].barh([feature_names[j] for j in sorted_idx[:10]],
                 [imp[j] for j in sorted_idx[:10]],
                 color='#3498db', edgecolor='white')
    axes[i].invert_yaxis()
    axes[i].set_title(f'{stock} Feature Importance', fontsize=10, fontweight='bold')
    axes[i].set_xlabel('Importance')

plt.suptitle('Top 10 Feature Importances (Best-Fit Models)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(4, 3, figsize=(16, 13))
axes = axes.flatten()

for i, stock in enumerate(stocks):
    r = results[stock]
    residuals = r['y_test'] - r['y_pred']
    ax = axes[i]
    ax.scatter(r['y_pred'], residuals, alpha=0.5, s=18,
               color='#9b59b6', edgecolors='white', linewidths=0.3)
    ax.axhline(0, color='red', linewidth=1, linestyle='--')
    ax.set_title(f'{stock}  (RMSE=${r["rmse"]:.2f})', fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted Price', fontsize=8)
    ax.set_ylabel('Residual', fontsize=8)
    ax.tick_params(labelsize=8)

plt.suptitle('Residual Plots: Predicted Price vs Prediction Error', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

buy_stocks  = [s for s in stocks if results[s]['signal'] == 'BUY']
sell_stocks = [s for s in stocks if results[s]['signal'] == 'SELL']
hold_stocks = [s for s in stocks if results[s]['signal'] == 'HOLD']

print('=' * 55)
print('      5-DAY ML PREDICTION SIGNAL SUMMARY')
print('=' * 55)
print(f'\n🟢 BUY  ({len(buy_stocks)} stocks):')
for s in buy_stocks:
    print(f'   {s:5s}  Current: ${results[s]["current"]:>8.2f}  →  Forecast: ${results[s]["next5"][-1]:>8.2f}  ({results[s]["change_pct"]:+.2f}%)')

print(f'\n🔴 SELL ({len(sell_stocks)} stocks):')
for s in sell_stocks:
    print(f'   {s:5s}  Current: ${results[s]["current"]:>8.2f}  →  Forecast: ${results[s]["next5"][-1]:>8.2f}  ({results[s]["change_pct"]:+.2f}%)')

if hold_stocks:
    print(f'\n🟡 HOLD ({len(hold_stocks)} stocks):')
    for s in hold_stocks:
        print(f'   {s:5s}  Current: ${results[s]["current"]:>8.2f}  →  Forecast: ${results[s]["next5"][-1]:>8.2f}  ({results[s]["change_pct"]:+.2f}%)')

print('\n' + '=' * 55)
best_r2  = max(stocks, key=lambda s: results[s]['r2'])
worst_r2 = min(stocks, key=lambda s: results[s]['r2'])
print(f'Best  model fit : {best_r2}  (R² = {results[best_r2]["r2"]:.3f})')
print(f'Worst model fit : {worst_r2}  (R² = {results[worst_r2]["r2"]:.3f})')
print('=' * 55)
print('\n⚠️  Disclaimer: ML predictions for educational use only.')
print('   Not financial advice.')