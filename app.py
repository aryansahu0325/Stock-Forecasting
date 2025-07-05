import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Evaluation results
results = pd.DataFrame({
    'Model': ['ARIMA', 'SARIMA', 'Prophet', 'LSTM'],
    'RMSE': [69.3731, 55.7803, 38.3167, 10.6288],
    'MAE': [63.3772, 51.0295, 31.3954, 8.8552],
    'R²': [-5.0340, -2.9011, -0.8408, 0.8448]
})

# Predicted vs Actual for best model (LSTM)
# Replace with your actual arrays if needed
import numpy as np

# Simulated values for display purposes
actual = np.linspace(150, 180, 100)
predicted = actual + np.random.normal(0, 2, 100)

# Streamlit App
st.set_page_config(layout="wide")
st.title("📈 ZIDIO Stock Forecasting Dashboard")
st.markdown("Compare models and explore forecasts.")

# Model selection
selected_model = st.selectbox("🔍 Select a model to view details", results['Model'])

# Show metrics
st.subheader("📊 Model Evaluation Metrics")
selected_row = results[results['Model'] == selected_model]
st.write(selected_row.set_index('Model'))

# Plot actual vs predicted for selected model
st.subheader(f"📉 {selected_model} Forecast - Actual vs Predicted")

fig, ax = plt.subplots(figsize=(10, 5))
if selected_model == "LSTM":
    ax.plot(actual, label='Actual Price', color='blue')
    ax.plot(predicted, label='Predicted (LSTM)', color='purple')
else:
    ax.plot(actual, label='Actual Price', color='blue')
    ax.plot(actual + np.random.normal(0, 10, 100), label=f'Predicted ({selected_model})', linestyle='--')

ax.set_xlabel("Time")
ax.set_ylabel("Stock Price")
ax.set_title(f"{selected_model} - Actual vs Predicted")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Show full comparison table and best model
st.subheader("📌 All Model Comparison")

st.dataframe(results.set_index("Model").style.highlight_min(axis=0, subset=['RMSE', 'MAE'], color='lightgreen').highlight_max(axis=0, subset=['R²'], color='lightgreen'))

best_model = results.loc[results['RMSE'].idxmin(), 'Model']
st.success(f"✅ Best Model Based on RMSE: **{best_model}**")

# Optional: comparison graph
st.subheader("📊 Comparison Graphs")
tab1, tab2, tab3 = st.tabs(["RMSE", "MAE", "R² Score"])

with tab1:
    fig1, ax1 = plt.subplots()
    sns.barplot(data=results, x='Model', y='RMSE', ax=ax1, palette='coolwarm')
    for i, val in enumerate(results['RMSE']):
        ax1.text(i, val + 2, f"{val:.2f}", ha='center')
    ax1.set_title("RMSE Comparison")
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots()
    sns.barplot(data=results, x='Model', y='MAE', ax=ax2, palette='crest')
    for i, val in enumerate(results['MAE']):
        ax2.text(i, val + 2, f"{val:.2f}", ha='center')
    ax2.set_title("MAE Comparison")
    st.pyplot(fig2)

with tab3:
    fig3, ax3 = plt.subplots()
    sns.barplot(data=results, x='Model', y='R²', ax=ax3, palette='magma')
    for i, val in enumerate(results['R²']):
        ax3.text(i, val + 0.1, f"{val:.2f}", ha='center')
    ax3.set_title("R² Score Comparison")
    st.pyplot(fig3)

st.markdown("---")
st.caption("Developed for ZIDIO by Intern – Beginner Level Project 🚀")
