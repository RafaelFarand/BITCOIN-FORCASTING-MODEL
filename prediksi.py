# prediksi.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import os
import json
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# BERT & Transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# Data
import yfinance as yf

# Set page config
st.set_page_config(
    page_title="Bitcoin Price Prediction Inference",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #F7931A;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #F7931A;
    }
    .best-config {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# LSTM MODEL DEFINITION
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)  # Output nya cuma Close price aja
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# FUNGSI HELPER
@st.cache_resource
def load_bert_model(model_name="ElKulako/cryptobert"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        pipe = TextClassificationPipeline(
            model=bert_model, 
            tokenizer=tokenizer, 
            top_k=1,
            truncation=True,
            max_length=512
        )
        return pipe
    except Exception as e:
        st.error(f"Error loading BERT model: {e}")
        return None

def analyze_sentiment(df_berita, pipe):
    if 'sentiment' in df_berita.columns and 'sentiment_score' in df_berita.columns:
        st.info("✅ Sentiment columns already exist. Skipping BERT processing.")
        label_map = {'Bullish': 1, 'Bearish': -1, 'Neutral': 0}
        if 'sentiment_numeric' not in df_berita.columns:
            df_berita['sentiment_numeric'] = df_berita['sentiment'].map(label_map)
        return df_berita
    
    sentiments = []
    scores = []
    
    progress_bar = st.progress(0)
    for idx, text in enumerate(df_berita['Text'].fillna("")):
        if text.strip() == "":
            sentiments.append("Neutral")
            scores.append(0.0)
        else:
            result = pipe(str(text))
            pred = result[0][0]
            sentiments.append(pred['label'])
            scores.append(pred['score'])
        progress_bar.progress((idx + 1) / len(df_berita))
    
    df_berita['sentiment'] = sentiments
    df_berita['sentiment_score'] = scores
    
    label_map = {'Bullish': 1, 'Bearish': -1, 'Neutral': 0}
    df_berita['sentiment_numeric'] = df_berita['sentiment'].map(label_map)
    
    return df_berita

@st.cache_data
def download_bitcoin_data(start_date, end_date):
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    end_dt_buffer = end_dt + timedelta(days=2)
    
    df_price = yf.download("BTC-USD", 
                          start=start_dt - timedelta(days=1),
                          end=end_dt_buffer,
                          progress=False)
    
    if isinstance(df_price.columns, pd.MultiIndex):
        df_price.columns = df_price.columns.droplevel(1)
    
    df_price.dropna(inplace=True)
    df_price.reset_index(inplace=True)
    
    if 'Date' not in df_price.columns and 'Datetime' in df_price.columns:
        df_price.rename(columns={'Datetime': 'Date'}, inplace=True)
    
    df_price = df_price[['Date', 'Close']]
    df_price['Date'] = pd.to_datetime(df_price['Date']).dt.normalize()
    
    mask = (df_price['Date'] >= start_dt) & (df_price['Date'] <= end_dt)
    df_price = df_price[mask]
    
    return df_price

def create_sequences_for_lstm(data, seq_length, num_features):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length), :]  # data input All features
        y = data[i + seq_length, 0]  # outputnya Only Close
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def inverse_transform_predictions(scaler, predictions, num_features):
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    temp = np.zeros((len(predictions), num_features))
    temp[:, 0] = predictions.squeeze()  # Only Close
    temp[:, 1] = 0  # Use 0 for sentiment (no dummy inverse transform)
    original_scale = scaler.inverse_transform(temp)[:, 0]
    return original_scale

def main():
    st.markdown('<h1 class="main-header">₿ Bitcoin Price Prediction Inference</h1>', unsafe_allow_html=True)
    st.markdown("### Load Model + Evaluation + Prediction (Focus on Close Price)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    st.sidebar.header("⚙️ Configuration")
    uploaded_file = st.sidebar.file_uploader("Upload Test News CSV", type=['csv'])
    st.sidebar.info("File must have 'Text' and 'Date' columns for test/prediction.")
    
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start Date Test", value=pd.to_datetime("2025-01-01"))
    end_date = col2.date_input("End Date Test", value=pd.to_datetime("2025-12-31"))
    
    inference_button = st.sidebar.button("Mulai Prediksi", type="primary")
    
    # Load model first
    models_dir = 'models'
    model_path = os.path.join(models_dir, 'best_model_latest.pth')
    scaler_path = os.path.join(models_dir, 'scaler_latest.pkl')
    config_path = os.path.join(models_dir, 'config_latest.json')
    
    if not all(os.path.exists(p) for p in [model_path, scaler_path, config_path]):
        st.error("❌ Model files not found! Run training.py first.")
        return
    
    with open(config_path, 'r') as f:
        best_config = json.load(f)
    
    scaler = joblib.load(scaler_path)
    num_features = 2  # jumlah fitur nya ada 2 Close, sentiment_mean
    
    model = LSTM(input_size=num_features, 
                 hidden_size=best_config['hidden_size'], 
                 num_layers=best_config['num_layers']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    st.success("✅ Model loaded successfully!")
    st.markdown('<div class="best-config">', unsafe_allow_html=True)
    st.subheader("🏆 Best Configuration Loaded")
    st.json(best_config)
    st.markdown('</div>', unsafe_allow_html=True)
    
    tab1, tab3, tab4 = st.tabs(["📊 Data", "📈 Evaluation", "🔮 Prediction"])
    
    if inference_button:
        if uploaded_file is None:
            st.error("❌ Please upload the test news CSV file first!")
            return
        
        # Process test data in Tab Data
        with tab1:
            st.header("📊 Data Preparation")
            
            st.subheader("1. Sentiment Analysis")
            with st.spinner("Loading BERT model..."):
                pipe = load_bert_model()
            
            df_berita = pd.read_csv(uploaded_file)
            st.write(f"Total news: {len(df_berita)}")
            
            with st.spinner("Analyzing sentiment (or skipping if exists)..."):
                df_berita = analyze_sentiment(df_berita, pipe)
            
            st.success("✅ Sentiment processing complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Sentiment Distribution:")
                st.bar_chart(df_berita['sentiment'].value_counts())
            with col2:
                st.write("Sample Data:")
                st.dataframe(df_berita[['Date', 'Text', 'sentiment', 'sentiment_score']])
        
            st.subheader("2. Download Bitcoin Price Data")
            with st.spinner("Downloading Bitcoin data..."):
                df_price = download_bitcoin_data(start_date, end_date)
            
            if len(df_price) == 0:
                st.error("❌ No Bitcoin data found for this date range!")
                return
            
            st.success(f"✅ Data downloaded: {len(df_price)} days")
            
            st.line_chart(df_price.set_index('Date')['Close'])
            
            # Merge with sentiment
            if 'Date' in df_berita.columns:
                df_berita['Date'] = pd.to_datetime(df_berita['Date'])
                df_sentiment_daily = df_berita.groupby('Date').agg({'sentiment_numeric': 'mean'}).reset_index()
                df_sentiment_daily.columns = ['Date', 'sentiment_mean']
                
                df_combined = pd.merge(df_price, df_sentiment_daily, on='Date', how='left')
                df_combined['sentiment_mean'].fillna(0, inplace=True)
                
                features = ['Close', 'sentiment_mean']
                st.success("✅ Sentiment data merged!")
            else:
                st.error("❌ News data missing 'Date' column!")
                return
            
            df_model = df_combined[features + ['Date']].dropna()
            st.subheader("Sample Fusion Data (Close Price + Sentiment)")
            st.dataframe(df_combined)
            st.write(f"Total features: {len(features)}")
            st.write(f"Data ready: {len(df_model)} days")
            
            # Store in session state for use in other tabs
            st.session_state['df_model'] = df_model
            st.session_state['features'] = features
            st.session_state['df_berita'] = df_berita
            st.session_state['df_price'] = df_price
            st.session_state['df_combined'] = df_combined
            st.session_state['scaler'] = scaler
        
        with tab3:
            st.header("📈 Model Evaluation")
            
            if 'df_model' not in st.session_state:
                st.error("Please process data in Data tab first!")
                return
            
            df_model = st.session_state['df_model']
            features = st.session_state['features']
            
            df_values = df_model[features].copy()
            scaled_data = scaler.transform(df_values)
            
            dates_all = df_model['Date'].copy()
            dates_for_y = dates_all.iloc[best_config['window_size']:].reset_index(drop=True)
            
            window_size = best_config['window_size']
            
            # Create sequences for evaluation (treat all as test in inference)
            X, y = create_sequences_for_lstm(scaled_data, window_size, num_features)
            
            X_test = X
            y_test = y
            
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
            
            dates_test = dates_for_y
            
            with torch.no_grad():
                y_pred_test = model(X_test_tensor).cpu().numpy().squeeze()
            
            y_test_np = y_test_tensor.cpu().numpy()
            
            y_test_usd = inverse_transform_predictions(scaler, y_test_np, num_features)
            y_pred_test_usd = inverse_transform_predictions(scaler, y_pred_test, num_features)
            
            mse_test = mean_squared_error(y_test_usd, y_pred_test_usd)
            rmse_test = np.sqrt(mse_test)
            mae_test = mean_absolute_error(y_test_usd, y_pred_test_usd)
            mape_test = np.mean(np.abs((y_test_usd - y_pred_test_usd) / (y_test_usd + 1e-8))) * 100
            rmse_percentage = (rmse_test / np.mean(y_test_usd)) * 100  # RMSE as percentage of mean actual value
            
            st.subheader("📊 Metrics (USD) for Close Price")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RMSE", f"${rmse_test:,.2f}")
            col2.metric("MAE", f"${mae_test:,.2f}")
            col3.metric("MAPE", f"{mape_test:.2f}%")
            col4.metric("RMSE %", f"{rmse_percentage:.2f}%")
            
            st.subheader("📊 Actual vs Predicted (Normalized Scale) for Close Price")
            fig_test, ax_test = plt.subplots(figsize=(18, 8))
            ax_test.axvspan(dates_test.iloc[0], dates_test.iloc[-1], alpha=0.15, color='red', label='Test Period')
            ax_test.plot(dates_test, y_test_np, label='Actual', color='#0066CC', linewidth=2.5)
            ax_test.plot(dates_test, y_pred_test, label='Predicted', color='#9933FF', linewidth=2.5, linestyle='--', marker='o', markersize=3)
            ax_test.set_title('Bitcoin Close Price: Actual vs Predictions (Normalized Scale)', fontsize=16, fontweight='bold')
            ax_test.set_xlabel('Date', fontsize=13)
            ax_test.set_ylabel('Normalized Price (0-1)', fontsize=13)
            ax_test.legend(fontsize=11)
            ax_test.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_test)
            
            col1, col2 = st.columns(2)
            
            with col1:
                residuals = y_test_usd - y_pred_test_usd
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(residuals, bins=50, color='#F18F01', alpha=0.7, edgecolor='black')
                ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
                ax.set_title('Prediction Error Distribution (USD)', fontsize=14, fontweight='bold')
                ax.set_xlabel('Error (USD)', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(y_test_usd, y_pred_test_usd, alpha=0.5, color='#6A4C93')
                ax.plot([y_test_usd.min(), y_test_usd.max()], 
                        [y_test_usd.min(), y_test_usd.max()], 
                        'r--', linewidth=2)
                ax.set_title('Actual vs Predicted (USD)', fontsize=14, fontweight='bold')
                ax.set_xlabel('Actual Close Price (USD)', fontsize=12)
                ax.set_ylabel('Predicted Close Price (USD)', fontsize=12)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        with tab4:
            st.header("🔮 Future Prediction")
            
            window_size = best_config['window_size']
            
            last_sequence = torch.tensor(scaled_data[-window_size:], dtype=torch.float32).to(device)
            
            avg_sentiment = np.mean(scaled_data[-window_size:, 1])  # Index 1 = sentiment_mean (normalized)
            
            st.info(f"📈 Using last {window_size} days data with average sentiment: {avg_sentiment:.4f} (normalized)")

            prediction_days = 1
            
            with st.spinner(f"Predicting {prediction_days} day ahead..."):
                with torch.no_grad():
                    seq_input = last_sequence.unsqueeze(0)
                    pred_scaled = model(seq_input).cpu().numpy().squeeze()
            
            # Convert predictions to numpy
            predictions_normalized = np.array([pred_scaled])
            
            # Inverse transform to USD
            predictions_usd = inverse_transform_predictions(scaler, predictions_normalized, num_features)
            
            # Start prediction from end_date + 1
            original_end_date = end_date
            prediction_start_date = pd.to_datetime(original_end_date) + timedelta(days=1)
            
            # Generate future dates
            future_dates = pd.date_range(start=prediction_start_date, periods=prediction_days, freq='D')
            
            # Display info
            st.info(f"📅 Prediction starts from: {prediction_start_date.date()} (end_date + 1 day)")
            st.info(f"📅 Prediction range: {future_dates[0].date()} to {future_dates[-1].date()}")
            
            # Create DataFrame
            prediction_df = pd.DataFrame({
                'Date': future_dates,
                'Pred_Close_USD': predictions_usd
            })
            
            # Get historical data for visualization (last 30 days) - Close only
            historical_dates = df_model['Date'].iloc[-30:]
            historical_usd = df_values['Close'].iloc[-30:].values
            
            # Display predictions
            st.subheader(f"📈 {prediction_days}-Day Future Prediction")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(14, 7))
                
                # Historical data (USD)
                ax.plot(historical_dates, historical_usd, 
                        label='Actual (30 last days)', 
                        color='#0066CC', linewidth=2.5, marker='o', markersize=4)
                
                # Connection line between last actual and prediction
                ax.plot([historical_dates.iloc[-1], future_dates[0]], 
                        [historical_usd[-1], predictions_usd[0]],
                        color='#9933FF', linewidth=2, linestyle='--', alpha=0.6)
                
                # Future predictions (USD)
                ax.plot(future_dates, predictions_usd, 
                        label=f'Predicted {prediction_days} Day', 
                        color='#9933FF', linewidth=2.5, marker='s', 
                        markersize=8, linestyle='')
                
                # Mark end_date
                end_date_dt = pd.to_datetime(original_end_date)
                ax.axvline(x=end_date_dt, color='gray', 
                           linestyle=':', linewidth=2, alpha=0.7, 
                           label=f'End Date ({end_date_dt.date()})')
                
                ax.set_title(f'Bitcoin Close Price Prediction {prediction_days} Day Ahead (USD)', 
                             fontsize=16, fontweight='bold')
                ax.set_xlabel('Date', fontsize=13)
                ax.set_ylabel('Price (USD)', fontsize=13)
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Currency formatter
                from matplotlib.ticker import FuncFormatter
                def currency_formatter(x, p):
                    return f'${x:,.0f}'
                ax.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.dataframe(prediction_df.style.format({
                    'Pred_Close_USD': '${:,.2f}'
                }), height=400)
                
                # Download button
                csv = prediction_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Prediction (CSV)",
                    data=csv,
                    file_name=f"prediction_{prediction_days}_day.csv",
                    mime="text/csv"
                )
            
            # Price change analysis (USD) - for Close
            st.subheader("📊 Price Change Analysis (USD)")
            last_actual_usd = historical_usd[-1]
            first_pred_usd = predictions_usd[0]
            
            change_1day = ((first_pred_usd - last_actual_usd) / last_actual_usd) * 100
            
            col1, col2 = st.columns(2)
            col1.metric("Current Price", f"${last_actual_usd:,.2f}")
            col2.metric("Prediction +1 Day", f"${first_pred_usd:,.2f}", f"{change_1day:+.2f}%")
            
            # Summary
            st.markdown("---")
            st.success(f"✅ Prediction complete!")
            st.write(f"- **Window size used**: {window_size} days")
            st.write(f"- **Original data range**: {historical_dates.iloc[0].date()} to {historical_dates.iloc[-1].date()}")
            st.write(f"- **End date input**: {original_end_date}")
            st.write(f"- **Prediction start**: {prediction_start_date.date()} (end_date + 1)")
            st.write(f"- **Prediction end**: {future_dates[-1].date()}")

if __name__ == "__main__":
    main()