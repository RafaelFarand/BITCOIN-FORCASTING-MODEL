# training.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import json
import shutil
from itertools import product

# BERT & Transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# Preprocessing & Metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Data
import yfinance as yf
import joblib

# Set page config
st.set_page_config(
    page_title="Bitcoin Price Prediction Training",
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

# Set random seed
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# LSTM MODEL DEFINITION
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)  # Output only Close
    
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
        st.info("Sentiment columns already exist. Skipping BERT processing.")
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
        x = data[i:(i + seq_length), :]  # All features
        y = data[i + seq_length, 0]  # Only Close
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model_with_config(config, scaled_data, device, num_features, max_epochs=50):
    window_size = config['window_size']
    hidden_size = config['hidden_size']
    num_layers = config['num_layers']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    
    X, y = create_sequences_for_lstm(scaled_data, window_size, num_features)
    
    train_split = int(len(X) * 0.8)
    
    X_train = X[:train_split]
    y_train = y[:train_split]
    X_val = X[train_split:]
    y_val = y[train_split:]
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device).unsqueeze(1)  # Make [batch, 1]
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device).unsqueeze(1)  # Make [batch, 1]
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    counter = 0
    best_model_state = None
    
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        for seqs, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seqs)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)
        
        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_val_tensor)
            val_loss = criterion(y_pred_val, y_val_tensor).item()
            val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    final_val_loss = min(val_losses)
    
    return {
        'model': model,
        'val_loss': final_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'X_train_tensor': X_train_tensor,
        'y_train_tensor': y_train_tensor,
        'X_val_tensor': X_val_tensor,
        'y_val_tensor': y_val_tensor,
        'train_split': train_split,
        'val_split': len(X) - train_split
    }

def inverse_transform_predictions(scaler, predictions, num_features):
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    temp = np.zeros((len(predictions), num_features))
    temp[:, 0] = predictions.squeeze()  # Only Close
    temp[:, 1] = scaler.inverse_transform(np.zeros((1, num_features)))[0, 1]  # Dummy for sentiment
    original_scale = scaler.inverse_transform(temp)[:, 0]
    return original_scale

def main():
    st.markdown('<h1 class="main-header">₿ Bitcoin Price Prediction Training</h1>', unsafe_allow_html=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.sidebar.info(f"Device: {device}")
    
    st.sidebar.header("Configuration")
    uploaded_file = st.sidebar.file_uploader("Upload News CSV", type=['csv'])
    st.sidebar.info("File must have 'Text' and 'Date' columns.")
    
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = col2.date_input("End Date", value=pd.to_datetime("2024-12-31"))
    
    st.sidebar.subheader("Tuning Parameter Ranges")
    window_sizes = st.sidebar.multiselect("Window Sizes", [7, 14, 30, 45, 60], default=[7, 14, 30, 45, 60])
    hidden_sizes = st.sidebar.multiselect("Hidden Sizes", [32, 64, 128], default=[32, 64, 128])
    num_layers_options = st.sidebar.multiselect("Num Layers", [1, 2, 3], default=[1, 2, 3])
    batch_sizes = st.sidebar.multiselect("Batch Sizes", [16, 32, 64, 128], default=[16, 32, 64, 128])
    learning_rates = st.sidebar.multiselect("Learning Rates", [0.0001, 0.001, 0.01], default=[0.001, 0.01])
    
    total_combinations = len(window_sizes) * len(hidden_sizes) * len(num_layers_options) * len(batch_sizes) * len(learning_rates)
    st.sidebar.warning(f"Total combinations: {total_combinations}")
    
    train_button = st.sidebar.button("Start Training", type="primary")
    
    tab1, tab2 = st.tabs(["Data", "Training"])
    
    if train_button:
        if uploaded_file is None:
            st.error("Please upload the news CSV file first!")
            return
        
        with tab1:
            st.header("Data Preparation")
            
            st.subheader("1. Sentiment Analysis")
            with st.spinner("Loading BERT model..."):
                pipe = load_bert_model()
            
            df_berita = pd.read_csv(uploaded_file)
            st.write(f"Total news: {len(df_berita)}")
            
            with st.spinner("Analyzing sentiment (or skipping if exists)..."):
                df_berita = analyze_sentiment(df_berita, pipe)
            
            st.success("Sentiment processing complete!")
            
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
                st.error("No Bitcoin data found for this date range!")
                return
            
            st.success(f"Data downloaded: {len(df_price)} days")
            
            st.line_chart(df_price.set_index('Date')['Close'])
            
            # Merge with sentiment
            if 'Date' in df_berita.columns:
                df_berita['Date'] = pd.to_datetime(df_berita['Date'])
                df_sentiment_daily = df_berita.groupby('Date').agg({
                    'sentiment_numeric': 'mean'
                }).reset_index()
                df_sentiment_daily.columns = ['Date', 'sentiment_mean']
                
                df_combined = pd.merge(df_price, df_sentiment_daily, on='Date', how='left')
                df_combined['sentiment_mean'].fillna(0, inplace=True)
                
                features = ['Close', 'sentiment_mean']
                st.success("Sentiment data merged!")
            else:
                st.error("News data missing 'Date' column!")
                return
            
            df_model = df_combined[features + ['Date']].dropna()
            st.subheader("Sample Fusion Data (Close Price + Sentiment)")
            st.dataframe(df_combined)
            st.write(f"Total features: {len(features)}")
            st.write(f"Data ready: {len(df_model)} days")
            
            # Store in session state
            st.session_state['df_model'] = df_model
            st.session_state['features'] = features
        
        with tab2:
            st.header("Model Training")
            
            if 'df_model' not in st.session_state:
                st.error("Please process data first!")
                return
            
            df_model = st.session_state['df_model']
            features = st.session_state['features']
            
            df_values = df_model[features].copy()
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df_values)
            
            st.write(f"Total Data: {len(df_values)} days")
            st.write(f"Training (80%): ~{int(len(df_values) * 0.8)} days")
            st.write(f"Validation (20%): ~{int(len(df_values) * 0.2)} days")
            
            st.subheader("Hyperparameter Tuning")
            st.info(f"Trying {total_combinations} combinations...")
            param_grid = list(product(window_sizes, hidden_sizes, num_layers_options, batch_sizes, learning_rates))
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_placeholder = st.empty()
            
            for idx, (ws, hs, nl, bs, lr) in enumerate(param_grid):
                config = {
                    'window_size': ws,
                    'hidden_size': hs,
                    'num_layers': nl,
                    'batch_size': bs,
                    'learning_rate': lr
                }
                status_text.text(f"Testing config {idx+1}/{total_combinations}: {config}")
                try:
                    result = train_model_with_config(
                        config, scaled_data, device, len(features)
                    )
                    results.append({
                        **config,
                        'val_loss': result['val_loss'],
                        'train_losses': result['train_losses'],
                        'val_losses': result['val_losses'],
                        'model': result['model'],
                        'X_train_tensor': result['X_train_tensor'],
                        'y_train_tensor': result['y_train_tensor'],
                        'X_val_tensor': result['X_val_tensor'],
                        'y_val_tensor': result['y_val_tensor'],
                        'train_split': result['train_split'],
                        'val_split': result['val_split']
                    })
                    results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['model', 'train_losses', 'val_losses', 'X_train_tensor', 'y_train_tensor', 'X_val_tensor', 'y_val_tensor', 'train_split', 'val_split']} for r in results])
                    results_df = results_df.sort_values('val_loss')
                    results_placeholder.dataframe(results_df.head(10).style.highlight_min(subset=['val_loss'], color='lightgreen'))
                except Exception as e:
                    st.warning(f"Error with config {config}: {str(e)}")
                progress_bar.progress((idx + 1) / total_combinations)
            
            best_result = min(results, key=lambda x: x['val_loss'])
            best_config = {k: v for k, v in best_result.items() if k not in ['val_loss', 'train_losses', 'val_losses', 'model', 'X_train_tensor', 'y_train_tensor', 'X_val_tensor', 'y_val_tensor', 'train_split', 'val_split']}
            
            st.success("Hyperparameter tuning complete!")
            st.markdown('<div class="best-config">', unsafe_allow_html=True)
            st.subheader("🏆 Best Configuration Found")
            st.json(best_config)
            st.metric("Best Validation Loss (MAE)", f"{best_result['val_loss']:.6f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Plot training history
            st.subheader("Training History - Best Model")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(best_result['train_losses'], label='Training Loss', color='#2E86AB')
            ax.plot(best_result['val_losses'], label='Validation Loss', color='#F18F01')
            ax.set_title('Training & Validation Loss Over Time')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (MAE)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Store results
            st.session_state['best_model'] = best_result['model']
            st.session_state['best_config'] = best_config
            st.session_state['scaler'] = scaler
            st.session_state['X_train_tensor'] = best_result['X_train_tensor']
            st.session_state['y_train_tensor'] = best_result['y_train_tensor']
            st.session_state['X_val_tensor'] = best_result['X_val_tensor']
            st.session_state['y_val_tensor'] = best_result['y_val_tensor']
            st.session_state['train_split'] = best_result['train_split']
            st.session_state['device'] = device
            st.session_state['num_features'] = len(features)
            
            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            models_dir = 'models'
            os.makedirs(models_dir, exist_ok=True)
            
            model_path = os.path.join(models_dir, f'best_model_{timestamp}.pth')
            scaler_path = os.path.join(models_dir, f'scaler_{timestamp}.pkl')
            config_path = os.path.join(models_dir, f'config_{timestamp}.json')
            
            torch.save(best_result['model'].state_dict(), model_path)
            joblib.dump(scaler, scaler_path)
            with open(config_path, 'w') as f:
                json.dump(best_config, f)
            
            # Copy to latest
            shutil.copy(model_path, os.path.join(models_dir, 'best_model_latest.pth'))
            shutil.copy(scaler_path, os.path.join(models_dir, 'scaler_latest.pkl'))
            shutil.copy(config_path, os.path.join(models_dir, 'config_latest.json'))
            
            st.success(f"Model saved to {models_dir}!")

if __name__ == "__main__":
    main()