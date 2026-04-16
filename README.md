Bitcoin Price Prediction with Sentiment Analysis**

judul Proyek
**Bitcoin Close Price Prediction Using LSTM + CryptoBERT Sentiment Analysis**

#### **Deskripsi**
Proyek ini bertujuan memprediksi harga penutupan (**Close Price**) Bitcoin satu hari ke depan menggunakan model **LSTM** yang dikombinasikan dengan analisis sentimen dari teks berita cryptocurrency.

Model menggabungkan data historis harga Bitcoin dengan fitur sentimen harian yang dihasilkan oleh model **CryptoBERT**, sehingga menghasilkan prediksi yang mempertimbangkan baik pola harga maupun sentimen pasar.

#### **Arsitektur Model**
- **Model Utama**: Long Short-Term Memory (LSTM)
- **Sentiment Analyzer**: CryptoBERT (`ElKulako/cryptobert`)
- **Jenis Prediksi**: One-step ahead forecasting (1 hari ke depan)
- **Fitur Input**: 
  - Close Price (harga penutupan)
  - Sentiment Mean (rata-rata sentimen harian: +1 Bullish, 0 Neutral, -1 Bearish)

#### **Dataset**
1. **Data Harga Bitcoin**  
   - Diambil secara real-time menggunakan library `yfinance` (ticker: BTC-USD)  
   - Periode: Sesuai rentang tanggal yang dipilih user (contoh: 2025-01-01 hingga 2025-12-31)

2. **Data Teks Berita**  
   - File CSV berisi kolom `Date` dan `Text`  
   - Diproses untuk menghasilkan sentimen harian menggunakan CryptoBERT

#### **Hyperparameter Model (Best Configuration)**
Model menggunakan konfigurasi terbaik hasil hyperparameter tuning:

- **Window Size**        : 7, 14, 30, 45, 60
- **Hidden Size**        : 32, 64, 128
- **Number of Layers**   : 1, 2, 3
- **Dropout**            : 0.2
- **Batch Size**         : 16, 32, 64, 128
- **Learning Rate**      : 0.0001, 0.001, 0.01
- **Optimizer**          : adam

setelah training didapat hasil paling rendah nilai errornya, dengan konfigurasi :
	window_size 7
  hidden_size	64
  num_layers 1
  batch_size 128
  learning_rate 0.01
dan di simpan ke dalam .json
Catatan: Nilai `hidden_size` dan `num_layers` disimpan dan diambil kembali dari file `config_latest.json` pada saat testing

#### **Preprocessing**
- Data harga dan sentimen digabungkan berdasarkan tanggal
- Missing sentiment diisi dengan nilai 0
- Seluruh fitur discaling menggunakan `MinMaxScaler`
- Sequence dibuat dengan sliding window sepanjang 60 hari
- Target prediksi: Harga Close hari berikutnya

#### **Metrik Evaluasi**
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- **RMSE%** = (RMSE / rata-rata harga aktual) × 100%

#### **Teknologi yang Digunakan**
- **Deep Learning**: PyTorch
- **NLP**: Hugging Face Transformers (CryptoBERT)
- **Web Interface**: Streamlit
- **Data Processing**: pandas, NumPy, scikit-learn
- **Data Source**: yfinance
- **Model Saving**: joblib & torch

#### **Fitur Aplikasi (Streamlit)**
- Upload file CSV berita untuk analisis sentimen
- Download data harga Bitcoin otomatis sesuai periode
- Evaluasi model pada data test
- Prediksi harga Bitcoin 1 hari ke depan
- Visualisasi grafik Actual vs Predicted dan Future Prediction
- Metrik evaluasi dalam USD dan persentase

#### **Tujuan Proyek**
Membangun model prediksi harga Bitcoin yang mengintegrasikan informasi sentimen pasar untuk meningkatkan kualitas prediksi, serta menyediakan antarmuka yang user-friendly untuk melakukan inference dan evaluasi.


